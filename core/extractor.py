import os
from contextlib import contextmanager
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.hooks import ActivationHook
from core.layer_config import LAYER_CONFIG


@contextmanager
def _null_context():
    yield


def _resolve_load_dtype(hidden_dim: int, *, on_cpu: bool) -> torch.dtype:
    if hidden_dim < 64 or on_cpu:
        return torch.float32
    return torch.float16


class ActivationExtractor:
    """Loads HF causal LM, registers hooks, supports encode vs autoregressive extract."""

    def __init__(self, model_name: str, device: str = "auto"):
        config = LAYER_CONFIG.get(model_name)
        if config is None:
            raise ValueError(
                f"Model {model_name} not in layer config. Add it to core/layer_config.py"
            )
        self.model_name = model_name
        self.layer_indices = list(config["layers"])
        self.hidden_dim = config["hidden_dim"]
        self.early_layer = self.layer_indices[0]
        self.late_layer = self.layer_indices[-1]

        use_fast = os.environ.get("ANIMA_SLOW_TOKENIZER", "0") != "1"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
        except Exception:
            if use_fast:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            else:
                raise
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cuda_ok = torch.cuda.is_available()
        force_cpu = os.environ.get("ANIMA_FORCE_CPU") == "1"
        on_cpu = force_cpu or device == "cpu" or (device == "auto" and not cuda_ok)
        device_map = "cpu" if on_cpu else device

        dtype = _resolve_load_dtype(config["hidden_dim"], on_cpu=on_cpu)
        load_kwargs: dict = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            # Prefer safetensors so older torch (<2.6) can load Hub weights without CVE-2025-32434 block
            "use_safetensors": True,
        }
        self._load_8bit = False
        if os.environ.get("ANIMA_LOAD_8BIT") == "1" and cuda_ok and config["hidden_dim"] >= 2048:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["device_map"] = "auto"
                load_kwargs.pop("torch_dtype", None)
                load_kwargs.pop("dtype", None)
                self._load_8bit = True
            except ImportError:
                pass

        # 8-bit + output_attentions is a known NaN source on some Qwen/Mistral stacks.
        # Default: attentions on; set ANIMA_OUTPUT_ATTENTIONS=0 to force off.
        # Auto-disable attentions under 8-bit unless explicitly forced on.
        attn_env = os.environ.get("ANIMA_OUTPUT_ATTENTIONS", "").strip()
        if attn_env == "1":
            self._output_attentions = True
        elif attn_env == "0":
            self._output_attentions = False
        else:
            self._output_attentions = not self._load_8bit

        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception:
            load_kwargs.pop("use_safetensors", None)
            # Older transformers want dtype= instead of torch_dtype=
            if "torch_dtype" in load_kwargs:
                load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if os.environ.get("ANIMA_LOAD_DYNAMIC_INT8") == "1" and on_cpu:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.model.eval()
        self.hook = ActivationHook(self.model, self.layer_indices)

    @staticmethod
    def _as_finite(x: float, default: float = 0.5) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        if v != v or v in (float("inf"), float("-inf")):  # NaN/Inf
            return default
        return v

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        logits_f = logits.float()
        if not torch.isfinite(logits_f).all():
            return float("nan")
        probs = torch.softmax(logits_f, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        max_entropy = torch.log(
            torch.tensor(float(logits_f.shape[-1]), dtype=torch.float32, device=logits_f.device)
        )
        return self._as_finite((entropy / max_entropy).item(), default=float("nan"))

    def _compute_logit_gap(self, logits: torch.Tensor) -> float:
        logits_f = logits.float()
        if not torch.isfinite(logits_f).all():
            return float("nan")
        top2 = torch.topk(logits_f, 2).values
        gap = (top2[0] - top2[1]).item()
        return self._as_finite(1.0 / (1.0 + max(gap, 0.0)), default=float("nan"))

    def _compute_attention_entropy(self, attentions) -> float:
        if attentions is None:
            return float("nan")
        last_layer_attn = None
        for layer_attn in reversed(attentions):
            if layer_attn is not None:
                last_layer_attn = layer_attn
                break
        if last_layer_attn is None:
            return float("nan")
        last_token_attn = last_layer_attn[0, :, -1, :].float()
        if not torch.isfinite(last_token_attn).all():
            return float("nan")
        # Renormalize in case of numerical drift under quantization.
        mass = last_token_attn.sum(-1, keepdim=True).clamp_min(1e-9)
        last_token_attn = last_token_attn / mass
        entropy = -(last_token_attn * torch.log(last_token_attn + 1e-9)).sum(-1).mean()
        max_entropy = torch.log(
            torch.tensor(float(last_token_attn.shape[-1]), dtype=torch.float32, device=last_token_attn.device)
        )
        return self._as_finite((entropy / max_entropy).item(), default=float("nan"))

    def _fuse_uncertainty(self, entropy: float, logit_gap: float, attn_entropy: float) -> float:
        """Weighted fuse that drops NaN channels and renormalizes remaining weights."""
        parts = [
            (0.35, entropy),
            (0.35, logit_gap),
            (0.30, attn_entropy),
        ]
        usable = [(w, v) for w, v in parts if v == v and v not in (float("inf"), float("-inf"))]
        if not usable:
            return float("nan")
        w_sum = sum(w for w, _ in usable)
        fused = sum((w / w_sum) * v for w, v in usable)
        return round(self._as_finite(fused, default=float("nan")), 4)

    def _uncertainty_pack(
        self, logits: torch.Tensor, attentions
    ) -> dict[str, float]:
        entropy = self._compute_entropy(logits)
        logit_gap_unc = self._compute_logit_gap(logits)
        attn_entropy = (
            self._compute_attention_entropy(attentions)
            if self._output_attentions
            else float("nan")
        )
        fused = self._fuse_uncertainty(entropy, logit_gap_unc, attn_entropy)
        return {
            "entropy": round(self._as_finite(entropy, default=float("nan")), 4)
            if entropy == entropy
            else float("nan"),
            "logit_gap": round(self._as_finite(logit_gap_unc, default=float("nan")), 4)
            if logit_gap_unc == logit_gap_unc
            else float("nan"),
            "attn_entropy": round(self._as_finite(attn_entropy, default=float("nan")), 4)
            if attn_entropy == attn_entropy
            else float("nan"),
            "fused": fused,
            "attn_enabled": bool(self._output_attentions),
            "load_8bit": bool(self._load_8bit),
        }

    def cleanup(self):
        self.hook.remove()

    def encode_sequence(
        self,
        text: str,
        max_length: Optional[int] = None,
        *,
        apply_chat_template: bool = True,
    ) -> list:
        """
        Single forward pass over tokenized text (stimulus encoding path).
        Returns one entry per input token position with layer activations and uncertainty at that step.
        Instruct models receive chat-template wrapping when configured in LAYER_CONFIG
        unless ``apply_chat_template=False`` (caller already formatted).
        """
        from core.prompt_format import format_user_text

        if apply_chat_template:
            text = format_user_text(self.tokenizer, text, model_name=self.model_name)
        max_len = max_length or getattr(self.tokenizer, "model_max_length", 1024)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        ).to(self.model.device)

        self.hook.clear()
        results: list = []

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=bool(self._output_attentions),
                use_cache=False,
            )

        seq_len = inputs.input_ids.shape[1]
        for pos in range(seq_len):
            logits = outputs.logits[0, pos, :]
            unc = self._uncertainty_pack(logits, outputs.attentions)

            tid = int(inputs.input_ids[0, pos].item())
            token_text = self.tokenizer.decode([tid])

            acts = {
                idx: self.hook.all_positions(idx)[pos].clone()
                for idx in self.layer_indices
            }

            results.append(
                {
                    "token_index": pos,
                    "token_id": tid,
                    "token_text": token_text,
                    "activations": acts,
                    "logits": logits.detach().cpu(),
                    "uncertainty_signals": {
                        "entropy": unc["entropy"],
                        "logit_gap": unc["logit_gap"],
                        "attn_entropy": unc["attn_entropy"],
                        "fused": unc["fused"],
                    },
                }
            )

        return results

    def extract_iter(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        *,
        probe=None,
        intervention_mode: str = "none",
    ):
        """Yield one result dict per generated token (for WebSocket streaming)."""
        from core.intervention import dampen_residual_step, should_dampen
        from core.prompt_format import format_user_text

        prompt = format_user_text(self.tokenizer, prompt, model_name=self.model_name)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        past_key_values = None
        recent_valences: list[float] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                self.hook.clear()
                if past_key_values is None:
                    model_inputs = {**inputs}
                else:
                    model_inputs = {
                        "input_ids": inputs["input_ids"][:, -1:],
                        "past_key_values": past_key_values,
                    }

                use_dampen = (
                    intervention_mode == "dampen"
                    and probe is not None
                    and should_dampen(recent_valences)
                )
                ctx = (
                    dampen_residual_step(self, probe)
                    if use_dampen
                    else _null_context()
                )
                with ctx:
                    outputs = self.model(
                        **model_inputs,
                        output_attentions=bool(self._output_attentions),
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1, :]
                unc = self._uncertainty_pack(logits, outputs.attentions)

                token_activations = {
                    idx: self.hook.last_token(idx).clone() for idx in self.layer_indices
                }
                if probe is not None:
                    recent_valences.append(float(probe.predict(token_activations)["valence"]))

                next_token_id = logits.argmax().unsqueeze(0).unsqueeze(0)
                tid = int(next_token_id.item())

                yield {
                    "token_id": tid,
                    "token_text": self.tokenizer.decode([tid]),
                    "activations": token_activations,
                    "logits": logits.detach().cpu(),
                    "uncertainty_signals": {
                        "entropy": unc["entropy"],
                        "logit_gap": unc["logit_gap"],
                        "attn_entropy": unc["attn_entropy"],
                        "fused": unc["fused"],
                    },
                }

                inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_id], dim=1)
                eos = self.tokenizer.eos_token_id
                if eos is not None and tid == eos:
                    break

    def extract(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        *,
        probe=None,
        intervention_mode: str = "none",
    ) -> list:
        """Autoregressive token generation with per-step hooks (does not remove hooks)."""
        return list(
            self.extract_iter(
                prompt,
                max_new_tokens,
                probe=probe,
                intervention_mode=intervention_mode,
            )
        )
