import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.hooks import ActivationHook
from core.layer_config import LAYER_CONFIG


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
            "dtype": dtype,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if os.environ.get("ANIMA_LOAD_8BIT") == "1" and cuda_ok and config["hidden_dim"] >= 2048:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["device_map"] = "auto"
                load_kwargs.pop("torch_dtype", None)
            except ImportError:
                pass

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self.hook = ActivationHook(self.model, self.layer_indices)

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum()
        max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float))
        return (entropy / max_entropy).item()

    def _compute_logit_gap(self, logits: torch.Tensor) -> float:
        top2 = torch.topk(logits.float(), 2).values
        gap = (top2[0] - top2[1]).item()
        return 1.0 / (1.0 + max(gap, 0.0))

    def _compute_attention_entropy(self, attentions) -> float:
        if attentions is None:
            return 0.5
        last_layer_attn = None
        for layer_attn in reversed(attentions):
            if layer_attn is not None:
                last_layer_attn = layer_attn
                break
        if last_layer_attn is None:
            return 0.5
        last_token_attn = last_layer_attn[0, :, -1, :]
        entropy = -(last_token_attn * torch.log(last_token_attn + 1e-9)).sum(-1).mean()
        max_entropy = torch.log(torch.tensor(last_token_attn.shape[-1], dtype=torch.float))
        return (entropy / max_entropy).item()

    def _fuse_uncertainty(self, entropy: float, logit_gap: float, attn_entropy: float) -> float:
        return round(0.35 * entropy + 0.35 * logit_gap + 0.30 * attn_entropy, 4)

    def cleanup(self):
        self.hook.remove()

    def encode_sequence(self, text: str, max_length: Optional[int] = None) -> list:
        """
        Single forward pass over tokenized text (stimulus encoding path).
        Returns one entry per input token position with layer activations and uncertainty at that step.
        """
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
                output_attentions=True,
                use_cache=False,
            )

        seq_len = inputs.input_ids.shape[1]
        for pos in range(seq_len):
            logits = outputs.logits[0, pos, :]
            entropy = self._compute_entropy(logits)
            logit_gap_unc = self._compute_logit_gap(logits)
            attn_entropy = self._compute_attention_entropy(outputs.attentions)
            fused = self._fuse_uncertainty(entropy, logit_gap_unc, attn_entropy)

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
                        "entropy": round(entropy, 4),
                        "logit_gap": round(logit_gap_unc, 4),
                        "attn_entropy": round(attn_entropy, 4),
                        "fused": fused,
                    },
                }
            )

        return results

    def extract_iter(self, prompt: str, max_new_tokens: int = 200):
        """Yield one result dict per generated token (for WebSocket streaming)."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        past_key_values = None

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
                outputs = self.model(
                    **model_inputs,
                    output_attentions=True,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[0, -1, :]

                entropy = self._compute_entropy(logits)
                logit_gap_unc = self._compute_logit_gap(logits)
                attn_entropy = self._compute_attention_entropy(outputs.attentions)
                uncertainty_fused = self._fuse_uncertainty(entropy, logit_gap_unc, attn_entropy)

                token_activations = {
                    idx: self.hook.last_token(idx).clone() for idx in self.layer_indices
                }

                next_token_id = logits.argmax().unsqueeze(0).unsqueeze(0)
                tid = int(next_token_id.item())

                yield {
                    "token_id": tid,
                    "token_text": self.tokenizer.decode([tid]),
                    "activations": token_activations,
                    "logits": logits.detach().cpu(),
                    "uncertainty_signals": {
                        "entropy": round(entropy, 4),
                        "logit_gap": round(logit_gap_unc, 4),
                        "attn_entropy": round(attn_entropy, 4),
                        "fused": uncertainty_fused,
                    },
                }

                inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token_id], dim=1)
                eos = self.tokenizer.eos_token_id
                if eos is not None and tid == eos:
                    break

    def extract(self, prompt: str, max_new_tokens: int = 200) -> list:
        """Autoregressive token generation with per-step hooks (does not remove hooks)."""
        return list(self.extract_iter(prompt, max_new_tokens))
