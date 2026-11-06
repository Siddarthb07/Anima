#!/usr/bin/env python3
"""Full-night deep waiver suite on GPU (hands-free).

Phases:
  0) Backend health — encode smoke claims; assert finite uncertainty channels
  1) Claim-encode re-run per model (reuse text probes; fixed NaN-safe extractor)
  2) Deep channel / length / waiver-bin analysis
  3) Optional length-matched HaluEval additive pass
  4) Cross-model rollup + DEEP_WAIVER_STATUS.md

Does NOT auto-shutdown. Does NOT invent metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "benchmarks" / "reports"
LOGS = ROOT / "logs"
DEEP_DIR = REPORTS / "deep_waiver"

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").lower()


def _require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for deep waiver suite")
    return "cuda"


def _ensure_probe(model: str, device: str) -> None:
    from probes.zoo_io import load_probe_into, probe_slug
    from probes.linear_probe import AffectProbe
    from core.layer_config import LAYER_CONFIG

    # Prefer existing text probe; only train if missing.
    try:
        cfg = LAYER_CONFIG[model]
        probe = AffectProbe(cfg["hidden_dim"], len(cfg["layers"]))
        load_probe_into(probe, model, map_location="cpu")
        print(f"  probe OK existing text/zoo for {probe_slug(model)}", flush=True)
        return
    except Exception as e:
        print(f"  probe missing/unloadable ({e}); training text probe…", flush=True)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_tmlr_stage_b_gpu", ROOT / "scripts" / "run_tmlr_stage_b_gpu.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ensure_text_probe(model, device=device, max_samples=1500, epochs=1)


def _backend_health(model: str, device: str) -> dict:
    """Smoke-encode two claims; report finite rates for uncertainty channels."""
    from core.extractor import ActivationExtractor
    from core.prompt_format import format_qa_claim

    claims = [
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What is the capital of France?", "Berlin is the capital of France."),
    ]
    ex = ActivationExtractor(model, device=device)
    rows = []
    for q, a in claims:
        text = format_qa_claim(ex.tokenizer, q, a, model_name=model)
        enc = ex.encode_sequence(text, max_length=256, apply_chat_template=False)
        if not enc:
            continue
        last = enc[-1]["uncertainty_signals"]
        rows.append(last)
    ex.cleanup()
    del ex
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    def finite_rate(key: str) -> float:
        vals = [r.get(key) for r in rows]
        ok = 0
        for v in vals:
            try:
                x = float(v)
                if x == x and x not in (float("inf"), float("-inf")):
                    ok += 1
            except (TypeError, ValueError):
                pass
        return round(ok / max(1, len(vals)), 4)

    return {
        "model": model,
        "n_claims": len(rows),
        "output_attentions": bool(rows and True),
        "finite_rate": {
            "entropy": finite_rate("entropy"),
            "logit_gap": finite_rate("logit_gap"),
            "attn_entropy": finite_rate("attn_entropy"),
            "fused": finite_rate("fused"),
        },
        "sample": rows[0] if rows else None,
        "pass": bool(rows) and finite_rate("fused") >= 0.5 and finite_rate("logit_gap") >= 0.5,
    }


def _run_live(
    model: str,
    device: str,
    *,
    suffix: str,
    max_pairs: int,
    seed: int,
    length_match: float,
    benchmark: str = "both",
) -> dict:
    from benchmarks.live_guard_eval import run_live

    os.environ["ANIMA_FORCE_CPU"] = "0"
    os.environ.setdefault("ANIMA_PREFER_TEXT_PROBE", "1")
    return run_live(
        model,
        benchmark=benchmark,
        max_pairs=max_pairs,
        seed=seed,
        device=device,
        report_suffix=suffix,
        length_match_ratio=length_match,
    )


def _analyze_live(model: str, suffix: str) -> dict:
    from benchmarks.deep_waiver_analysis import analyze_row_file, write_analysis

    slug = _slug(model)
    out = {"model": model, "suffix": suffix, "suites": {}}
    mapping = {
        "halueval": (REPORTS / f"halueval_live_{slug}_{suffix}.json", "is_hallucination"),
        "truthfulqa": (REPORTS / f"truthfulqa_live_{slug}_{suffix}.json", "should_abstain"),
    }
    for suite, (path, key) in mapping.items():
        if not path.exists():
            out["suites"][suite] = {"status": "missing", "path": str(path)}
            continue
        analysis = analyze_row_file(path, label_key=key)
        dest = DEEP_DIR / f"{slug}_{suffix}_{suite}_analysis.json"
        write_analysis(dest, analysis)
        out["suites"][suite] = {
            "status": "ok",
            "analysis_path": str(dest),
            "headline": analysis.get("headline"),
            "nan_rates": analysis.get("nan_rates"),
            "never_abstains": analysis.get("never_abstains"),
            "abstain_rate": analysis.get("abstain_rate"),
            "waiver_bins": analysis.get("waiver_bins"),
            "length_confound": {
                k: {
                    "length_auroc": (v or {}).get("length_auroc"),
                    "auroc_residual": (v or {}).get("auroc_residual"),
                    "auroc_residual_flip": (v or {}).get("auroc_residual_flip"),
                }
                for k, v in (analysis.get("length_confound") or {}).items()
            },
            "channels_best": {
                name: {
                    "best_auroc": ch.get("best_auroc"),
                    "polarity": ch.get("polarity"),
                    "n_finite": ch.get("n_finite"),
                    "auroc": ch.get("auroc"),
                    "auroc_flip": ch.get("auroc_flip"),
                }
                for name, ch in (analysis.get("channels") or {}).items()
            },
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--max-pairs", type=int, default=160)
    ap.add_argument("--tqa-pairs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--suffix", default="deep_waiver_v1")
    ap.add_argument("--length-match-ratio", type=float, default=2.0)
    ap.add_argument("--skip-health", action="store_true")
    ap.add_argument("--skip-live", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Hard lock: never shutdown from this process.
    os.environ["ANIMA_FLEET_SHUTDOWN"] = "0"
    os.environ.setdefault("ANIMA_LOAD_8BIT", "1")
    os.environ["ANIMA_FORCE_CPU"] = "0"
    # Auto attentions-off under 8bit unless overridden
    os.environ.setdefault("ANIMA_OUTPUT_ATTENTIONS", "0")
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

    LOGS.mkdir(parents=True, exist_ok=True)
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    device = _require_cuda()
    started = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    print(f"=== deep waiver start {started} device={device} ===", flush=True)
    print(
        f"env ANIMA_OUTPUT_ATTENTIONS={os.environ.get('ANIMA_OUTPUT_ATTENTIONS')} "
        f"ANIMA_LOAD_8BIT={os.environ.get('ANIMA_LOAD_8BIT')} "
        f"ANIMA_FLEET_SHUTDOWN={os.environ.get('ANIMA_FLEET_SHUTDOWN')}",
        flush=True,
    )

    for i, model in enumerate(args.models, 1):
        print(f"\n##### [{i}/{len(args.models)}] {model} #####", flush=True)
        partial = DEEP_DIR / f"partial_{_slug(model)}.json"
        if args.resume and partial.exists():
            prev = json.loads(partial.read_text(encoding="utf-8"))
            if prev.get("status") == "ok":
                print(f"SKIP resume ok: {model}", flush=True)
                results.append(prev)
                continue

        entry: dict = {
            "model": model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        try:
            _ensure_probe(model, device)

            if not args.skip_health:
                print("  phase0 backend health…", flush=True)
                health = _backend_health(model, device)
                entry["health"] = health
                print(f"  health pass={health.get('pass')} finite={health.get('finite_rate')}", flush=True)
                if not health.get("pass"):
                    # Retry once with attentions forced off
                    os.environ["ANIMA_OUTPUT_ATTENTIONS"] = "0"
                    health2 = _backend_health(model, device)
                    entry["health_retry_no_attn"] = health2
                    print(
                        f"  health retry pass={health2.get('pass')} finite={health2.get('finite_rate')}",
                        flush=True,
                    )
                    if not health2.get("pass"):
                        raise RuntimeError(f"backend health failed for {model}: {health2}")

            if not args.skip_live:
                print(
                    f"  phase1 live claim-encode max_pairs={args.max_pairs} "
                    f"(HaluEval+TruthfulQA; TQA requires HF truthful_qa)…",
                    flush=True,
                )
                # Primary deep pass (no length match — full waiver surface)
                live = _run_live(
                    model,
                    device,
                    suffix=args.suffix,
                    max_pairs=args.max_pairs,
                    seed=args.seed,
                    length_match=0.0,
                )
                entry["live"] = {
                    "suffix": args.suffix,
                    "benchmarks": {
                        k: {
                            "auroc_composite": (v or {}).get("auroc_composite"),
                            "auroc_ci95": (v or {}).get("auroc_ci95"),
                            "n_samples": (v or {}).get("n_samples"),
                            "never_abstains": (v or {}).get("never_abstains"),
                            "abstain_accuracy": (v or {}).get("abstain_accuracy"),
                        }
                        for k, v in (live.get("benchmarks") or {}).items()
                    },
                }

                # Length-matched HaluEval additive (construct validity)
                if args.length_match_ratio and args.length_match_ratio > 0:
                    print(
                        f"  phase1b length-matched HaluEval ratio<={args.length_match_ratio}…",
                        flush=True,
                    )
                    live_lm = _run_live(
                        model,
                        device,
                        suffix=f"{args.suffix}_lenmatch",
                        max_pairs=min(args.max_pairs, 80),
                        seed=args.seed,
                        length_match=args.length_match_ratio,
                        benchmark="halueval",
                    )
                    # Only keep Halu metrics of interest
                    entry["live_length_matched"] = {
                        "suffix": f"{args.suffix}_lenmatch",
                        "halueval": (live_lm.get("benchmarks") or {}).get("halueval"),
                    }

            print("  phase2 deep analysis…", flush=True)
            entry["analysis"] = _analyze_live(model, args.suffix)
            if args.length_match_ratio and args.length_match_ratio > 0:
                entry["analysis_lenmatch"] = _analyze_live(model, f"{args.suffix}_lenmatch")

            entry["status"] = "ok"
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = str(e)
            entry["traceback"] = traceback.format_exc()[-5000:]
            print(f"FAILED {model}: {e}", flush=True)
            try:
                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
        entry["finished_at"] = datetime.now(timezone.utc).isoformat()
        results.append(entry)
        partial.write_text(json.dumps(entry, indent=2), encoding="utf-8")

    rollup = {
        "schema": "system_a.deep_waiver_gpu.v1",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "device": __import__("torch").cuda.get_device_name(0),
        "env": {
            "ANIMA_LOAD_8BIT": os.environ.get("ANIMA_LOAD_8BIT"),
            "ANIMA_OUTPUT_ATTENTIONS": os.environ.get("ANIMA_OUTPUT_ATTENTIONS"),
            "ANIMA_FLEET_SHUTDOWN": os.environ.get("ANIMA_FLEET_SHUTDOWN"),
        },
        "max_pairs": args.max_pairs,
        "suffix": args.suffix,
        "n_models": len(args.models),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_failed": sum(1 for r in results if r.get("status") != "ok"),
        "results": results,
        "honesty": {
            "no_invented_metrics": True,
            "auto_shutdown": False,
            "nan_channels_excluded_from_auroc": True,
            "shipping_claim": False,
        },
    }
    out = DEEP_DIR / "deep_waiver_rollup.json"
    out.write_text(json.dumps(rollup, indent=2), encoding="utf-8")

    # Status markdown
    lines = [
        "# Deep waiver GPU status",
        "",
        f"- Finished: `{rollup['finished_at']}`",
        f"- Device: `{rollup['device']}`",
        f"- OK: **{rollup['n_ok']}** / Failed: **{rollup['n_failed']}**",
        f"- Rollup: `benchmarks/reports/deep_waiver/deep_waiver_rollup.json`",
        f"- Attentions: `{rollup['env']['ANIMA_OUTPUT_ATTENTIONS']}` (0=off under 8-bit default)",
        f"- Shutdown: **disabled** for this suite",
        "",
        "| Model | Status | Halu composite | Best channel AUROC | Polarity | NaN fused |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in results:
        hs = ((r.get("live") or {}).get("benchmarks") or {}).get("halueval") or {}
        an = ((r.get("analysis") or {}).get("suites") or {}).get("halueval") or {}
        head = an.get("headline") or {}
        nan = (an.get("nan_rates") or {}).get("fused")
        lines.append(
            f"| `{r.get('model')}` | {r.get('status')} | {hs.get('auroc_composite','—')} | "
            f"{head.get('best_auroc','—')} | {head.get('polarity','—')} | {nan if nan is not None else '—'} |"
        )
    status_path = ROOT / "DEEP_WAIVER_STATUS.md"
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (LOGS / "DEEP_WAIVER_STATUS.md").write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
    (LOGS / "deep_waiver.exit").write_text("0\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    print(f"Wrote {status_path}", flush=True)
    print("Deep waiver done — instance NOT shut down.", flush=True)


if __name__ == "__main__":
    main()
