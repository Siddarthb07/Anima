"""Run benchmarks across all registered models; write per-model manifests + rollup."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.layer_config import LAYER_CONFIG  # noqa: E402
from probes.zoo_io import checkpoint_path, probe_slug  # noqa: E402

FIXTURE = ROOT / "benchmarks" / "fixtures" / "poc_emotional_prompts.json"
REPORTS = ROOT / "benchmarks" / "reports"


def _has_text_probe(model: str) -> bool:
    slug = probe_slug(model)
    return checkpoint_path(slug, "_text").exists()


def _has_brain_probe(model: str) -> bool:
    slug = probe_slug(model)
    return checkpoint_path(slug, "_narratives_pca").exists() or checkpoint_path(slug, "default").exists()


def _tiers_for(model: str) -> str:
    parts = ["internal", "external_guard"]
    if _has_text_probe(model):
        parts.append("external_text")
    if _has_brain_probe(model):
        parts.append("external")
    return ",".join(parts)


def _collect_prompt_examples(model: str) -> list[dict]:
    """Run positive/negative/hedge prompts through extractor (no API)."""
    from core.extractor import ActivationExtractor
    from probes.linear_probe import AffectProbe
    from probes.zoo_io import load_probe_into

    prompts = json.loads(FIXTURE.read_text(encoding="utf-8"))["prompts"]
    ids = ("positive", "negative", "hedge_volatile")
    out: list[dict] = []
    try:
        ex = ActivationExtractor(model)
        probe = AffectProbe(ex.hidden_dim, len(ex.layer_indices))
        load_probe_into(probe, model)
        for item in prompts:
            if item["id"] not in ids:
                continue
            rows = list(ex.extract_iter(item["text"], max_new_tokens=24, probe=probe))
            if not rows:
                continue
            vals = [float(probe.predict(r["activations"])["valence"]) for r in rows]
            arous = [float(probe.predict(r["activations"])["arousal"]) for r in rows]
            import numpy as np

            swing = float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0
            out.append(
                {
                    "id": item["id"],
                    "prompt": item["text"],
                    "expected": item.get("expected_valence_sign"),
                    "mean_valence": round(float(np.mean(vals)), 4),
                    "mean_arousal": round(float(np.mean(arous)), 4),
                    "valence_swing": round(swing, 4),
                    "n_tokens": len(rows),
                    "sample_output": "".join(r.get("token_text", "") for r in rows[:12]),
                }
            )
            if len(vals) > 3:
                std = float(np.std(vals))
                out[-1]["stability_score"] = round(max(0.0, 1.0 - std * 2), 3)
        ex.cleanup()
    except Exception as exc:
        out.append({"id": "_error", "error": str(exc)})
    return out


def _run_benchmark(model: str, tiers: str) -> dict | None:
    env = os.environ.copy()
    env.setdefault("ANIMA_FORCE_CPU", "1")
    env.setdefault("SKIP_BRAINSCORE", "1")
    env.setdefault("NARRATIVES_ROOT", str(ROOT / "data" / "narratives_minimal"))
    cmd = [sys.executable, "-m", "benchmarks.run_all", "--model", model, "--tiers", tiers]
    print(f"  -> {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"model": model, "status": "timeout", "tiers": tiers}
    except subprocess.CalledProcessError as exc:
        return {"model": model, "status": "error", "tiers": tiers, "code": exc.returncode}
    slug = probe_slug(model)
    latest = REPORTS / f"latest_{slug}_manifest.json"
    if not latest.exists():
        latest = REPORTS / "latest_manifest.json"
    if latest.exists():
        return json.loads(latest.read_text(encoding="utf-8"))
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", help="comma-separated model ids (default: CPU-feasible)")
    p.add_argument("--include-gpu", action="store_true")
    p.add_argument("--skip-examples", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("ANIMA_FORCE_CPU", "1")
    models = list(LAYER_CONFIG.keys())
    if args.only:
        models = [m.strip() for m in args.only.split(",")]
    elif not args.include_gpu:
        models = [m for m in models if not LAYER_CONFIG[m].get("requires_gpu")]

    rollup: list[dict] = []
    for model in models:
        print(f"\n=== {model} ===", flush=True)
        cfg = LAYER_CONFIG[model]
        if cfg.get("requires_gpu") and not args.include_gpu:
            rollup.append({"model": model, "status": "deferred", "reason": "requires_gpu"})
            continue
        if cfg.get("gated"):
            print("  (gated HF repo — deferred pending access approval)", flush=True)
            rollup.append(
                {
                    "model": model,
                    "status": "deferred",
                    "reason": "gated_hf_repo",
                    "manifest": None,
                    "prompt_examples": [],
                }
            )
            continue
        tiers = _tiers_for(model)
        manifest = _run_benchmark(model, tiers)
        examples: list[dict] = []
        if not args.skip_examples and manifest:
            print("  -> collecting prompt examples", flush=True)
            examples = _collect_prompt_examples(model)
        entry = {
            "model": model,
            "tiers": tiers,
            "has_text_probe": _has_text_probe(model),
            "has_brain_probe": _has_brain_probe(model),
            "manifest": manifest,
            "prompt_examples": examples,
        }
        rollup.append(entry)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_run": len(rollup),
        "entries": rollup,
    }
    path = REPORTS / "all_models_rollup.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
