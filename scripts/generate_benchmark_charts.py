"""Generate PNG charts from all_models_rollup.json + council_rollup.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "benchmarks" / "reports"
OUT = ROOT / "docs" / "images" / "benchmarks"


def _short_model(name: str) -> str:
    if name == "hf-internal-testing/tiny-random-gpt2":
        return "tiny-random-gpt2"
    return name.split("/")[-1][:22]


def _manifest_entry(manifest: dict | None, benchmark: str) -> dict | None:
    if not manifest:
        return None
    for e in manifest.get("entries") or []:
        if e.get("benchmark") == benchmark and e.get("status") == "ok":
            return e
    return None


def _load_rollup(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_council(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _prompt_map(examples: list[dict]) -> dict[str, dict]:
    return {e["id"]: e for e in examples if e.get("id") and not e["id"].startswith("_")}


def generate_charts(rollup: dict, council: list[dict], out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")
    written: list[Path] = []

    entries = [e for e in rollup.get("entries", []) if e.get("manifest") or e.get("prompt_examples")]
    if not entries:
        return written

    models = [_short_model(e["model"]) for e in entries]
    council_by = {c["model"]: c for c in council}

    # --- 1. Council aggregate scores ---
    scores = [council_by.get(e["model"], {}).get("aggregate_score") for e in entries]
    score_colors = ["#64748b"] * len(models)
    if any(s is not None for s in scores):
        fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.45)))
        vals = [s if s is not None else 0 for s in scores]
        score_colors = ["#22d3ee" if (s or 0) >= 60 else "#fb7185" for s in vals]
        y = np.arange(len(models))
        ax.barh(y, vals, color=score_colors, height=0.6)
        ax.axvline(60, color="#94a3b8", linestyle="--", linewidth=1, label="pass threshold (60)")
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel("Council score (0–100)", fontsize=11)
        ax.set_title("Model validity — council aggregate score", fontsize=13, pad=12)
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        p = out_dir / "council_scores.png"
        fig.savefig(p, dpi=150, facecolor="#0c0f14")
        plt.close(fig)
        written.append(p)

    # --- 2. Probe benchmarks (GoEmotions + brain holdout) ---
    go_v, go_a, brain_v = [], [], []
    for e in entries:
        m = e.get("manifest")
        go = _manifest_entry(m, "go_emotions")
        narr = _manifest_entry(m, "narratives_holdout")
        go_v.append(float(go["pearson_valence"]) if go and go.get("pearson_valence") is not None else np.nan)
        go_a.append(float(go["pearson_arousal"]) if go and go.get("pearson_arousal") is not None else np.nan)
        brain_v.append(float(narr["val_r_valence"]) if narr and narr.get("val_r_valence") is not None else np.nan)

    if any(not np.isnan(x) for x in go_v + go_a + brain_v):
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(models))
        w = 0.25
        ax.bar(x - w, go_v, w, label="GoEmotions r (valence)", color="#a78bfa")
        ax.bar(x, go_a, w, label="GoEmotions r (arousal)", color="#38bdf8")
        ax.bar(x + w, brain_v, w, label="Brain holdout r (valence)", color="#fbbf24")
        ax.axhline(0.15, color="#22d3ee", linestyle=":", linewidth=1, alpha=0.8, label="text probe gate (+0.15)")
        ax.axhline(0, color="#64748b", linestyle="-", linewidth=0.8)
        ax.axhline(-0.1, color="#fb7185", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=28, ha="right", fontsize=9)
        ax.set_ylabel("Pearson r", fontsize=11)
        ax.set_title("Probe signal — where training tracks targets vs struggles", fontsize=13, pad=12)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(-0.55, 0.35)
        fig.tight_layout()
        p = out_dir / "probe_pearson_r.png"
        fig.savefig(p, dpi=150, facecolor="#0c0f14")
        plt.close(fig)
        written.append(p)

    # --- 3. Live prompt valence (positive vs negative) ---
    pos_v, neg_v, gaps = [], [], []
    for e in entries:
        pm = _prompt_map(e.get("prompt_examples") or [])
        pv = pm.get("positive", {}).get("mean_valence")
        nv = pm.get("negative", {}).get("mean_valence")
        pos_v.append(float(pv) if pv is not None else np.nan)
        neg_v.append(float(nv) if nv is not None else np.nan)
        gaps.append((float(pv) - float(nv)) if pv is not None and nv is not None else np.nan)

    if any(not np.isnan(x) for x in pos_v + neg_v):
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(models))
        w = 0.35
        ax.bar(x - w / 2, pos_v, w, label="Positive prompt mean valence", color="#34d399")
        ax.bar(x + w / 2, neg_v, w, label="Negative prompt mean valence", color="#f87171")
        ax.axhline(0.2, color="#34d399", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(-0.1, color="#f87171", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(0, color="#64748b", linestyle="-", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=28, ha="right", fontsize=9)
        ax.set_ylabel("Mean valence readout", fontsize=11)
        ax.set_title("Live prompt separation — legible emotion vs weak/inverted readouts", fontsize=13, pad=12)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.35, 0.85)
        fig.tight_layout()
        p = out_dir / "prompt_valence_separation.png"
        fig.savefig(p, dpi=150, facecolor="#0c0f14")
        plt.close(fig)
        written.append(p)

    # --- 4. Valence gap (positive − negative) ---
    if any(not np.isnan(g) for g in gaps):
        fig, ax = plt.subplots(figsize=(10, max(4, len(models) * 0.45)))
        colors = ["#22d3ee" if (g or 0) >= 0.25 else "#fbbf24" if (g or 0) >= 0.1 else "#fb7185" for g in gaps]
        y = np.arange(len(models))
        ax.barh(y, [g if not np.isnan(g) else 0 for g in gaps], color=colors, height=0.6)
        ax.axvline(0.25, color="#22d3ee", linestyle="--", linewidth=1, label="strong gap (0.25)")
        ax.axvline(0.1, color="#fbbf24", linestyle=":", linewidth=1, label="weak gap (0.10)")
        ax.axvline(0, color="#64748b", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel("Valence gap (positive − negative prompt)", fontsize=11)
        ax.set_title("Emotion steering headroom — larger gap = clearer positive/negative split", fontsize=13, pad=12)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        p = out_dir / "valence_gap.png"
        fig.savefig(p, dpi=150, facecolor="#0c0f14")
        plt.close(fig)
        written.append(p)

    # --- 5. Stability on hedge-heavy prompt ---
    stab, swing = [], []
    for e in entries:
        pm = _prompt_map(e.get("prompt_examples") or [])
        h = pm.get("hedge_volatile", {})
        stab.append(float(h["stability_score"]) if h.get("stability_score") is not None else np.nan)
        swing.append(float(h["valence_swing"]) if h.get("valence_swing") is not None else np.nan)

    if any(not np.isnan(s) for s in stab):
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(models))
        ax.bar(x, stab, color="#22d3ee", alpha=0.85, label="Stability score (hedge prompt)")
        ax2 = ax.twinx()
        ax2.plot(x, swing, "o-", color="#fb7185", linewidth=2, markersize=7, label="Valence swing")
        ax.axhline(0.35, color="#fbbf24", linestyle=":", linewidth=1, label="unstable threshold (0.35)")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=28, ha="right", fontsize=9)
        ax.set_ylabel("Stability score (↑ calmer readouts)", fontsize=10)
        ax2.set_ylabel("Valence swing on hedge prompt", fontsize=10, color="#fb7185")
        ax.set_title("Intervention surface — choppy readouts on hedged language", fontsize=13, pad=12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        fig.tight_layout()
        p = out_dir / "hedge_stability.png"
        fig.savefig(p, dpi=150, facecolor="#0c0f14")
        plt.close(fig)
        written.append(p)

    # --- 6. Combined overview (2x2) ---
    if len(entries) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor("#0c0f14")

        ax = axes[0, 0]
        if any(s is not None for s in scores):
            ax.barh(models, [s if s is not None else 0 for s in scores], color=score_colors)
            ax.axvline(60, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_title("Council score", fontsize=11)
        ax.set_xlim(0, 100)

        ax = axes[0, 1]
        x = np.arange(len(models))
        ax.bar(x - 0.17, go_v, 0.34, label="GoE valence", color="#a78bfa")
        ax.bar(x + 0.17, brain_v, 0.34, label="Brain holdout", color="#fbbf24")
        ax.axhline(0.15, color="#22d3ee", linestyle=":", linewidth=1)
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
        ax.set_title("Probe Pearson r", fontsize=11)
        ax.legend(fontsize=7)

        ax = axes[1, 0]
        ax.bar(x - 0.17, pos_v, 0.34, color="#34d399", label="Positive")
        ax.bar(x + 0.17, neg_v, 0.34, color="#f87171", label="Negative")
        ax.axhline(0, color="#64748b", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
        ax.set_title("Live prompt mean valence", fontsize=11)
        ax.legend(fontsize=7)

        ax = axes[1, 1]
        gap_colors = ["#22d3ee" if (g or 0) >= 0.25 else "#fb7185" for g in gaps]
        ax.bar(x, [g if not np.isnan(g) else 0 for g in gaps], color=gap_colors)
        ax.axhline(0.25, color="#22d3ee", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
        ax.set_title("Valence gap (pos − neg)", fontsize=11)

        fig.suptitle("Anima benchmark overview — CPU tier, 2026-07-06", fontsize=14, y=1.01)
        fig.tight_layout()
        p = out_dir / "benchmark_overview.png"
        fig.savefig(p, dpi=150, facecolor="#0c0f14", bbox_inches="tight")
        plt.close(fig)
        written.append(p)

    # manifest for README / site
    meta = {
        "generated_from": str(REPORTS / "all_models_rollup.json"),
        "charts": [str(x.relative_to(ROOT)).replace("\\", "/") for x in written],
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rollup", default=str(REPORTS / "all_models_rollup.json"))
    p.add_argument("--council", default=str(REPORTS / "council_rollup.json"))
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    rollup = _load_rollup(Path(args.rollup))
    council = _load_council(Path(args.council))
    paths = generate_charts(rollup, council, Path(args.out))
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
