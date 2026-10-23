#!/usr/bin/env python3
"""Render IEEE-ready paper figures from sealed Anima report JSON (no re-eval)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmarks" / "reports"
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("qwen_qwen2_5-0_5b-instruct", "Qwen2.5-0.5B"),
    ("tinyllama_tinyllama-1_1b-chat-v1_0", "TinyLlama-1.1B"),
]
SUITES = ("halueval", "truthfulqa")


def _load(name: str) -> dict:
    p = REPORTS / name
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def fig1_protocol_f_vs_l():
    """Fixture smoke vs baseline live AUROC (collapse figure)."""
    labels = []
    f_vals, l_vals = [], []
    for slug, short in MODELS:
        live = _load(f"live_guard_{slug}.json")
        gsb = _load(f"greenshield_bench_{slug}.json")
        for suite in SUITES:
            labels.append(f"{short}\n{suite}")
            lb = (live.get("benchmarks") or {}).get(suite) or {}
            gs = (gsb.get("suites") or {}).get(suite) or {}
            f_vals.append(float(gs.get("protocol_f_auroc") or 1.0))
            l_vals.append(float(lb.get("auroc_composite") or gs.get("protocol_l_auroc") or 0.5))

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    ax.bar(x - w / 2, f_vals, w, label="Protocol F (fixture)", color="#90a4ae")
    ax.bar(x + w / 2, l_vals, w, label="Protocol L (live baseline)", color="#c62828")
    ax.axhline(0.5, color="#455a64", ls="--", lw=0.8, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig. 1  Fixture smoke vs live baseline (green-shield collapse)")
    ax.legend(fontsize=7, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_fixture_vs_live.png")
    fig.savefig(OUT / "fig1_fixture_vs_live.pdf")
    plt.close(fig)


def fig2_l_prime_unlock():
    """Baseline L vs L′ LOQO AUROC."""
    labels, base, adapted = [], [], []
    for slug, short in MODELS:
        live = _load(f"live_guard_{slug}.json")
        lp = _load(f"protocol_l_prime_{slug}.json")
        for suite in SUITES:
            labels.append(f"{short}\n{suite}")
            lb = (live.get("benchmarks") or {}).get(suite) or {}
            ls = (lp.get("suites") or {}).get(suite) or {}
            base.append(float(lb.get("auroc_composite") or 0.5))
            adapted.append(float(ls.get("auroc_loqo") or 0.5))

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    ax.bar(x - w / 2, base, w, label="Baseline L", color="#c62828")
    ax.bar(x + w / 2, adapted, w, label="L′ LOQO (named OP)", color="#2e7d32")
    ax.axhline(0.55, color="#1565c0", ls=":", lw=0.9, label="Primary unlock floor (0.55)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig. 2  Named Protocol L′ recovery vs baseline collapse")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_lprime_unlock.png")
    fig.savefig(OUT / "fig2_lprime_unlock.pdf")
    plt.close(fig)


def fig3_badge_state():
    """Baseline vs adapted claim_allowed."""
    names, base_c, ad_c = [], [], []
    for slug, short in MODELS:
        tig_b = _load(f"trust_integrity_{slug}.json")
        tig_a = _load(f"trust_integrity_adapted_{slug}.json")
        names.append(short)
        base_c.append(1 if tig_b.get("trust_ui_claim_allowed") else 0)
        ad_c.append(1 if tig_a.get("trust_ui_claim_allowed") else 0)

    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=200)
    ax.bar(x - w / 2, base_c, w, label="Baseline L claim_allowed", color="#c62828")
    ax.bar(x + w / 2, ad_c, w, label="Adapted L′ claim_allowed", color="#2e7d32")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LOCKED (0)", "UNLOCKED (1)"])
    ax.set_title("Fig. 3  Trust Integrity Gate claim bit")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_trust_badge.png")
    fig.savefig(OUT / "fig3_trust_badge.pdf")
    plt.close(fig)


def fig4_fmri_ceilings():
    """Word-rate encoding vs VA probe holdout."""
    depth = _load("fmri_depth_distilgpt2.json")
    stories = depth.get("stories") or {}
    wr = [float((stories.get(s) or {}).get("word_rate_max_r") or 0) for s in ("pieman", "lucy")]
    va = depth.get("anima_va_probe") or {}
    r_v = float(va.get("val_r_valence") or 0)
    labels = ["pieman\nword-rate max r", "lucy\nword-rate max r", "holdout\nPCA→VA r_v"]
    vals = wr + [r_v]
    colors = ["#1565c0", "#1565c0", "#ef6c00"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    ax.bar(np.arange(3), vals, color=colors)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.05, 0.75)
    ax.set_title("Fig. 4  Narratives depth: encoding ceiling vs VA probe")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_fmri_ceilings.png")
    fig.savefig(OUT / "fig4_fmri_ceilings.pdf")
    plt.close(fig)


def fig5_se_matched():
    """Semantic entropy vs Anima composite on matched rows."""
    labels, se, anima = [], [], []
    for slug, short in MODELS:
        se_j = _load(f"semantic_entropy_matched_{slug}.json")
        suites = se_j.get("suites") or {}
        for suite in SUITES:
            s = suites.get(suite) or {}
            b = s.get("baselines") or {}
            labels.append(f"{short}\n{suite}")
            se.append(float((b.get("semantic_entropy") or {}).get("auroc") or 0.5))
            anima.append(float((b.get("anima_composite") or {}).get("auroc") or 0.5))

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    ax.bar(x - w / 2, se, w, label="Matched semantic entropy", color="#5c6bc0")
    ax.bar(x + w / 2, anima, w, label="Anima baseline composite", color="#c62828")
    ax.axhline(0.5, color="#455a64", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig. 5  Matched-row semantic entropy vs Anima composite")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_se_matched.png")
    fig.savefig(OUT / "fig5_se_matched.pdf")
    plt.close(fig)


def main():
    fig1_protocol_f_vs_l()
    fig2_l_prime_unlock()
    fig3_badge_state()
    fig4_fmri_ceilings()
    fig5_se_matched()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
