"""Generate complete Anima patent + paper document pack on Desktop (PDF + Markdown)."""

from __future__ import annotations

import html
import re
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer

DESKTOP = Path.home() / "OneDrive" / "Desktop"
if not DESKTOP.is_dir():
    DESKTOP = Path.home() / "Desktop"

PACK_DIR = DESKTOP / "Anima_Document_Pack"
TODAY = date.today().isoformat()

# ── Authoritative benchmark numbers (2026-07-06, git 65a175f) ──────────────
BENCH = {
    "run_date": "2026-07-06",
    "git_sha": "65a175f",
    "disclosure_date": "2026-05-17",
    "us_grace_deadline": "2027-05-17",
    "models": [
        {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "council": 94.0,
            "pass": True,
            "goe_v": 0.188,
            "goe_a": 0.412,
            "brain_v": None,
            "gap": 0.488,
            "pos_v": 0.304,
            "neg_v": -0.185,
            "hero": True,
        },
        {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "council": 91.0,
            "pass": True,
            "goe_v": 0.236,
            "goe_a": 0.418,
            "brain_v": None,
            "gap": 0.267,
            "pos_v": 0.372,
            "neg_v": 0.105,
            "hero": False,
        },
        {
            "name": "distilgpt2",
            "council": 82.2,
            "pass": True,
            "goe_v": 0.163,
            "goe_a": 0.000,
            "brain_v": -0.393,
            "gap": 0.314,
            "pos_v": 0.595,
            "neg_v": 0.281,
            "hero": False,
        },
        {
            "name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "council": 58.5,
            "pass": False,
            "goe_v": 0.000,
            "goe_a": None,
            "brain_v": None,
            "gap": -0.149,
            "pos_v": 0.768,
            "neg_v": 0.917,
            "hero": False,
        },
        {
            "name": "hf-internal-testing/tiny-random-gpt2",
            "council": 50.2,
            "pass": False,
            "goe_v": 0.004,
            "goe_a": None,
            "brain_v": -0.109,
            "gap": -0.004,
            "pos_v": 0.143,
            "neg_v": 0.147,
            "hero": False,
        },
    ],
    "guard": {
        "n_samples": 52,
        "abstain_accuracy": 1.0,
        "auroc": 1.0,
        "tp": 26,
        "fp": 0,
        "tn": 26,
        "fn": 0,
        "note": "Fixture-policy smoke test on pre-labeled readout rows — NOT live LM hallucination detection",
    },
    "constants": {
        "valence_suppression": 0.35,
        "uncertainty_overclaim": -0.30,
        "stability_window": 8,
        "stability_threshold": 0.45,
        "stability_weights": "0.45 vol + 0.35 delta + 0.20 abstain",
        "guard_composite": "0.55 fused_cal + 0.45 probe_u",
        "fused_abstain": 0.82,
        "probe_u_abstain": 0.78,
        "hedge_threshold": 2,
        "fused_uncertainty": "0.35 entropy + 0.35 logit_gap + 0.30 attn_entropy",
        "dampen_alpha": 0.12,
        "dampen_valence_delta": 0.25,
    },
}

AUTHOR = {
    "name": "Siddarth Boggarapu",
    "email": "siddarthb078@gmail.com",
    "phone": "+91 9606754584",
    "location": "Bangalore, Karnataka, India",
    "repo": "https://github.com/Siddarthb07/Anima",
}


def _bench_table_md() -> str:
    lines = [
        "| Model | Council | Pass | GoE r(v/a) | Brain r(v)* | Pos−Neg gap |",
        "|-------|---------|------|------------|-------------|-------------|",
    ]
    for m in BENCH["models"]:
        goe = f"{m['goe_v']:.3f} / {m['goe_a']:.3f}" if m["goe_a"] is not None else f"{m['goe_v']:.3f} / —"
        brain = f"{m['brain_v']:.3f}" if m["brain_v"] is not None else "—"
        lines.append(
            f"| {m['name']} | {m['council']:.1f} | {'Yes' if m['pass'] else 'No'} | {goe} | {brain} | {m['gap']:.3f} |"
        )
    lines.append("\n*Brain holdout on synthetic_minimal (story lucy); not real ds002345.")
    return "\n".join(lines)


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=base["Title"], fontSize=15, spaceAfter=12, textColor=colors.HexColor("#1a1a2e")),
        "subtitle": ParagraphStyle("S", parent=base["Normal"], fontSize=9, textColor=colors.HexColor("#555"), spaceAfter=14),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontSize=12, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#16213e")),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=10.5, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("B", parent=base["Normal"], fontSize=9.5, leading=13, spaceAfter=5),
        "bullet": ParagraphStyle("Bu", parent=base["Normal"], fontSize=9.5, leading=13, leftIndent=16, spaceAfter=3),
        "mono": ParagraphStyle("M", parent=base["Code"], fontSize=8, leading=10.5, fontName="Courier", spaceAfter=4),
    }


def md_to_flowables(text: str, styles: dict) -> list:
    """Minimal markdown → reportlab flowables."""
    flow = []
    for raw in text.split("\n"):
        line = raw.rstrip()
        if not line:
            flow.append(Spacer(1, 0.06 * inch))
            continue
        if line.startswith("# "):
            flow.append(Paragraph(html.escape(line[2:]), styles["title"]))
        elif line.startswith("## "):
            flow.append(Paragraph(html.escape(line[3:]), styles["h1"]))
        elif line.startswith("### "):
            flow.append(Paragraph(html.escape(line[4:]), styles["h2"]))
        elif line.startswith("---"):
            flow.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey))
        elif line.startswith("|"):
            flow.append(Paragraph(html.escape(line), styles["mono"]))
        elif line.startswith("- ") or line.startswith("* "):
            body = line[2:]
            body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", body)
            flow.append(Paragraph(f"• {body}", styles["bullet"]))
        else:
            body = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
            body = re.sub(r"`(.+?)`", r"<font face='Courier' size='8'>\1</font>", body)
            flow.append(Paragraph(body, styles["body"]))
    return flow


def write_md(name: str, content: str) -> Path:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    p = PACK_DIR / name
    p.write_text(content, encoding="utf-8")
    return p


def write_pdf(name: str, markdown_content: str) -> Path:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    out = PACK_DIR / name
    doc = SimpleDocTemplate(
        str(out),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = _styles()
    story = md_to_flowables(markdown_content, styles)
    doc.build(story)
    # Also copy to desktop root for easy access
    root_copy = DESKTOP / name
    root_copy.write_bytes(out.read_bytes())
    return out


def doc_index() -> str:
    return f"""# Anima Document Pack — Index

**Generated:** {TODAY}  
**Inventor:** {AUTHOR['name']} · {AUTHOR['email']}  
**Repository:** {AUTHOR['repo']} (public since {BENCH['disclosure_date']})

## Files in this folder

| File | Description |
|------|-------------|
| `00_INDEX.pdf` | This index |
| `01_USA_Research_Paper_Plan.pdf` | 12-week plan to submit US workshop paper |
| `02_India_Patent_Plan.pdf` | India filing strategy + future inventions |
| `03_Patent_Filing_Letter.pdf` | Cover letter to patent counsel |
| `04_Annex_A_Invention_Disclosure.pdf` | Technical disclosure for provisional |
| `05_Annex_B_Prior_Art.pdf` | Prior art references (MERA, ITI, CAA, etc.) |
| `06_Paper_Draft.pdf` | System paper with benchmark numbers |
| `07_Audit_Issues.pdf` | Issue log from original PDF review |
| `08_Council_Synthesis.pdf` | Patent/paper verdict + action list |

All `.md` sources included alongside PDFs.

## Key numbers at a glance

- Benchmark run: **{BENCH['run_date']}** · git `{BENCH['git_sha']}`
- Best council score: **94.0** (TinyLlama-1.1B-Chat)
- Guard fixture: n=**{BENCH['guard']['n_samples']}**, AUROC **{BENCH['guard']['auroc']:.2f}** (fixture smoke only)
- US patent grace deadline: **~{BENCH['us_grace_deadline']}**
- India Anima core: **likely not fileable** (public since {BENCH['disclosure_date']})
"""


def doc_usa_plan() -> str:
    g = BENCH["guard"]
    return f"""# Anima — USA Research Paper Submission Plan

**{AUTHOR['name']}** · {TODAY} · {AUTHOR['repo']}

## 1. Goal

Maximize acceptance at a **US interpretability / trustworthy ML workshop** (NeurIPS, ICML, or ICLR workshop track) in 2026–2027. Target **systems + methodology**, not SOTA hallucination detection.

## 2. Current benchmark baseline ({BENCH['run_date']})

{_bench_table_md()}

**Guard metrics (all models, fixture only):** n={g['n_samples']}, accuracy={g['abstain_accuracy']:.2f}, AUROC={g['auroc']:.2f}, TP/FP/TN/FN = {g['tp']}/{g['fp']}/{g['tn']}/{g['fn']}.  
**{g['note']}**

**Hero model for paper figures:** TinyLlama-1.1B-Chat (council **94.0**, gap **0.488**, neg valence **−0.185**).

## 3. Acceptance probability estimates

| Stage | Estimated acceptance | Notes |
|-------|---------------------|-------|
| Workshop, current draft only | **20–30%** | Guard not validated on live gens |
| Workshop, after E1 + figures | **45–55%** | Credible systems contribution |
| EMNLP Findings / demo, after E1–E4 | **55–65%** | Full ablation + baselines |
| NeurIPS/ICML main (reliability) | **<15%** | Do not target first |

## 4. Required experiments (with success metrics)

### E1 — Live guard evaluation (CRITICAL, weeks 1–2)

- **Task:** Run guard on live LM generations labeled for hallucination (HaluEval-QA + TruthfulQA subset, target n≥200).
- **Success metric:** AUROC ≥ **0.65** on live generations; report calibration curve.
- **Commands:** `anima benchmark --model TinyLlama/TinyLlama-1.1B-Chat-v1.0` + new live guard runner.

### E2 — Ablation study (week 3)

| Config | Expected metric |
|--------|-----------------|
| Guard fusion only | Baseline AUROC |
| + layer-disagreement as abstain reason | ΔAUROC |
| + stability gate (`guard_mode=gate`) | ΔAUROC |
| − lexical hedging | ΔAUROC |
| Full system | Final AUROC |

**Success:** Full fusion beats best single-signal by ≥ **0.03** AUROC.

### E3 — Baselines (week 4)

- Semantic entropy (Farquhar et al., 2024) — same prompt set.
- Self-consistency n=5 — same prompt set.
- **Success:** Document compute cost (Anima = 1 forward pass vs n samples).

### E4 — MERA comparison (week 5)

- Matched MCQ slice vs Hedstrom et al. (ICML 2025).
- **Success:** Table showing cost, abstain rate, accuracy tradeoff.

## 5. 12-week timeline

| Week | Deliverable |
|------|-------------|
| 1–2 | E1 live guard eval on TinyLlama + Qwen |
| 3 | E2 ablations |
| 4 | E3 semantic-entropy baseline |
| 5 | E4 MERA table + architecture figure |
| 6 | Paper v3 internal review |
| 7 | Patent/arXiv sequencing decision |
| 8–10 | Workshop submission |
| 11–12 | HF Gradio Space (v2.1.0) linked in paper |

## 6. Target venues (USA)

1. NeurIPS Mechanistic Interpretability / Safe Generative AI workshop
2. ICML interpretability workshop
3. ACL/EMNLP System Demonstrations
4. arXiv cs.CL — **only after** US provisional decision

## 7. Budget estimate (USD)

| Item | Cost |
|------|------|
| GPU cloud for E1 (optional) | $0–50 |
| Workshop registration | $0–100 |
| US provisional (via counsel) | $2,000–5,000 |
| **Total (paper path)** | **~$50–150** excluding patent |

## 8. Immediate actions (this week)

- Implement live HaluEval guard runner
- Set TinyLlama as hero in all figures
- Check workshop CFP deadlines
- Do not post arXiv until provisional decision
"""


def doc_india_plan() -> str:
    return f"""# Anima & Portfolio — India Patent Plan

**{AUTHOR['name']}** · {TODAY} · {AUTHOR['location']}

## 1. Anima core — India status

| Fact | Value |
|------|-------|
| Public GitHub created | **{BENCH['disclosure_date']}** |
| Section 31 grace (exhibition/learned society) | **Does NOT cover GitHub** |
| **India filing for disclosed Anima core** | **Likely NO** |

## 2. US vs India comparison

| Jurisdiction | File? | Grant odds | Deadline |
|--------------|-------|------------|----------|
| **India (Anima core)** | No | ~0% | Already disclosed |
| **USA (Anima combination)** | Conditional yes | 15–30% | **{BENCH['us_grace_deadline']}** |
| **India (future inventions)** | Yes if private first | 25–55% | Before next public release |

**Section 39 note:** Indian resident filing US first may need **Form 25 (FFL)** — consult Indian patent agent.

## 3. Disclosed on GitHub — cannot claim in India

- Affect probes + layer-softmax fusion
- Layer-disagreement (thresholds {BENCH['constants']['valence_suppression']} / {BENCH['constants']['uncertainty_overclaim']})
- Rolling stability (window={BENCH['constants']['stability_window']}, threshold={BENCH['constants']['stability_threshold']})
- Guard fusion ({BENCH['constants']['guard_composite']})
- Conditional steering (α={BENCH['constants']['dampen_alpha']})
- Validity council rubric (benchmarks/council.py)

## 4. Future inventions — file India BEFORE publish

### Candidate A: Unified guard integration (best Anima follow-on)

Wire layer-disagreement + stability **into** `abstain_recommended`.  
**Patentability if undisclosed:** **35–45%** · **Dev time:** 2–3 weeks · **Cost:** ₹25,000–75,000 provisional

### Candidate B: Real ds002345 uncertainty-supervised probe training (v3.0)

**Patentability:** **25–35%** · Requires ~100 GB download + GPU

### Candidate C: NeuralVortex FNO on CFD data (separate repo)

**Patentability if still private:** **40–55%** · **Highest India ROI in portfolio**

### Skip for patent

- text2sql-rag (crowded prior art)
- Lexprobe (trade secret better than patent)

## 5. India filing protocol

1. Build privately (no GitHub, no demo, no paper)
2. Indian patent agent + novelty search (₹15,000–40,000)
3. File Indian provisional (Form 1/2)
4. Form 25 if filing US within 12 months
5. PCT optional within 12 months
6. **Then** git push / arXiv / portfolio

## 6. Recommended budget (INR)

| Priority | Item | Cost (INR) |
|----------|------|------------|
| 1 | Counsel opinion (Anima India dead?) | 5,000–15,000 |
| 2 | NeuralVortex provisional (if private) | 25,000–75,000 |
| 3 | Anima unified-guard provisional | 25,000–75,000 |
| **Skip** | Anima core re-filing | 0 (waste) |

## 7. Decision matrix

| Invention | India today | Action |
|-----------|-------------|--------|
| Anima core (public {BENCH['disclosure_date']}) | NO | US paper + optional US provisional |
| Unified guard fusion (new) | YES* | Build private → IN provisional → publish |
| NeuralVortex (if private) | YES* | IN provisional FIRST |
| text2sql-rag | Unlikely | Paper only |

*Only if not publicly disclosed before filing date.
"""


def doc_patent_letter() -> str:
    c = BENCH["constants"]
    return f"""# Instructions to Patent Counsel

**From:** {AUTHOR['name']}, Founder Athera · {AUTHOR['location']}  
**Email:** {AUTHOR['email']} · **Phone:** {AUTHOR['phone']}  
**Date:** {TODAY}

**To:** [ATTORNEY / PATENT AGENT NAME — fill before sending]

**Re:** Provisional Patent — Multi-Signal Reliability Gating System ("Anima")

---

## 1. Time-sensitive disclosure (address first)

Source code: **{AUTHOR['repo']}**  
Public since: **{BENCH['disclosure_date']}** (UTC)

Please advise:
1. **India:** Is filing still available? (I understand likely **no** — no grace for GitHub.)
2. **USA:** 12-month grace under 35 U.S.C. § 102(b)(1)(A) — deadline **~{BENCH['us_grace_deadline']}** (~10 months remain as of {TODAY}).
3. **PCT:** Viable jurisdictions and timeline.
4. **Section 39 / Form 25:** Required if I file US first as Indian resident?

## 2. Invention summary

Four coordinated mechanisms on shared per-token readout stream:
1. Layer-disagreement (valence shift > {c['valence_suppression']}, uncertainty shift < {c['uncertainty_overclaim']})
2. Rolling stability (window={c['stability_window']}, threshold={c['stability_threshold']}, weights: {c['stability_weights']})
3. Guard abstention ({c['guard_composite']}; fused ≥ {c['fused_abstain']} or ≥2 reasons)
4. Conditional steering (α={c['dampen_alpha']}, valence delta ≥ {c['dampen_valence_delta']})

**Claim drafting note:** Mechanisms are coordinated but not all fused into one abstain score. Layer-disagreement and stability are parallel outputs unless `guard_mode=gate`.

## 3. Preliminary benchmark data (for spec examples only)

{_bench_table_md()}

Guard fixture: n={BENCH['guard']['n_samples']}, AUROC={BENCH['guard']['auroc']:.2f} — **policy smoke test, not hallucination detection.**

## 4. Closest prior art

**MERA** (Hedstrom et al., ICML 2025) — abstention + calibrated steering. See Annex B.

## 5. Engagement requested

| Stage | Action |
|-------|--------|
| 1 | Patentability opinion on **combination claim** |
| 2 | Jurisdiction / timing guidance |
| 3 | US provisional draft (conditional) |
| 4 | Filing (conditional) |

**Materials enclosed:** Annex A, Annex B, Paper Draft, source code reference.

I prefer an honest "do not file" over optimistic filing.

---

**Enclosures:** Annex A, Annex B, Draft Paper, GitHub reference
"""


def doc_annex_a() -> str:
    c = BENCH["constants"]
    return f"""# Annex A — Invention Disclosure

**Invention:** Multi-Signal Reliability Gating System for Language Model Generation ("Anima")  
**Inventor:** {AUTHOR['name']} · **Date:** {TODAY}  
**Repository:** {AUTHOR['repo']} (public since {BENCH['disclosure_date']})

## 1. Disclosure timing

Public GitHub since **{BENCH['disclosure_date']}**. No academic paper published as of {TODAY}.  
US grace deadline approximately **{BENCH['us_grace_deadline']}**. India filing likely unavailable for disclosed subject matter.

## 2. System overview

Forward hooks → AffectProbe (valence [−1,1], arousal [0,1], uncertainty [0,1]) → four mechanisms:

| Mechanism | Drives abstain? | Default |
|-----------|-----------------|---------|
| Guard fusion | **Yes** | Always on |
| Layer disagreement | No (API events) | detect_suppression=true |
| Rolling stability | Region relabel only | guard_mode=observe |
| Conditional steering | N/A | intervention_mode=none |

## 3. Technical constants (as implemented)

- Fused uncertainty: {c['fused_uncertainty']}
- Guard composite: {c['guard_composite']}
- Abstain: fused_cal ≥ {c['fused_abstain']} OR ≥2 reasons (fused/probe/hedging ≥ {c['hedge_threshold']} words)
- Layer disagreement: valence_shift > {c['valence_suppression']}; uncertainty_shift < {c['uncertainty_overclaim']}
- Stability: window={c['stability_window']}, score threshold={c['stability_threshold']}, weights {c['stability_weights']}
- Steering: α={c['dampen_alpha']}, trigger |Δvalence| ≥ {c['dampen_valence_delta']}

## 4. Benchmark validation ({BENCH['run_date']}, git {BENCH['git_sha']})

{_bench_table_md()}

Guard fixture (n={BENCH['guard']['n_samples']}): accuracy={BENCH['guard']['abstain_accuracy']:.2f}, AUROC={BENCH['guard']['auroc']:.2f}. Confirms policy execution on supplied readout rows only.

## 5. Novelty hypothesis (for counsel to test)

Combination of all four mechanisms + affect-framed probe readout space, operating during autoregressive generation. Individual mechanisms not novel. Differentiate from MERA (no helper model; layer-disagreement + stability as parallel signals).

## 6. Materials

Source: {AUTHOR['repo']} · Technical overview: docs/ANIMA_TECHNICAL_OVERVIEW.md · Manifests: benchmarks/reports/
"""


def doc_annex_b() -> str:
    return f"""# Annex B — Prior Art References

**Date:** {TODAY} · Preliminary inventor search — counsel must conduct independent search

## A. Linear probing / hidden states

- **Marks & Tegmark (2023)** — Geometry of Truth, arXiv:2310.06824
- **Thai et al. (2025)** — Factuality Probes, EMNLP Findings, DOI: 10.18653/v1/2025.findings-emnlp.880
- **O'Neill et al. (2025)** — Single Direction of Truth, arXiv:2507.23221

## B. Activation steering

- **Li et al. (2023)** — Inference-Time Intervention, NeurIPS, arXiv:2306.03341
- **Rimsky et al. (2024)** — Contrastive Activation Addition, ACL, DOI: 10.18653/v1/2024.acl-long.828
- **Zou et al. (2023)** — Representation Engineering, arXiv:2310.01405

## C. Abstention + steering (closest combination prior art)

- **Hedstrom et al. (2025)** — MERA: Mechanistic Error Reduction with Abstention, ICML, PMLR 267:22924-22945, arXiv:2510.13290
  - **Differentiation:** MERA uses helper model; Anima uses probe-derived affect signals + layer-disagreement + stability without helper model.

## D. Uncertainty baselines

- **Farquhar et al. (2024)** — Semantic Entropy, Nature

## E. Claim drafting notes

1. Draft narrowly around four-mechanism combination + affect framing
2. Do not claim individual probing/steering/abstention standalone
3. Include dependent claims on thresholds listed in Annex A §3
4. Do not overclaim guard performance (fixture n={BENCH['guard']['n_samples']} only)
5. MERA (C) is primary prosecution risk
"""


def doc_paper_draft() -> str:
    c = BENCH["constants"]
    g = BENCH["guard"]
    return f"""# Anima: Affect-Probed Reliability Gating for Language Model Generation

**{AUTHOR['name']}** · Independent · {AUTHOR['location']} · {AUTHOR['email']}  
**Draft v3 · {TODAY}** · Benchmark run {BENCH['run_date']} · git {BENCH['git_sha']}

## Abstract

We present Anima, an open-source system reading valence, arousal, and uncertainty from causal LM hidden states via linear probes and forward hooks. Four coordinated mechanisms operate on a shared readout stream: (1) layer-disagreement detection, (2) rolling stability scoring, (3) guard fusion abstention, (4) optional conditional steering. We report CPU-tier benchmarks across five evaluated models (0.1B–1.7B). Best council score: **94.0** (TinyLlama-1.1B-Chat). Guard AUROC **{g['auroc']:.2f}** on n=**{g['n_samples']}** fixture rows only — not live hallucination detection. Systems/methodology contribution, not SOTA reliability claim.

## 1. Introduction

Anima integrates affect-framed probing with layer-disagreement, stability, guard fusion, and optional steering in a real-time instrumentation stack (FastAPI + WebSocket + dashboard). Abstention driven by guard module; other mechanisms are parallel unless `guard_mode=gate`.

## 2. Related Work

- **Probing:** Marks & Tegmark 2023; Thai et al. EMNLP 2025
- **Steering:** Li et al. NeurIPS 2023 (ITI); Rimsky et al. ACL 2024 (CAA)
- **Abstention:** Hedstrom et al. ICML 2025 (MERA) — closest prior art

## 3. System

**Probe:** layer-softmax fusion → valence/arousal/uncertainty heads.  
**Auxiliary uncertainty:** {c['fused_uncertainty']}  
**Guard:** {c['guard_composite']}; abstain if fused ≥ {c['fused_abstain']} or ≥2 reasons.  
**Layer disagreement:** > {c['valence_suppression']} / < {c['uncertainty_overclaim']}.  
**Stability:** window {c['stability_window']}, threshold {c['stability_threshold']}.  
**Steering:** α={c['dampen_alpha']}, |Δvalence| ≥ {c['dampen_valence_delta']}, opt-in.

## 4. Experiments

CPU tier, {BENCH['run_date']}. Defaults: guard_mode=observe, intervention_mode=none, detect_suppression=true.

## 5. Results

### 5.1 Cross-model council

{_bench_table_md()}

### 5.2 Hero: TinyLlama-1.1B-Chat

| Metric | Value |
|--------|-------|
| Council score | **94.0** |
| GoEmotions r (v / a) | **0.188 / 0.412** |
| Pos − Neg gap | **0.488** |
| Negative prompt mean valence | **−0.185** |
| Guard fixture AUROC† | {g['auroc']:.2f} (n={g['n_samples']}) |

†Fixture smoke test only.

### 5.3 Qwen2.5-0.5B-Instruct

| Metric | Value |
|--------|-------|
| Council score | **91.0** |
| GoEmotions r (v / a) | **0.236 / 0.418** |
| Pos − Neg gap | **0.267** |

### 5.4 distilgpt2 (brain track, synthetic_minimal)

| Metric | Value |
|--------|-------|
| Council score | **82.2** |
| GoEmotions r (v / a) | **0.163 / 0.000** |
| Brain holdout r (v / a) | **−0.393 / −0.022** |
| Val MSE | **0.176** |

### 5.5 Failure cases

- **SmolLM2-1.7B:** council **58.5**, inverted gap **−0.149**
- **tiny-random-gpt2:** council **50.2** (noise control)

## 6. Limitations

1. Guard AUROC {g['auroc']:.2f} is fixture-only (n={g['n_samples']})
2. Brain tier is synthetic_minimal, not ds002345
3. No ablation vs single-signal baselines yet
4. No semantic-entropy baseline yet

## 7. Planned work before submission

1. Live HaluEval/TruthfulQA guard (target AUROC ≥ 0.65)
2. Ablations (target Δ ≥ 0.03 vs best single signal)
3. MERA comparison table
4. Workshop submission (NeurIPS/ICML interpretability track)

## References

Li et al. 2023; Rimsky et al. 2024; Hedstrom et al. 2025; Thai et al. 2025; Marks & Tegmark 2023; Farquhar et al. 2024.
"""


def doc_audit() -> str:
    return f"""# Audit Issues — Original PDFs vs Fixed Pack

**Date:** {TODAY}

## Critical fixes applied

| ID | Issue | Fix |
|----|-------|-----|
| C1 | Repository URL placeholder | → {AUTHOR['repo']} |
| C2 | Disclosure date placeholder | → {BENCH['disclosure_date']} |
| C3 | India filing implied possible | Explicit: likely barred |
| C4 | Guard AUROC 1.00 overclaim | Fixture n={BENCH['guard']['n_samples']} disclaimer everywhere |
| C5 | GoE numbers inconsistent | Reconciled to manifests ({BENCH['run_date']}) |
| C6 | Missing Annex B | Created with verified citations |
| C7 | Related Work TODO | Filled |
| C8 | 3 vs 4 mechanisms | Corrected to four |
| F1 | Fused pipeline overclaim | Parallel signal paths documented |

## Verified constants

| Parameter | Value |
|-----------|-------|
| valence_suppression | {BENCH['constants']['valence_suppression']} |
| uncertainty_overclaim | {BENCH['constants']['uncertainty_overclaim']} |
| stability_window | {BENCH['constants']['stability_window']} |
| stability_threshold | {BENCH['constants']['stability_threshold']} |
| guard composite | {BENCH['constants']['guard_composite']} |
| fused_abstain | {BENCH['constants']['fused_abstain']} |
| dampen_alpha | {BENCH['constants']['dampen_alpha']} |
"""


def doc_council() -> str:
    return f"""# Council Synthesis — Patent & Paper Verdict

**Date:** {TODAY}

## Executive verdict

| Question | Verdict |
|----------|---------|
| US provisional filing | **Conditional YES** — narrow combination claims; deadline ~{BENCH['us_grace_deadline']} |
| India filing (Anima core) | **Likely NO** — public since {BENCH['disclosure_date']} |
| Paper submission now | **NO** for hallucination claim; **YES** for workshop after E1 eval |
| Grant confidence | **15–30%** US; **~0%** India for disclosed core |

## Key finding F1

Four mechanisms are **parallel**, not all fused into abstain. Guard uses fused/probe/hedging only. Claims must describe **system combination**, not single voting function.

## Priority actions

1. Counsel timing call (this week)
2. US provisional if counsel agrees (before {BENCH['us_grace_deadline']})
3. Live guard eval — target AUROC ≥ 0.65, n≥200
4. Ablations — target Δ ≥ 0.03 AUROC
5. Workshop submit after E1–E4
6. Skip India spend on Anima core
7. Consider India provisional for **NeuralVortex** or **unified guard** if built privately first

## Honest positioning

> Open-source instrumentation reading valence/arousal/uncertainty from LLM hidden states with multi-signal reliability layer, council-validated across models scoring 50.2–94.0.

Do NOT say "hallucination detector" until live benchmark ships.
"""


DOCUMENTS = [
    ("00_INDEX.pdf", "00_INDEX.md", doc_index),
    ("01_USA_Research_Paper_Plan.pdf", "01_USA_Research_Paper_Plan.md", doc_usa_plan),
    ("02_India_Patent_Plan.pdf", "02_India_Patent_Plan.md", doc_india_plan),
    ("03_Patent_Filing_Letter.pdf", "03_Patent_Filing_Letter.md", doc_patent_letter),
    ("04_Annex_A_Invention_Disclosure.pdf", "04_Annex_A_Invention_Disclosure.md", doc_annex_a),
    ("05_Annex_B_Prior_Art.pdf", "05_Annex_B_Prior_Art.md", doc_annex_b),
    ("06_Paper_Draft.pdf", "06_Paper_Draft.md", doc_paper_draft),
    ("07_Audit_Issues.pdf", "07_Audit_Issues.md", doc_audit),
    ("08_Council_Synthesis.pdf", "08_Council_Synthesis.md", doc_council),
]

# Legacy desktop names (user requested "original plans too")
LEGACY = [
    ("Anima_USA_Research_Paper_Plan.pdf", doc_usa_plan),
    ("Anima_India_Patent_Plan.pdf", doc_india_plan),
]


def main():
    print(f"Desktop pack folder: {PACK_DIR}")
    written = []
    for pdf_name, md_name, builder in DOCUMENTS:
        content = builder()
        write_md(md_name, content)
        p = write_pdf(pdf_name, content)
        written.append(p)
        print(f"  OK {pdf_name}")

    for pdf_name, builder in LEGACY:
        content = builder()
        p = write_pdf(pdf_name, content)
        print(f"  OK {pdf_name} (desktop root)")

    print(f"\nDone — {len(written) + len(LEGACY)} PDFs on Desktop + {PACK_DIR}")


if __name__ == "__main__":
    main()
