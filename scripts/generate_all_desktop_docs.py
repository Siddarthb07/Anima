"""Generate full Anima patent + paper document pack as PDFs on Desktop."""

from __future__ import annotations

import re
import shutil
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parent.parent
PATENT_DIR = ROOT / "docs" / "patent"
DESKTOP = Path.home() / "OneDrive" / "Desktop"
if not DESKTOP.is_dir():
    DESKTOP = Path.home() / "Desktop"

TODAY = date.today().isoformat()
DISCLOSURE = "2026-05-17"
US_DEADLINE = "2027-05-17"
BENCH_DATE = "2026-07-06"
GIT_SHA = "65a175f"

# Authoritative council + manifest numbers (2026-07-06)
MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 94.0, True, 0.188, None, None, 0.488, -0.185, 0.304),
    ("Qwen/Qwen2.5-0.5B-Instruct", 91.0, True, 0.236, 0.418, None, 0.267, 0.105, 0.372),
    ("distilgpt2", 82.2, True, 0.163, 0.000, 0.001, 0.314, 0.281, 0.595),
    ("EleutherAI/pythia-410m", 94.0, True, 0.239, 0.393, None, 0.664, -0.440, 0.224),
    ("facebook/opt-350m", 67.8, True, 0.150, 0.300, None, -0.073, -0.008, -0.081),
    ("HuggingFaceTB/SmolLM2-1.7B-Instruct", 58.5, False, 0.000, None, None, -0.149, 0.917, 0.768),
    ("hf-internal-testing/tiny-random-gpt2", 50.2, False, 0.004, None, -0.109, -0.004, 0.147, 0.143),
    ("meta-llama/Llama-3.2-1B-Instruct", 94.0, True, 0.356, 0.463, None, 0.752, -0.356, 0.396),
    ("google/gemma-2-2b-it", 58.5, False, 0.000, 0.000, None, 0.000, 1.000, 1.000),
]

THRESHOLDS = [
    ("VALENCE_SUPPRESSION_THRESHOLD", "0.35", "core/suppression.py"),
    ("UNCERTAINTY_OVERCLAIM_THRESHOLD", "−0.30", "core/suppression.py"),
    ("Stability window W", "8 tokens", "probes/guard_config.yaml"),
    ("Stability score threshold", "0.45", "probes/guard_config.yaml"),
    ("Stability weights", "0.45 vol + 0.35 delta + 0.20 abstain", "core/stability.py"),
    ("Guard composite", "0.55·fused_cal + 0.45·probe_u", "core/guard.py"),
    ("fused_abstain", "0.82", "probes/guard_config.yaml"),
    ("probe_uncertainty_abstain", "0.78", "probes/guard_config.yaml"),
    ("hedge_words_threshold", "2", "probes/guard_config.yaml"),
    ("Fused uncertainty", "0.35 entropy + 0.35 logit_gap + 0.30 attn_entropy", "core/extractor.py"),
    ("Dampen alpha", "0.12", "core/intervention.py"),
    ("Dampen valence delta", "0.25", "core/intervention.py"),
]


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\u2212", "-")
        .replace("\u2265", ">=")
        .replace("\u2264", "<=")
        .replace("\u2020", "(dagger)")
        .replace("\u2713", "YES")
        .replace("\u2717", "NO")
    )


def _md_inline(text: str) -> str:
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier' size='9'>\1</font>", text)
    return text


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("T", parent=base["Title"], fontSize=15, spaceAfter=12, textColor=colors.HexColor("#1a1a2e")),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontSize=12, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontSize=10.5, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontSize=10, spaceBefore=6, spaceAfter=3),
        "body": ParagraphStyle("B", parent=base["Normal"], fontSize=9.5, leading=13, spaceAfter=5),
        "bullet": ParagraphStyle("Bu", parent=base["Normal"], fontSize=9.5, leading=13, leftIndent=14, bulletIndent=4, spaceAfter=3),
        "mono": ParagraphStyle("M", parent=base["Code"], fontSize=8, leading=10, fontName="Courier", spaceAfter=4),
        "small": ParagraphStyle("S", parent=base["Normal"], fontSize=8, leading=10, textColor=colors.grey),
    }


def md_to_story(md: str, styles: dict) -> list:
    story = []
    in_code = False
    code_lines: list[str] = []

    for raw in md.splitlines():
        line = raw.rstrip()

        if line.strip().startswith("```"):
            if in_code:
                story.append(Paragraph(_esc("\n".join(code_lines)).replace("\n", "<br/>"), styles["mono"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            story.append(Spacer(1, 0.06 * inch))
            continue

        if line.strip() == "---":
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.lightgrey))
            continue

        if line.startswith("# "):
            story.append(Paragraph(_md_inline(line[2:]), styles["title"]))
            continue
        if line.startswith("## "):
            story.append(Paragraph(_md_inline(line[3:]), styles["h1"]))
            continue
        if line.startswith("### "):
            story.append(Paragraph(_md_inline(line[4:]), styles["h2"]))
            continue

        if line.startswith("|") and "|" in line[1:]:
            story.append(Paragraph(_md_inline(line), styles["mono"]))
            continue

        if line.lstrip().startswith("- ") or line.lstrip().startswith("* "):
            body = line.lstrip()[2:]
            story.append(Paragraph(f"&bull; {_md_inline(body)}", styles["bullet"]))
            continue

        m = re.match(r"^(\d+)\.\s+(.+)$", line.lstrip())
        if m:
            story.append(Paragraph(f"{m.group(1)}. {_md_inline(m.group(2))}", styles["bullet"]))
            continue

        if line.startswith(">"):
            story.append(Paragraph(_md_inline(line.lstrip("> ").strip()), styles["body"]))
            continue

        story.append(Paragraph(_md_inline(line), styles["body"]))

    return story


def write_pdf_from_md(out_path: Path, md: str) -> Path:
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
    )
    styles = _styles()
    story = md_to_story(md, styles)
    doc.build(story)
    return out_path


def usa_plan_md() -> str:
    rows = "\n".join(
        f"| {m[0]} | {m[1] if m[1] is not None else '—'} | "
        f"{'YES' if m[2] else ('—' if m[1] is None else 'NO')} | "
        f"{f'{m[3]:.3f}' if m[3] is not None else '—'} / "
        f"{f'{m[4]:.3f}' if m[4] is not None else '—'} | "
        f"{f'{m[5]:.3f}' if m[5] is not None else '—'} | "
        f"{f'{m[6]:.3f}' if m[6] is not None else '—'} |"
        for m in MODELS
    )
    return f"""# Anima — USA Research Paper Submission Plan

**Siddarth Boggarapu** · {TODAY} · siddarthb078@gmail.com  
**Repository:** https://github.com/Siddarthb07/Anima · **Benchmark run:** {BENCH_DATE} · **git SHA:** {GIT_SHA}

---

## 1. Goal and honest framing

Submit Anima as a **US systems / interpretability paper** (workshop first), not a hallucination SOTA claim.

**One-line positioning:**
> Open-source instrumentation that reads valence/arousal/uncertainty from LLM hidden states during generation and exposes a multi-signal reliability layer — abstention recommendations, layer-disagreement flags, stability scoring, and optional corrective steering.

**Do NOT claim:** hallucination detector, brain alignment (without synthetic_minimal label), AUROC 1.00 without fixture footnote, "the model feels X."

---

## 2. Current benchmark numbers (authoritative)

### 2.1 Cross-model validity council ({BENCH_DATE})

Council rubric weights: schema 15% · probe signal 35% · honesty 20% · prompt separation 30%. Pass threshold: **≥60 aggregate**.

| Model | Council | Pass | GoE r (v/a) | Brain r (v) | Pos−Neg gap |
|-------|---------|------|-------------|-------------|-------------|
{rows}

**Hero model (paper + demo):** TinyLlama-1.1B-Chat — council **94.0**, gap **0.488**, negative prompt mean valence **−0.185**.  
**Backup demo:** Qwen2.5-0.5B — council **91.0**, GoE valence r **0.236**, arousal r **0.418**.

### 2.2 Guard fixture smoke tests (NOT hallucination detection)

| Benchmark | n | Abstain acc | AUROC | Note |
|-----------|---|-------------|-------|------|
| HaluEval guard fixture | 52 | 1.00 | 1.00 | Pre-labeled readout rows only |
| TruthfulQA guard fixture | 52 | 1.00 | 1.00 | Policy execution smoke test |
| Confusion (each) | — | tp=26 fp=0 tn=26 fn=0 | — | Perfect separability by fixture design |

### 2.3 Implementation constants (must match paper §3)

| Parameter | Value | Source |
|-----------|-------|--------|
{chr(10).join(f'| {t[0]} | {t[1]} | {t[2]} |' for t in THRESHOLDS)}

### 2.4 distilgpt2 brain track (synthetic_minimal only)

| Metric | Value |
|--------|-------|
| Narratives holdout Pearson r (valence / arousal) | −0.393 / −0.022 |
| Val MSE | 0.176 |
| GoEmotions validation r (valence / arousal) | 0.163 / 0.000 |
| Holdout story | lucy (train: pieman, tunnel) |

---

## 3. Acceptance probability estimates

| Submission target | Current draft | After E1–E4 experiments |
|-------------------|---------------|-------------------------|
| NeurIPS/ICML/ICLR **workshop** | 20–30% | **45–55%** |
| ACL/EMNLP **Findings / Demo** | 15–25% | **55–65%** |
| Main conference (reliability SOTA) | <15% | <20% — do not target |

---

## 4. Required experiments (acceptance drivers)

### E1 — Real guard evaluation (CRITICAL · 2 weeks · ~15 hrs)

Run guard on **live LM generations** labeled for hallucination risk — not fixture rows.

**Targets:** HaluEval-QA subset (n≥200) + TruthfulQA subset (n≥200) on TinyLlama hero model.  
**Report:** AUROC, precision/recall at abstain threshold 0.82, calibration plot, latency ms/token.

**Success bar for paper:** AUROC **≥0.65** on at least one benchmark (honest, not SOTA).

### E2 — Ablation study (1 week · ~8 hrs)

| Config | Expected table row |
|--------|-------------------|
| Guard fusion only | baseline |
| + layer-disagreement as abstain reason | delta AUROC |
| + stability gate (guard_mode=gate) | delta AUROC |
| − lexical hedging | delta AUROC |
| Full system | full row |

### E3 — Baselines (1 week · ~8 hrs)

- Semantic entropy (Farquhar et al., Nature 2024) — same prompt set  
- Self-consistency n=5 — same prompt set  
- Report compute: Anima = 1 forward pass vs entropy = 5+ samples

### E4 — MERA comparison (1 week · ~6 hrs)

Matched short QA / MCQ slice vs Hedstrom et al. (ICML 2025). Table: accuracy, abstain rate, inference cost.

---

## 5. Target US venues (ranked)

| Tier | Venue | Deadline note | Fit |
|------|-------|---------------|-----|
| A | NeurIPS Mech Interp / SafeGenAI workshop | Check 2026 CFP | Best first target |
| A | ICML interpretability workshop | Check 2026 CFP | Strong fit |
| A | ICLR interpretability workshop | Check 2026 CFP | Strong fit |
| B | ACL System Demonstrations | After E1 | Needs live demo URL |
| B | EMNLP Findings short paper | After E1–E4 | Needs real guard numbers |
| Preprint | arXiv cs.CL / cs.LG | **After US provisional decision** | Not before ~counsel OK |

---

## 6. 12-week timeline (hours budget ~10/week)

| Week | Dates (approx) | Deliverable | Hours |
|------|----------------|-------------|-------|
| 1 | Jul 14–20 | Counsel call + E1 runner scaffold | 10 |
| 2 | Jul 21–27 | E1 HaluEval live eval on TinyLlama | 12 |
| 3 | Jul 28–Aug 3 | E1 TruthfulQA + E2 ablation start | 12 |
| 4 | Aug 4–10 | E2 complete + architecture figure render | 10 |
| 5 | Aug 11–17 | E3 semantic entropy baseline | 10 |
| 6 | Aug 18–24 | E4 MERA comparison + paper v3 draft | 12 |
| 7 | Aug 25–31 | US provisional decision + arXiv if cleared | 5 |
| 8–9 | Sep 1–14 | HF Gradio Space (v2.1.0) + demo video | 15 |
| 10 | Sep 15–21 | Internal review + workshop formatting | 8 |
| 11–12 | Sep 22–Oct 5 | Submit Tier A workshop | 6 |

**Total estimated:** ~110 hours

---

## 7. Paper document checklist

- [x] Related Work filled (ITI, CAA, RepE, MERA, Factuality Probes)
- [x] Benchmark numbers reconciled to {BENCH_DATE} manifests
- [x] Signal integration honesty (parallel vs fused abstain)
- [ ] E1 live guard results table
- [ ] E2 ablation table
- [ ] E3 baseline comparison table
- [ ] Architecture figure (rendered PNG for submission)
- [ ] HF Space URL in abstract footnote

---

## 8. Sequencing with patent

| Step | Action | Deadline |
|------|--------|----------|
| 1 | US counsel timing opinion | ASAP |
| 2 | US provisional (if viable) | Before {US_DEADLINE} |
| 3 | arXiv post | After step 2 decision |
| 4 | Workshop submit | After E1 minimum |

**Public disclosure date:** {DISCLOSURE} · **US grace remaining:** ~10 months as of {TODAY}
"""


def india_plan_md() -> str:
    return f"""# Anima & Portfolio — India Patent Filing Plan

**Siddarth Boggarapu** · {TODAY} · Indian resident · Bangalore  
**Email:** siddarthb078@gmail.com · **Phone:** +91 9606754584

---

## 1. India law — what applies to Anima today

| Fact | Value |
|------|-------|
| Anima GitHub public since | **{DISCLOSURE}** |
| Repository URL | https://github.com/Siddarthb07/Anima |
| India Section 31 grace | Exhibitions / learned society only — **NOT GitHub** |
| US 12-month grace | Until **{US_DEADLINE}** |
| Form 25 (FFL) if US first | Required for Indian residents in many cases — confirm with agent |

**Verdict — Anima core (already on GitHub):** **NOT fileable in India** for disclosed subject matter.

---

## 2. Patentability estimates by invention

| Invention | India today | US today | Action |
|-----------|-------------|----------|--------|
| Anima 4-mechanism stack (public) | **0%** | 15–30% grant | US provisional only if counsel agrees |
| Unified guard fusion (undisclosed) | **35–45%** if filed before publish | 20–35% | Build private → IN provisional → GitHub |
| Real ds002345 v3 pipeline (undisclosed) | **25–35%** | 20–30% | After download; before Release v3.0 |
| NeuralVortex FNO on CFD data (if private) | **40–55%** | 35–50% | **Best India shot** — file before any public release |
| text2sql-rag Spider rebuild | **<10%** | <15% | Paper only, skip patent |
| Lexprobe legal RAG | Trade secret preferred | Trade secret | Patent = public spec in 18 months |

---

## 3. Anima — disclosed vs still fileable components

### Already public — CANNOT claim in India

All items below are in https://github.com/Siddarthb07/Anima since {DISCLOSURE}:

- AffectProbe with layer-softmax fusion (valence/arousal/uncertainty)
- Layer-disagreement (thresholds **0.35** / **−0.30**)
- Rolling stability (window **8**, threshold **0.45**, weights **0.45/0.35/0.20**)
- Guard fusion (composite **0.55/0.45**, abstain **0.82**, hedge **≥2** words)
- Conditional valence steering (α=**0.12**, delta threshold **0.25**)
- Multi-judge validity council (benchmarks/council.py)
- Narratives encoding pipeline (synthetic_minimal tier)
- TRIBEv2 surrogate ROI projection

### Future Anima — file India BEFORE next public release

**Candidate A — Unified guard integration (Priority 1 for Anima follow-on)**

Wire layer-disagreement + stability INTO `abstain_recommended` as first-class reasons.

| Item | Estimate |
|------|----------|
| Dev effort | 2–3 weeks (~20 hrs) |
| India provisional cost | ₹25,000–₹75,000 |
| Patentability if undisclosed | 35–45% |
| Must do before | Any git push to public Anima repo |

**Candidate B — Real ds002345 uncertainty-supervised probe training (v3.0)**

| Item | Estimate |
|------|----------|
| Dev effort | 40–80 hrs + ~100 GB disk |
| India patentability | 25–35% (narrow method claims) |
| Scientific value | High for paper; patent secondary |

---

## 4. Portfolio projects outside Anima

### NeuralVortex (HIGHEST India priority if still private)

FNO surrogate trained on your CFD simulators (vortex ring, BEMT propeller data).

| Item | Value |
|------|-------|
| Estimated India patentability | **40–55%** (parametric CFD→FNO→flight surrogate) |
| Estimated cost | ₹40,000–₹90,000 provisional |
| Protocol | Indian provisional **FIRST** → then GitHub + arXiv |
| Claim focus | Operator learning on self-generated CFD parameter sweeps for drone design |

### text2sql-rag — skip patent

Schema linking + sqlglot + Spider/BIRD is crowded (DAIL-SQL, defog, etc.). Target EMNLP/ACL paper instead.

### Lexprobe — trade secret over patent

Multi-service legal RAG stack; patent forces early publication of method.

---

## 5. India filing protocol (all future inventions)

1. **Build privately** — no GitHub, no demo, no paper, no LinkedIn post  
2. **Novelty search** — ₹15,000–₹40,000 via Indian patent agent  
3. **File Indian provisional** — Form 1/2, establishes priority date in India  
4. **Form 25 FFL** — if filing US within 6 weeks without Indian filing first  
5. **US provisional within 12 months** — if pursuing US (optional)  
6. **PCT within 12 months of Indian provisional** — if going international  
7. **Only then:** git push, arXiv, HF Space, portfolio site  

---

## 6. Budget summary (INR)

| Item | Cost estimate |
|------|---------------|
| Indian patent agent opinion (Anima dead-on-arrival check) | ₹5,000–₹15,000 |
| India provisional — NeuralVortex OR Unified Guard | ₹25,000–₹75,000 |
| US provisional (via US counsel) | $2,000–$5,000 USD |
| Novelty search (India) | ₹15,000–₹40,000 |
| **Recommended minimum spend** | ₹30,000 India slot on **NeuralVortex** if still private |

**Do NOT spend:** ₹50k+ trying to patent disclosed Anima core in India.

---

## 7. Decision matrix

| Question | Answer |
|----------|--------|
| File India for current Anima? | **No** |
| File US for current Anima? | **Maybe** — conditional on counsel, before {US_DEADLINE} |
| Best India patent target? | **NeuralVortex** (if undisclosed) or **Unified Guard** (build first) |
| File before GitHub ever again? | **Yes** — mandatory protocol |

---

## 8. Immediate actions (this week)

1. Email Indian patent agent: "GitHub public {DISCLOSURE} — any India path for Anima?" (expect **no**)  
2. Check NeuralVortex repo visibility — if private/local, **start India provisional immediately**  
3. Do **not** push unified-guard code to public GitHub until filing decision  
4. Query Form 25 requirements before any US provisional as Indian resident  
5. Allocate **₹50,000** patent budget: NeuralVortex OR Unified Guard, not disclosed Anima
"""


def benchmark_sheet_md() -> str:
    rows = "\n".join(
        f"| {m[0]} | {m[1] if m[1] is not None else '—'} | "
        f"{'YES' if m[2] else ('—' if m[1] is None else 'NO')} | "
        f"{f'{m[3]:.3f}' if m[3] is not None else '—'} | "
        f"{f'{m[4]:.3f}' if m[4] is not None else '—'} | "
        f"{f'{m[5]:.3f}' if m[5] is not None else '—'} | "
        f"{f'{m[6]:.3f}' if m[6] is not None else '—'} | "
        f"{f'{m[7]:.3f}' if len(m) > 7 and m[7] is not None else '—'} | "
        f"{f'{m[8]:.3f}' if len(m) > 8 and m[8] is not None else '—'} |"
        for m in MODELS
    )
    return f"""# Anima — Benchmark & Constants Data Sheet

**Generated:** {TODAY} · **Source run:** {BENCH_DATE} · **git SHA:** {GIT_SHA}  
**Manifest path:** benchmarks/reports/council_rollup.json

---

## Council rollup (7 models)

| Model | Council | Pass | GoE r (v) | GoE r (a) | Brain r (v) | Gap | Neg val | Pos val |
|-------|---------|------|-----------|-----------|-------------|-----|---------|---------|
{rows}

---

## Guard fixtures (all models — smoke test only)

| Field | Value |
|-------|-------|
| HaluEval n | 52 |
| TruthfulQA n | 52 |
| Abstain accuracy | 1.00 |
| AUROC composite | 1.00 |
| tp / fp / tn / fn (each) | 26 / 0 / 26 / 0 |
| Interpretation | Policy on pre-labeled readout rows — NOT live LM hallucination detection |

---

## Qwen2.5-0.5B-Instruct detail (latest_qwen manifest)

| Metric | Value |
|--------|-------|
| GoEmotions Pearson r (valence / arousal) | 0.236 / 0.418 |
| Train holdout val_pearson_valence (2000 samples, seed 42) | 0.33 |
| probe_origin | text_emotion |

---

## distilgpt2 detail (latest_distilgpt2 manifest)

| Metric | Value |
|--------|-------|
| GoEmotions Pearson r (valence / arousal) | 0.163 / 0.000 |
| Narratives holdout r (valence / arousal) | −0.393 / −0.022 |
| Val MSE | 0.176 |
| probe_origin | narratives_fMRI_synthetic_minimal |

---

## Implementation constants

| Parameter | Value | File |
|-----------|-------|------|
{chr(10).join(f'| {t[0]} | {t[1]} | {t[2]} |' for t in THRESHOLDS)}

---

## Jurisdiction dates

| Item | Date |
|------|------|
| Public GitHub disclosure | {DISCLOSURE} |
| US grace deadline (approx) | {US_DEADLINE} |
| Months US grace remaining (approx) | ~10 as of {TODAY} |
| India filing for disclosed Anima | Not available |
"""


def index_md(generated: list[tuple[str, str]]) -> str:
    lines = "\n".join(f"- **{name}** → `{fname}`" for name, fname in generated)
    return f"""# Anima Document Pack — Master Index

**Siddarth Boggarapu** · {TODAY}  
**Desktop folder:** {DESKTOP}

---

## Documents in this pack

{lines}

---

## Quick reference numbers

| Item | Value |
|------|-------|
| Public disclosure | {DISCLOSURE} |
| US grace deadline | {US_DEADLINE} |
| Best council model | TinyLlama-1.1B-Chat (**94.0**) |
| Best GoE valence r | Qwen2.5-0.5B (**0.236**) |
| Guard fixture n | 52 (smoke test only) |
| India patent for disclosed Anima | No |
| US provisional for Anima | Conditional yes |

---

## Recommended order of use

1. **Anima_Document_Index.pdf** (this file)  
2. **Anima_USA_Research_Paper_Plan.pdf** — paper roadmap  
3. **Anima_India_Patent_Plan.pdf** — India + portfolio patent strategy  
4. **Anima_Patent_Filing_Letter.pdf** — send to US counsel (fill attorney fields)  
5. **Anima_Annex_A_Invention_Disclosure.pdf** + **Anima_Annex_B_Prior_Art.pdf** — attach to counsel letter  
6. **Anima_Paper_Draft.pdf** — workshop submission base  
7. **Anima_Benchmark_Numbers.pdf** — copy-paste tables for paper  
8. **Anima_Council_Synthesis.pdf** + **Anima_Audit_Issues.pdf** — internal QA reference
"""


def main():
    DESKTOP.mkdir(parents=True, exist_ok=True)

    # Save markdown sources alongside PDFs
    md_sources = {
        "Anima_USA_Research_Paper_Plan.md": usa_plan_md(),
        "Anima_India_Patent_Plan.md": india_plan_md(),
        "Anima_Benchmark_Numbers.md": benchmark_sheet_md(),
    }
    for fname, content in md_sources.items():
        (DESKTOP / fname).write_text(content, encoding="utf-8")

    for src in [
        "PATENT_FILING_LETTER.md",
        "ANNEX_A_INVENTION_DISCLOSURE.md",
        "ANNEX_B_PRIOR_ART.md",
        "PAPER_DRAFT.md",
        "AUDIT_ISSUES.md",
        "COUNCIL_SYNTHESIS.md",
    ]:
        shutil.copy2(PATENT_DIR / src, DESKTOP / src.replace(".md", "_source.md").replace("PATENT_FILING", "Patent_Filing").replace("ANNEX_A", "Annex_A").replace("ANNEX_B", "Annex_B").replace("PAPER_DRAFT", "Paper_Draft").replace("AUDIT_ISSUES", "Audit_Issues").replace("COUNCIL_SYNTHESIS", "Council_Synthesis"))

    pdf_jobs: list[tuple[str, str]] = [
        ("Anima_USA_Research_Paper_Plan.pdf", usa_plan_md()),
        ("Anima_India_Patent_Plan.pdf", india_plan_md()),
        ("Anima_Benchmark_Numbers.pdf", benchmark_sheet_md()),
        ("Anima_Patent_Filing_Letter.pdf", (PATENT_DIR / "PATENT_FILING_LETTER.md").read_text(encoding="utf-8")),
        ("Anima_Annex_A_Invention_Disclosure.pdf", (PATENT_DIR / "ANNEX_A_INVENTION_DISCLOSURE.md").read_text(encoding="utf-8")),
        ("Anima_Annex_B_Prior_Art.pdf", (PATENT_DIR / "ANNEX_B_PRIOR_ART.md").read_text(encoding="utf-8")),
        ("Anima_Paper_Draft.pdf", (PATENT_DIR / "PAPER_DRAFT.md").read_text(encoding="utf-8")),
        ("Anima_Audit_Issues.pdf", (PATENT_DIR / "AUDIT_ISSUES.md").read_text(encoding="utf-8")),
        ("Anima_Council_Synthesis.pdf", (PATENT_DIR / "COUNCIL_SYNTHESIS.md").read_text(encoding="utf-8")),
    ]

    generated: list[tuple[str, str]] = []
    for fname, md in pdf_jobs:
        out = DESKTOP / fname
        write_pdf_from_md(out, md)
        generated.append((fname.replace(".pdf", ""), fname))
        print(f"OK  {out}")

    index_name = "Anima_Document_Index.pdf"
    write_pdf_from_md(DESKTOP / index_name, index_md(generated))
    print(f"OK  {DESKTOP / index_name}")

    print(f"\nDone — {len(generated) + 1} PDFs on Desktop: {DESKTOP}")


if __name__ == "__main__":
    main()
