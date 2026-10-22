# Real fMRI depth push — how far correlations go

**Date:** 2026-10-23  
**Data:** FCP-INDI Narratives subset `data/ds002345` — stories `pieman`+`lucy`, subjects `022/026/041` (tunnel omitted for size).  
**Model:** `distilgpt2`  
**Artifact:** `benchmarks/reports/fmri_depth_distilgpt2.json`  
**Split:** train `pieman` → holdout `lucy` (`benchmarks/splits/narratives_holdout_depth.json`)

This is a measurement note, not a paper draft.

---

## Three constructs (do not mix them)

| # | Construct | What it asks | Result on this subset |
|---|-----------|--------------|------------------------|
| 1 | **Word-rate → BOLD** | Do any voxels track speech rate? | **Yes — strong sparse signal** |
| 2 | **Anima PCA→VA probe** | Can whole-brain PCA axes be read as valence/arousal by an LM probe? | **No — ~noise on holdout** |
| 3 | **LM mid-layer → BOLD PCA** | Do LM TR features predict low-D BOLD components on a held-out story? | **Partial — max PC r≈0.42; mean≈0** |

---

## 1) Encoding baseline (most legible)

Word-rate predictivity on ISC-style mean BOLD (3 subjects), HRF lag 3 TRs, 8k voxel subsample:

| Story | mean r (all voxels) | max r | p95 r | fraction r>0.1 |
|-------|--------------------:|------:|------:|---------------:|
| pieman | −0.036 | **0.523** | 0.261 | 19.4% |
| lucy | −0.058 | **0.667** | 0.350 | 22.2% |

**Read:** whole-brain *average* correlation is ~0 (correct). A minority of voxels track the stimulus hard (max r 0.5–0.7). That is the classic Narratives / naturalistic encoding picture (Nastase et al. 2021).

**Why this is interesting:** it proves the downloaded BOLD is real signal, not synthetic grids, and gives a number admissions/readers recognize from encoding papers.

---

## 2) Product brain probe (PCA → valence/arousal) — ceiling is low

Holdout (`lucy`) after training on `pieman`:

| Metric | Value |
|--------|------:|
| `val_r_valence` | **0.022** |
| `val_r_arousal` | **NaN** (degenerate component) |
| `val_mse` | 0.859 |
| `probe_origin` | `narratives_fMRI` |
| n_train / n_val | 297 / 367 TRs |

**Read:** real fMRI does **not** rescue the VA product path. Mapping whole-brain PCA components to “valence/arousal” is the wrong scientific object. Prior synthetic holdout was already negative/noisy; real data stays near chance.

**Do not claim:** “brain-aligned emotion” or “fMRI-validated affect” from this probe.

---

## 3) Deeper path — LM features → BOLD PCA (held-out story)

RidgeCV on mid-layer TR features → 20 BOLD PCs (60% variance on train voxels):

| Holdout | mean r across PCs | max r across PCs |
|---------|------------------:|-----------------:|
| lucy | −0.049 | **0.422** |

Some PCs hit r≈0.3–0.42; others flip sign (unstable with **one** train story). This is more scientifically aligned with encoding literature than VA, but **not** ready as a headline without: tunnel+more subjects, language ROI mask, voxel-wise maps, and nested CV.

---

## How far can it be pushed? (honest ladder)

| Depth | Effort | Expected legibility |
|-------|--------|---------------------|
| A. Word-rate / lexical baselines (done) | hours | High — cite max/p95/frac |
| B. Add `tunnel` + more subjects | +0.5–1 day download/train | Stabilizes encoding |
| C. Language-parcel / atlas ROI encoding | 1–2 days | High for “LLM tracks auditory language cortex” |
| D. Voxel-wise ridge + ISC / Brain-Score-style | days–weeks | Paper-grade encoding (separate from trust-UI) |
| E. Keep pushing PCA→VA | wasted | Stays uninteresting |

TRIBEv2 remains a **surrogate ROI sketch** (`alignment/tribe_encoder.py`) — not voxel TRIBE.

---

## Does real fMRI make Anima more legible / interesting?

**Yes — if and only if the claim is encoding alignment (stimulus/LM → BOLD), not emotion-from-brain.**

| Storyline | Legible? | Interesting? |
|-----------|----------|--------------|
| Trust-UI / green-shield collapse (Protocol L/L′) | Yes | Yes — current hero |
| Text-emotion probes (GoEmotions) | Yes | Moderate |
| Synthetic “brain” VA probe | No | No — actively confusing |
| Real fMRI VA probe (r≈0.02) | Misleading if overclaimed | No |
| Real fMRI encoding baselines (max r 0.5–0.7) | **Yes** | **Yes** — rare for apps |
| Real fMRI LM→BOLD (with ROI/voxel maps later) | Yes when done right | Yes — scientific-ML spike |

**Recommendation:** keep trust-UI as the archival claim; add a short, honest **Narratives encoding** appendix with word-rate max/p95 + `probe_origin: narratives_fMRI` data card. Do **not** lead with VA brain correlations until the construct changes (atlas ROIs or encoding scores, not PCA→VA).

---

## Reproduce

```powershell
python scripts/download_narratives_minimal_bold.py   # pieman+lucy × 3 subs
$env:NARRATIVES_ROOT=".\data\ds002345"
python scripts/push_fmri_depth.py --model distilgpt2 --root data/ds002345
```

Full 3-story subset: `python scripts/download_narratives_fcp_indi.py` (adds tunnel ~440MB).
