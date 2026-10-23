# Anima paper package

IEEE-shaped draft for scrutiny. **Markdown narrative** is the readable source; **`.tex`** is a conference skeleton.

| File | Role |
|------|------|
| [CLAIM_SHEET.md](CLAIM_SHEET.md) | Allowed / forbidden claims |
| [ABSTRACT.md](ABSTRACT.md) | ≤250-word abstract |
| [OUTLINE.md](OUTLINE.md) | Section map |
| [ANIMA_IEEE_DRAFT.md](ANIMA_IEEE_DRAFT.md) | Full prose draft + tables |
| [anima_ieee_draft.tex](anima_ieee_draft.tex) | IEEEtran skeleton |
| [RESULTS_LOCKED.md](RESULTS_LOCKED.md) | Figure ↔ claim map |
| [figures/](figures/) | PNG + PDF from sealed JSON |

```bash
python scripts/render_paper_figures.py
# optional: pdflatex anima_ieee_draft.tex  (needs IEEEtran)
```

Drafting norms used: IEEE Author Center structure; captions below figures / above tables; claim–evidence ladder; REFORMS reproducibility pointers; explicit limitations.
