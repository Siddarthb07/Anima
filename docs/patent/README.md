# Anima Patent & Paper Package

Fixed drafts replacing the PDFs in `Downloads/` (2026-07-12). Source of truth for counsel and workshop submission.

| File | Purpose |
|------|---------|
| [PATENT_FILING_LETTER.md](PATENT_FILING_LETTER.md) | Cover letter to patent counsel — send after filling `[ATTORNEY]` placeholders |
| [ANNEX_A_INVENTION_DISCLOSURE.md](ANNEX_A_INVENTION_DISCLOSURE.md) | Technical invention disclosure for provisional spec |
| [ANNEX_B_PRIOR_ART.md](ANNEX_B_PRIOR_ART.md) | Prior art with verified citations (MERA, ITI, CAA, Factuality Probes, etc.) |
| [PAPER_DRAFT.md](PAPER_DRAFT.md) | System paper draft — workshop track, not hallucination SOTA |
| [AUDIT_ISSUES.md](AUDIT_ISSUES.md) | Issue log: what was wrong in original PDFs and what was fixed |
| [COUNCIL_SYNTHESIS.md](COUNCIL_SYNTHESIS.md) | LLM council verdict + priority action list |

## Key facts (confirmed)

- **Repo:** https://github.com/Siddarthb07/Anima
- **Public disclosure:** 2026-05-17 (GitHub repo created public)
- **US grace deadline (approx.):** 2027-05-17
- **India filing:** Likely unavailable — confirm with counsel
- **Inventor:** Siddarth Boggarapu · siddarthb078@gmail.com · +91 9606754584

## Sequencing

1. Counsel timing opinion (US / India / PCT) — **this week**
2. US provisional (if viable) — before grace expires
3. Real HaluEval/TruthfulQA guard eval + ablations — before paper
4. arXiv / workshop — **after** provisional decision

## Honest positioning

> Open-source instrumentation that reads valence/arousal/uncertainty from LLM hidden states and exposes a multi-signal reliability layer — abstention recommendations, layer-disagreement flags, stability scoring, and optional corrective steering.

Do **not** call it a hallucination detector until end-to-end guard benchmarks ship.

## Export to PDF

From repo root (requires pandoc):

```powershell
pandoc docs/patent/PATENT_FILING_LETTER.md -o "$env:USERPROFILE\Downloads\anima-patent-filing-letter-v2.pdf"
pandoc docs/patent/ANNEX_A_INVENTION_DISCLOSURE.md -o "$env:USERPROFILE\Downloads\anima-invention-disclosure-annexA-v2.pdf"
pandoc docs/patent/PAPER_DRAFT.md -o "$env:USERPROFILE\Downloads\anima-paper-draft-v2.pdf"
```
