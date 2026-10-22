# Brain probe data card

## Published tiers

| Tier | Root path | `probe_origin` | Use in papers/apps |
|------|-----------|----------------|-------------------|
| **Synthetic minimal** | `data/narratives_minimal/` | `narratives_fMRI_synthetic_minimal` | Dev, CI, dashboard demo only |
| **Real Narratives** | User-provided `NARRATIVES_ROOT` → ds002345 subset | `narratives_fMRI` | Research-grade claims |

## Synthetic minimal (shipped today)

- **Source:** `scripts/build_synthetic_brain_dataset.py`
- **Marker:** `data/narratives_minimal/dataset_meta.json` → `"source": "synthetic_brain_minimal"`
- **Stories:** `pieman`, `tunnel`, `lucy`
- **Subjects:** `01`, `02`, `03`
- **BOLD:** 200-voxel synthetic grids with word-rate coupling (not human neuroimaging)

**Do not** describe this tier as OpenNeuro ds002345 or real fMRI.

## Real ds002345 (depth-push subset — 2026-10-23)

**Gate before any real-fMRI claim:**

```powershell
python scripts/validate_narratives_root.py --root $env:NARRATIVES_ROOT
# full 3-story layout: layout_ok=true AND recommended_probe_origin=narratives_fMRI
```

### Real subset (laptop depth push)

- OpenNeuro / FCP-INDI Narratives: **ds002345** (Nastase et al., 2021)
- Download date: **2026-10-23**
- Tool: `python scripts/download_narratives_minimal_bold.py` (curl from `s3://fcp-indi/.../narratives/`)
- `NARRATIVES_ROOT`: `data/ds002345`
- Stories: **pieman**, **lucy** (`tunnel` not downloaded yet — validate will flag missing BOLD)
- Subjects: **022, 026, 041**
- BOLD: raw `*_task-*_bold.nii.gz` from FCP-INDI (not fMRIPrep MNI preproc filenames)
- Depth metrics: `docs/notes/FMRI_DEPTH_PUSH.md` + `benchmarks/reports/fmri_depth_distilgpt2.json`
- **Standing claim:** word-rate encoding baselines are real; PCA→VA holdout r≈0.02 is **not** emotion alignment

To complete the 3-story layout: `python scripts/download_narratives_fcp_indi.py`

### Recommended laptop subset

- **Stories:** start with `pieman`, `tunnel`, `lucy` + 1–2 more from corpus
- **Subjects:** 8–12 with complete runs per story
- **Disk:** plan **15–40 GB** for subset; **80 GB free** on machine recommended with HF cache

### Download pointers

- Dataset: https://openneuro.org/datasets/ds002345
- Layout expected by `alignment/narratives_loader.py`: see [`scripts/download_narratives.md`](../scripts/download_narratives.md)
- Optional fetch helper: `python scripts/download_narratives_minimal.py --fetch-real`

```powershell
$env:NARRATIVES_ROOT="D:\data\ds002345_subset"
anima train --model distilgpt2 --narratives-root $env:NARRATIVES_ROOT
```

After training, confirm `probes/zoo/distilgpt2_narratives_pca.meta.json` has:

- `"probe_origin": "narratives_fMRI"`
- `"narratives_root": "<your path>"`
- `"train_stories"` / `"holdout_stories"` matching the split file
