# Downloading the Narratives fMRI dataset

Anima training and Tier-A benchmarks expect the [Narratives dataset](https://openneuro.org/datasets/ds002345) (OpenNeuro **ds002345**).

## Layout expected by `NarrativesLoader`

```
ds002345/
  stimuli/
    pieman.txt
    pieman_words.json
    ...
  sub-XXX/
    func/
      sub-XXX_task-pieman_*_bold.nii.gz
```

## Options

1. **Recommended laptop subset (FCP-INDI S3, ~700MB BOLD + stimuli)** — works on Python 3.9 without a full OpenNeuro dump:
   ```powershell
   python scripts/download_narratives_fcp_indi.py
   # writes data/ds002345 with pieman/tunnel/lucy + subjects 022/026/041
   ```
2. **OpenNeuro web** — download selected subjects/stories to save space.
3. **DataLad** — `datalad install https://github.com/OpenNeuroDatasets/ds002345.git`
4. **openneuro-py** — `pip install openneuro-py` (needs Python ≥3.10 for current releases; 2023.1.0 works on 3.9 but full-tree traversal is slow). Prefer the FCP-INDI script above for a bounded subset.

## Environment

```powershell
$env:NARRATIVES_ROOT=".\data\ds002345"
python scripts/validate_narratives_root.py --root $env:NARRATIVES_ROOT
anima train --model distilgpt2 --narratives-root $env:NARRATIVES_ROOT
```

Full OpenNeuro dump is large (~100GB+). For Anima brain-probe retrain, stories `pieman`, `tunnel`, and `lucy` (plus a few multi-story subjects) are enough.
