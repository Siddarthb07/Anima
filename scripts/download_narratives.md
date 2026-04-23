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

1. **OpenNeuro web** — download selected subjects/stories to save space.
2. **DataLad** — `datalad install https://github.com/OpenNeuroDatasets/ds002345.git`
3. **openneuro-py** — `pip install openneuro-py` then use their download CLI for `ds002345`.

## Environment

```powershell
$env:NARRATIVES_ROOT="C:\data\ds002345"
anima train --model distilgpt2 --narratives-root $env:NARRATIVES_ROOT
```

Full dataset is large (~100GB+). For development, a subset with stories `pieman`, `tunnel`, and `lucy` is enough.
