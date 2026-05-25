# Researcher quickstart (~10 minutes)

Reproduce **brain-aligned readouts** with published probes (no retraining).

## 1. Install

```bash
git clone https://github.com/Siddarthb07/Anima.git
cd Anima
pip install -e ".[dev]"
python scripts/download_zoo.py    # Release v1.1.0 CPU tier
```

## 2. Verify probes

```bash
python -c "
from probes.zoo_io import load_meta, probe_slug
s = probe_slug('distilgpt2')
m = load_meta(s, '_narratives_pca')
print('probe_origin', m.get('probe_origin'))
print('holdout', m.get('holdout_stories'))
print('val_r_valence', m.get('val_r_valence'))
"
```

Expect `narratives_fMRI_synthetic_minimal` until v1.2 real-FMRI Release.

## 3. Run API

```bash
anima api --port 8010
curl http://127.0.0.1:8010/models
curl -X POST http://127.0.0.1:8010/encode -H "Content-Type: application/json" \
  -d "{\"model\":\"distilgpt2\",\"text\":\"A calm walk turned tense.\"}"
```

`/models` returns `brain_data_tier`, `train_stories`, `holdout_stories`, and `brain_val_r_valence` when a brain checkpoint exists.

## 4. Re-run holdout benchmark

```bash
$env:NARRATIVES_ROOT=".\data\narratives_minimal"   # or your ds002345 path
anima benchmark --model distilgpt2 --tiers external
```

Manifest: `benchmarks/reports/latest_distilgpt2_manifest.json`

## 5. Train your own HF model

```bash
python scripts/download_narratives_minimal.py   # or set NARRATIVES_ROOT to ds002345
anima train --model <hf_id from core/layer_config.py> --narratives-root $NARRATIVES_ROOT
anima train-text --model <hf_id> --max-samples 1500
```

Slug rule: last segment of HF id, lowercased, hyphens → underscores.

## 6. Cite

```bibtex
@software{anima2026,
  author = {Boggarapu, Siddarth},
  title = {Anima: Open brain probes for Hugging Face causal language models},
  year = {2026},
  url = {https://github.com/Siddarthb07/Anima}
}
```

Dataset (when using real tier): OpenNeuro ds002345 (Nastase et al.).

## Limits

Read [`USAGE_AND_LIMITATIONS.md`](USAGE_AND_LIMITATIONS.md) before citing metrics in a paper or application.
