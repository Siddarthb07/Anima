# Data directory

| Path | Purpose |
|------|---------|
| `narratives_minimal/` | Minimal Narratives-**layout** dataset for brain-probe training (generated; synthetic BOLD). |
| `ds002345/` | Optional full OpenNeuro Narratives download (`--fetch-real`). |

Generate the minimal set (any machine):

```bash
python scripts/download_narratives_minimal.py
export NARRATIVES_ROOT="$(pwd)/data/narratives_minimal"
```

For published brain-alignment work, use real [ds002345](https://openneuro.org/datasets/ds002345) and set `NARRATIVES_ROOT` accordingly.
