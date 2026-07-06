"""Council scoring tests."""

from benchmarks.council import score_manifest


def test_council_distilgpt2_manifest():
    manifest = {
        "manifest_schema_version": 1,
        "model": "distilgpt2",
        "generated_at": "2026-01-01T00:00:00Z",
        "entries": [
            {"benchmark": "smoke_extract", "status": "ok", "n_tokens": 4},
            {
                "benchmark": "go_emotions",
                "status": "ok",
                "pearson_valence": 0.16,
                "probe_origin": "text_emotion",
            },
            {
                "benchmark": "halueval",
                "status": "ok",
                "n_samples": 52,
                "auroc_composite": 1.0,
                "abstain_accuracy": 1.0,
            },
        ],
    }
    examples = [
        {"id": "positive", "mean_valence": 0.35},
        {"id": "negative", "mean_valence": -0.15},
    ]
    report = score_manifest(manifest, examples)
    assert report.model == "distilgpt2"
    assert 0 <= report.aggregate_score <= 100
    assert len(report.verdicts) == 4
