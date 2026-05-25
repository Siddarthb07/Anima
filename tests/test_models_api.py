"""GET /models exposes brain probe metadata without loading HF weights."""

from fastapi.testclient import TestClient

import api.server as srv


def test_list_models_brain_fields(monkeypatch, tmp_path):
    import json

    import probes.zoo_io as zio

    zoo = tmp_path / "zoo"
    zoo.mkdir()
    slug = "distilgpt2"
    meta = {
        "probe_origin": "narratives_fMRI_synthetic_minimal",
        "narratives_root": "/data/narratives_minimal",
        "train_stories": ["pieman", "tunnel"],
        "holdout_stories": ["lucy"],
        "val_r_valence": 0.28,
    }
    (zoo / f"{slug}_narratives_pca.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (zoo / f"{slug}_narratives_pca.pt").write_bytes(b"x")

    monkeypatch.setattr(zio, "ZOO_DIR", zoo)
    monkeypatch.setattr(srv, "_registry", {})
    monkeypatch.setattr(srv, "_tribe_registry", {})
    monkeypatch.setattr(srv, "_probe_meta_cache", {})
    monkeypatch.setattr(srv, "_calib_cache", {})

    client = TestClient(srv.app)
    resp = client.get("/models")
    assert resp.status_code == 200
    row = next(m for m in resp.json()["models"] if m["model_id"] == "distilgpt2")
    assert row["brain_data_tier"] == "synthetic_minimal"
    assert row["train_stories"] == ["pieman", "tunnel"]
    assert row["holdout_stories"] == ["lucy"]
    assert row["brain_val_r_valence"] == 0.28
