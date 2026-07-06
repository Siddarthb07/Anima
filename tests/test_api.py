import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") != "1",
    reason="HF model tests: set RUN_HF_TESTS=1 (downloads weights; may require GPU/stable torch on Windows).",
)

from probes.linear_probe import AffectProbe
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    import api.server as srv
    import probes.zoo_io as zio

    monkeypatch.setattr(srv, "_registry", {})
    monkeypatch.setattr(srv, "_tribe_registry", {})
    monkeypatch.setattr(srv, "_probe_meta_cache", {})
    monkeypatch.setattr(srv, "_calib_cache", {})

    zoo_dir = tmp_path / "zoo"
    zoo_dir.mkdir(parents=True)
    slug = "distilgpt2"
    probe = AffectProbe(768, 2)
    torch.save({"state_dict": probe.state_dict(), "probe_origin": "test"}, zoo_dir / f"{slug}.pt")
    monkeypatch.setattr(zio, "ZOO_DIR", zoo_dir)

    return TestClient(srv.app)


def test_generate_endpoint(client):
    resp = client.post(
        "/generate",
        json={
            "model": "distilgpt2",
            "prompt": "The capital of France is",
            "max_new_tokens": 4,
            "detect_suppression": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "tokens" in data and isinstance(data["tokens"], list)
    assert data["tokens"]
    assert "summary" in data
    assert "probe_origin" in data["summary"]
    assert isinstance(data["suppression_events"], list)
    assert data["tokens"][0]["brain_alignment_note"]
    assert "tribe_v2" in data["tokens"][0]
    assert "guard" in data["tokens"][0]
    tv = data["tokens"][0]["tribe_v2"]
    assert tv["roi_scores"] and len(tv["roi_scores"]) >= 3
    aff = data["tokens"][0]["affect"]
    assert -1 <= aff["valence"] <= 1
    assert "stability_score" in data["summary"]
    assert "guard_mode" in data["summary"]


def test_generate_guard_gate_mode(client):
    resp = client.post(
        "/generate",
        json={
            "model": "distilgpt2",
            "prompt": "Hello",
            "max_new_tokens": 4,
            "guard_mode": "gate",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["summary"]["guard_mode"] == "gate"
