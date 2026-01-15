import os
from pathlib import Path

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

    monkeypatch.setattr(srv, "_registry", {})
    monkeypatch.setattr(srv, "_tribe_registry", {})

    def zoo(slug: str) -> Path:
        p = tmp_path / f"{slug}.pt"
        if not p.exists():
            probe = AffectProbe(768, 2)
            torch.save(probe.state_dict(), p)
        return p

    monkeypatch.setattr(srv, "_zoo_path", zoo)

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
    assert isinstance(data["suppression_events"], list)
    assert data["tokens"][0]["brain_alignment_note"]
    assert "tribe_v2" in data["tokens"][0]
    tv = data["tokens"][0]["tribe_v2"]
    assert tv["roi_scores"] and len(tv["roi_scores"]) >= 3
    assert "derived_va" in tv and "valence" in tv["derived_va"]
    assert tv["methodology_note"]
    aff = data["tokens"][0]["affect"]
    assert -1 <= aff["valence"] <= 1
    assert 0 <= aff["arousal"] <= 1
    assert 0 <= aff["uncertainty"] <= 1
