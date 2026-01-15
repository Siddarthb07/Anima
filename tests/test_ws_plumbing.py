"""WebSocket /generate plumbing without downloading HF weights."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from fastapi.testclient import TestClient

from alignment.tribe_encoder import TRIBEv2Encoder
from probes.linear_probe import AffectProbe


class FakeExtractor:
    layer_indices = [0, 1]
    hidden_dim = 8
    early_layer = 0
    late_layer = 1

    def extract(self, prompt: str, max_new_tokens: int):
        h = torch.randn(8)
        hi = {
            "entropy": 0.95,
            "logit_gap": 0.92,
            "attn_entropy": 0.88,
            "fused": 0.91,
        }
        lo = {
            "entropy": 0.08,
            "logit_gap": 0.07,
            "attn_entropy": 0.09,
            "fused": 0.08,
        }
        rows = [
            {
                "token_id": 10,
                "token_text": "?",
                "activations": {0: h.clone(), 1: h.clone()},
                "uncertainty_signals": hi,
            },
            {
                "token_id": 11,
                "token_text": "!",
                "activations": {0: h.clone(), 1: h.clone()},
                "uncertainty_signals": lo,
            },
        ]
        return rows[: max(1, min(len(rows), max_new_tokens))]

    def extract_iter(self, prompt: str, max_new_tokens: int):
        yield from self.extract(prompt, max_new_tokens)


@pytest.fixture()
def ws_client(monkeypatch):
    import api.server as srv

    monkeypatch.setattr(srv, "_registry", {})
    monkeypatch.setattr(srv, "_tribe_registry", {})

    def fake_get(model_name: str):
        _ = model_name
        return FakeExtractor(), AffectProbe(8, 2)

    monkeypatch.setattr(srv, "get_extractor_and_probe", fake_get)
    monkeypatch.setattr(
        srv,
        "get_tribe_encoder",
        lambda model_name, hidden_dim: TRIBEv2Encoder(int(hidden_dim), seed=0),
    )

    return TestClient(srv.app)


def test_ws_streams_tokens_uncertainty_and_done(ws_client):
    with ws_client.websocket_connect("/ws/generate") as ws:
        ws.send_json(
            {
                "model": "fake-model",
                "prompt": "hello",
                "max_new_tokens": 2,
                "detect_suppression": False,
            }
        )
        m1 = ws.receive_json()
        assert m1["kind"] == "token"
        assert m1["readout"]["uncertainty_signals"]["fused"] == 0.91
        assert m1["readout"]["confidence_tier"] in ("HIGH", "MEDIUM", "LOW")
        assert m1["readout"]["tribe_v2"]["roi_scores"]
        assert "derived_va" in m1["readout"]["tribe_v2"]

        m2 = ws.receive_json()
        assert m2["kind"] == "token"
        assert m2["readout"]["uncertainty_signals"]["fused"] == 0.08

        m3 = ws.receive_json()
        assert m3["kind"] == "done"
        assert "summary" in m3
        assert m3["summary"]["tribe_v2_mean_rois"]


def test_ws_error_envelope_on_failure(ws_client, monkeypatch):
    import api.server as srv

    def boom(*_a, **_k):
        raise RuntimeError("deliberate failure for WS error path")

    monkeypatch.setattr(srv, "get_extractor_and_probe", boom)

    with ws_client.websocket_connect("/ws/generate") as ws:
        ws.send_json(
            {
                "model": "fake-model",
                "prompt": "hello",
                "max_new_tokens": 2,
                "detect_suppression": False,
            }
        )
        m = ws.receive_json()
        assert m["kind"] == "error"
        assert "deliberate failure" in m["message"]
