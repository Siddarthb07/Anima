import torch

from probes.linear_probe import AffectProbe


def test_encode_endpoint(monkeypatch, tmp_path):
    import api.server as srv
    import probes.zoo_io as zio

    monkeypatch.setattr(srv, "_registry", {})
    monkeypatch.setattr(srv, "_tribe_registry", {})
    monkeypatch.setattr(srv, "_probe_meta_cache", {})
    monkeypatch.setattr(srv, "_calib_cache", {})

    zoo_dir = tmp_path / "zoo"
    zoo_dir.mkdir()
    slug = "tiny_random_gpt2"
    probe = AffectProbe(32, 2)
    torch.save({"state_dict": probe.state_dict(), "probe_origin": "test"}, zoo_dir / f"{slug}.pt")
    monkeypatch.setattr(zio, "ZOO_DIR", zoo_dir)

    class FakeExtractor:
        hidden_dim = 32
        layer_indices = [0, 1]

        def encode_sequence(self, text, max_length=128):
            return [
                {
                    "token_index": 0,
                    "token_id": 1,
                    "token_text": "hi",
                    "activations": {0: torch.zeros(32), 1: torch.zeros(32)},
                    "uncertainty_signals": {
                        "entropy": 0.5,
                        "logit_gap": 0.5,
                        "attn_entropy": 0.5,
                        "fused": 0.5,
                    },
                }
            ]

        def cleanup(self):
            pass

    def fake_get(model_name):
        return FakeExtractor(), probe, {"probe_origin": "test"}

    monkeypatch.setattr(srv, "get_extractor_and_probe", fake_get)
    from alignment.tribe_encoder import TRIBEv2Encoder

    monkeypatch.setattr(srv, "get_tribe_encoder", lambda *a, **k: TRIBEv2Encoder(32, seed=0))

    from fastapi.testclient import TestClient

    c = TestClient(srv.app)
    r = c.post(
        "/encode",
        json={"model": "hf-internal-testing/tiny-random-gpt2", "text": "hello", "max_length": 32},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["tokens"]
    assert "probe_origin" in data["summary"]
