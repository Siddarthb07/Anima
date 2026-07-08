"""API path sanitization and request validation (no HF weights)."""

from fastapi.testclient import TestClient


def test_generate_rejects_oversized_max_new_tokens():
    import api.server as srv

    client = TestClient(srv.app)
    resp = client.post(
        "/generate",
        json={"model": "hf-internal-testing/tiny-random-gpt2", "prompt": "Hi", "max_new_tokens": 9999},
    )
    assert resp.status_code == 422


def test_sanitize_narratives_root_strips_windows_path():
    import api.server as srv

    raw = r"C:\Users\dev\Anima\data\narratives_minimal"
    assert srv._sanitize_narratives_root(raw) == "data/narratives_minimal"


def test_api_key_required_when_configured(monkeypatch):
    import api.server as srv
    from probes.linear_probe import AffectProbe

    class _FakeEx:
        hidden_dim = 32
        layer_indices = [1]
        early_layer = 1
        late_layer = 1

        def extract(self, prompt, max_new_tokens, **kwargs):
            return []

        def cleanup(self):
            pass

    fake_probe = AffectProbe(32, 1)

    def _fake_get(model_name):
        return _FakeEx(), fake_probe, {"probe_origin": "test"}

    monkeypatch.setattr(srv, "get_extractor_and_probe", _fake_get)
    monkeypatch.setattr(srv, "get_tribe_encoder", lambda *a, **k: srv.TRIBEv2Encoder(32))
    monkeypatch.setenv("ANIMA_API_KEY", "secret")

    client = TestClient(srv.app)
    resp = client.post(
        "/generate",
        json={"model": "hf-internal-testing/tiny-random-gpt2", "prompt": "Hi", "max_new_tokens": 2},
    )
    assert resp.status_code == 401

    resp_ok = client.post(
        "/generate",
        json={"model": "hf-internal-testing/tiny-random-gpt2", "prompt": "Hi", "max_new_tokens": 2},
        headers={"X-Anima-API-Key": "secret"},
    )
    assert resp_ok.status_code == 200
