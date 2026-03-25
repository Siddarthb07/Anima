from fastapi.testclient import TestClient

from api.server import app


def test_health():
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_list():
    c = TestClient(app)
    r = c.get("/models")
    assert r.status_code == 200
    assert "models" in r.json()
    assert len(r.json()["models"]) >= 1
