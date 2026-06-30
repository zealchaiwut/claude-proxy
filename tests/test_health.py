from fastapi.testclient import TestClient


def test_health_default_shape():
    """AC: GET /health returns 200 with {status: ok, upstream: <UPSTREAM_BASE_URL>}."""
    from main import app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["upstream"] == "https://api.anthropic.com"


def test_health_upstream_override():
    """AC: UPSTREAM_BASE_URL env var is reflected in /health response."""
    from main import app
    from config import get_settings, Settings

    custom = Settings(upstream_base_url="https://my-gateway.internal")
    app.dependency_overrides[get_settings] = lambda: custom
    try:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["upstream"] == "https://my-gateway.internal"
    finally:
        app.dependency_overrides.clear()
