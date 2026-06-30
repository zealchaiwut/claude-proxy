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
    """AC: UPSTREAM_BASE_URL env var is reflected in /health response via the default profile."""
    import os
    from main import app

    os.environ["UPSTREAM_BASE_URL"] = "https://my-gateway.internal"
    try:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["upstream"] == "https://my-gateway.internal"
    finally:
        del os.environ["UPSTREAM_BASE_URL"]
