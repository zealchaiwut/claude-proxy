"""Tests for issue #50: /health and /ready lifecycle endpoints."""
import os
import time
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_ready_cache():
    import routers.health as health_mod
    health_mod._ready_cache["result"] = None
    health_mod._ready_cache["expires_at"] = 0.0
    yield
    health_mod._ready_cache["result"] = None
    health_mod._ready_cache["expires_at"] = 0.0


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


# --- AC: GET /health response shape and field presence ---

def test_health_returns_200(client):
    """AC: /health always returns HTTP 200 while the process is running."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_has_all_required_fields(client):
    """AC: /health returns {status, version, active_default_profile, upstream}."""
    resp = client.get("/health")
    body = resp.json()
    assert "status" in body
    assert "version" in body
    assert "active_default_profile" in body
    assert "upstream" in body


def test_health_status_is_ok(client):
    """AC: status reflects process health — always 'ok' while running."""
    resp = client.get("/health")
    assert resp.json()["status"] == "ok"


def test_health_version_present(client):
    """AC: version field is present and non-empty."""
    resp = client.get("/health")
    version = resp.json()["version"]
    assert isinstance(version, str)
    assert len(version) > 0


def test_health_active_default_profile_present(client):
    """AC: active_default_profile field names the resolved default profile."""
    resp = client.get("/health")
    profile = resp.json()["active_default_profile"]
    assert isinstance(profile, str)
    assert len(profile) > 0


def test_health_upstream_present(client):
    """AC: upstream field shows the active default profile's upstream URL."""
    resp = client.get("/health")
    upstream = resp.json()["upstream"]
    assert isinstance(upstream, str)
    assert upstream.startswith("http")


def test_health_never_calls_upstream(client):
    """AC: /health never initiates a call to the upstream service."""
    import httpx
    with patch.object(httpx.AsyncClient, "send", side_effect=AssertionError("upstream called")) as mock_send:
        resp = client.get("/health")
        assert resp.status_code == 200
        mock_send.assert_not_called()


def test_health_no_secrets_in_response(client):
    """AC: /health response does not leak API keys or secrets."""
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-secret-value-12345")
    resp = client.get("/health")
    body_str = resp.text
    assert "sk-ant-test-secret-value-12345" not in body_str
    assert "api_key" not in body_str.lower()
    # Clean up only if we set it
    if os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-secret-value-12345":
        del os.environ["ANTHROPIC_API_KEY"]


# --- AC: GET /ready reachability probe ---

def test_ready_returns_200_when_ok(client):
    """AC: /ready HTTP 200 when upstream is reachable."""
    with patch("routers.health._tcp_probe", AsyncMock(return_value=True)):
        resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_returns_200_when_degraded(client):
    """AC: /ready HTTP 200 even when upstream is unreachable (body carries status)."""
    with patch("routers.health._tcp_probe", AsyncMock(return_value=False)):
        resp = client.get("/ready")
    assert resp.status_code == 200


def test_ready_status_ok_when_reachable(client):
    """AC: /ready returns {status: 'ok', profile: ...} when upstream is reachable."""
    with patch("routers.health._tcp_probe", AsyncMock(return_value=True)):
        resp = client.get("/ready")
    body = resp.json()
    assert body["status"] == "ok"
    assert "profile" in body


def test_ready_status_degraded_when_unreachable(client):
    """AC: /ready returns {status: 'degraded', profile: ...} when upstream unreachable."""
    with patch("routers.health._tcp_probe", AsyncMock(return_value=False)):
        resp = client.get("/ready")
    body = resp.json()
    assert body["status"] == "degraded"
    assert "profile" in body


def test_ready_profile_field_is_string(client):
    """AC: profile field in /ready response names the active default profile."""
    with patch("routers.health._tcp_probe", AsyncMock(return_value=True)):
        resp = client.get("/ready")
    profile = resp.json()["profile"]
    assert isinstance(profile, str)
    assert len(profile) > 0


def test_ready_no_secrets_in_response(client):
    """AC: /ready response does not leak API keys or secrets."""
    os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-secret-value-12345")
    with patch("routers.health._tcp_probe", AsyncMock(return_value=True)):
        resp = client.get("/ready")
    body_str = resp.text
    assert "sk-ant-test-secret-value-12345" not in body_str
    if os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-secret-value-12345":
        del os.environ["ANTHROPIC_API_KEY"]


# --- AC: /ready cache behaviour ---

def test_ready_cache_second_call_within_ttl_skips_probe(client):
    """AC: second call within TTL does not trigger a second upstream probe."""
    call_count = 0

    async def counting_probe(host, port, timeout):
        nonlocal call_count
        call_count += 1
        return True

    with patch("routers.health._tcp_probe", counting_probe):
        client.get("/ready")
        client.get("/ready")

    assert call_count == 1, f"Expected 1 probe within TTL, got {call_count}"


def test_ready_cache_expired_triggers_fresh_probe(client):
    """AC: after TTL expires a fresh probe is issued."""
    from main import app
    from config import get_settings, Settings

    short_ttl = Settings(ready_cache_ttl=0.01)
    app.dependency_overrides[get_settings] = lambda: short_ttl

    call_count = 0

    async def counting_probe(host, port, timeout):
        nonlocal call_count
        call_count += 1
        return True

    try:
        with patch("routers.health._tcp_probe", counting_probe):
            client.get("/ready")
            time.sleep(0.05)
            client.get("/ready")

        assert call_count == 2, f"Expected 2 probes after TTL expiry, got {call_count}"
    finally:
        app.dependency_overrides.clear()


def test_ready_cache_default_ttl_is_five_seconds(client):
    """AC: default cache TTL is 5 seconds (settings default)."""
    from config import Settings
    s = Settings()
    assert s.ready_cache_ttl == 5.0
