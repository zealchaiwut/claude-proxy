"""Tests for issue #43: GET /metrics rolling summary endpoint.

AC coverage:
- ac-structure: GET /metrics returns 200 with {"profiles": {...}} body
- ac-fields: each profile entry has all 7 required fields with correct types
- ac-in-memory: collector.snapshot() never touches JSONL file
- ac-exclusions: /health and /metrics calls do not appear in profiles
- ac-errors: error responses increment error_count; tokens/cost NOT added
- ac-window: METRICS_WINDOW_SECONDS drops samples older than the window
- ac-multiprofile: two distinct profiles show isolated counts
- ac-pytest: per-profile request_count, tokens, cost, error_count assertions
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig
from services.metrics_collector import MetricsCollector

STUB_ANTHROPIC = "http://stub-anthropic.test"
CLIENT_MODEL = "claude-haiku-4-5-20251001"

_COST_PER_INPUT = 3.0 / 1_000_000
_COST_PER_OUTPUT = 15.0 / 1_000_000

REQUIRED_FIELDS = {
    "request_count",
    "error_count",
    "total_input_tokens",
    "total_output_tokens",
    "total_est_cost_usd",
    "p50_latency_ms",
    "p95_latency_ms",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_body(input_tokens: int = 10, output_tokens: int = 5) -> bytes:
    return json.dumps({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi"}],
        "model": CLIENT_MODEL,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }).encode()


def _error_body() -> bytes:
    return json.dumps({"error": {"type": "invalid_request_error", "message": "bad"}}).encode()


def _request_body(model: str = CLIENT_MODEL) -> bytes:
    return json.dumps({
        "model": model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hello"}],
    }).encode()


def _registry(*names: str) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        n: ProfileConfig(kind="passthrough", upstream=STUB_ANTHROPIC) for n in names
    }))


def _http_client(body: bytes, status: int = 200) -> object:
    async def _post(url, *, content, headers, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status
        resp.content = body
        resp.headers = {"content-type": "application/json"}
        return resp

    m = MagicMock()
    m.post = _post
    m.aclose = AsyncMock()
    return m


def _setup(client, registry: ProfileRegistry, collector: MetricsCollector) -> None:
    app.state.http_client = client
    app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
    app.state.profile_registry = registry
    app.state.config_from_file = True
    app.state.request_logger = None
    app.state.metrics_collector = collector


def _post(tc, profile: str, body: bytes = None) -> None:
    tc.post(
        "/v1/messages",
        content=body or _request_body(),
        headers={"content-type": "application/json", "x-ccproxy-profile": profile},
    )


# ---------------------------------------------------------------------------
# AC: structure — 200 with {"profiles": {}}
# ---------------------------------------------------------------------------


def test_metrics_returns_200_and_profiles_key():
    """GET /metrics must return 200 with a top-level 'profiles' key."""
    collector = MetricsCollector()
    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body()), _registry("alpha"), collector)
        resp = tc.get("/metrics")
    assert resp.status_code == 200
    assert "profiles" in resp.json()


# ---------------------------------------------------------------------------
# AC: fields — required 7 fields present and correct types
# ---------------------------------------------------------------------------


def test_metrics_profile_has_all_required_fields():
    """Each profile entry must expose all 7 required fields."""
    collector = MetricsCollector()
    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5)), _registry("alpha"), collector)
        _post(tc, "alpha")
        resp = tc.get("/metrics")
    profile = resp.json()["profiles"]["alpha"]
    missing = REQUIRED_FIELDS - set(profile.keys())
    assert not missing, f"Missing fields: {missing}"


def test_metrics_field_types():
    """All 7 fields must have correct numeric types."""
    collector = MetricsCollector()
    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5)), _registry("alpha"), collector)
        _post(tc, "alpha")
        resp = tc.get("/metrics")
    p = resp.json()["profiles"]["alpha"]
    assert isinstance(p["request_count"], int)
    assert isinstance(p["error_count"], int)
    assert isinstance(p["total_input_tokens"], int)
    assert isinstance(p["total_output_tokens"], int)
    assert isinstance(p["total_est_cost_usd"], float)
    assert isinstance(p["p50_latency_ms"], float)
    assert isinstance(p["p95_latency_ms"], float)


# ---------------------------------------------------------------------------
# AC: multiprofile — two profiles, correct isolated counts
# ---------------------------------------------------------------------------


def test_two_profiles_isolated_request_counts():
    """After requests to two profiles, request_count must be isolated per-profile."""
    collector = MetricsCollector()
    registry = _registry("alpha", "beta")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(100, 50)), registry, collector)
        for _ in range(3):
            _post(tc, "alpha")
        app.state.http_client = _http_client(_anthropic_body(200, 100))
        for _ in range(2):
            _post(tc, "beta")
        resp = tc.get("/metrics")

    profiles = resp.json()["profiles"]
    assert profiles["alpha"]["request_count"] == 3
    assert profiles["beta"]["request_count"] == 2


def test_two_profiles_isolated_token_counts():
    """Token accumulation must be isolated per profile."""
    collector = MetricsCollector()
    registry = _registry("alpha", "beta")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(100, 50)), registry, collector)
        for _ in range(3):
            _post(tc, "alpha")
        app.state.http_client = _http_client(_anthropic_body(200, 100))
        for _ in range(2):
            _post(tc, "beta")
        resp = tc.get("/metrics")

    profiles = resp.json()["profiles"]
    assert profiles["alpha"]["total_input_tokens"] == 300   # 3 × 100
    assert profiles["alpha"]["total_output_tokens"] == 150  # 3 × 50
    assert profiles["beta"]["total_input_tokens"] == 400    # 2 × 200
    assert profiles["beta"]["total_output_tokens"] == 200   # 2 × 100


# ---------------------------------------------------------------------------
# AC: errors — error_count increments, tokens/cost NOT added
# ---------------------------------------------------------------------------


def test_error_increments_error_count():
    """A non-2xx response must increment error_count."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_error_body(), status=400), registry, collector)
        _post(tc, "alpha")
        resp = tc.get("/metrics")

    p = resp.json()["profiles"]["alpha"]
    assert p["error_count"] == 1
    assert p["request_count"] == 1


def test_error_does_not_add_to_token_totals():
    """Tokens from a non-2xx response must NOT appear in token/cost totals."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        # 1 success: 100 in / 50 out
        _setup(_http_client(_anthropic_body(100, 50)), registry, collector)
        _post(tc, "alpha")
        # 1 error
        app.state.http_client = _http_client(_error_body(), status=400)
        _post(tc, "alpha")
        resp = tc.get("/metrics")

    p = resp.json()["profiles"]["alpha"]
    assert p["request_count"] == 2
    assert p["error_count"] == 1
    assert p["total_input_tokens"] == 100   # only from the success
    assert p["total_output_tokens"] == 50
    assert p["total_est_cost_usd"] > 0      # cost from the 1 success


def test_error_latency_is_still_recorded():
    """Latency must be recorded even for error responses."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_error_body(), status=500), registry, collector)
        _post(tc, "alpha")
        resp = tc.get("/metrics")

    p = resp.json()["profiles"]["alpha"]
    assert p["p50_latency_ms"] >= 0
    assert p["p95_latency_ms"] >= 0


# ---------------------------------------------------------------------------
# AC: cost is computed for success requests
# ---------------------------------------------------------------------------


def test_cost_accumulated_for_success():
    """total_est_cost_usd must be positive after a successful request with tokens."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(1000, 500)), registry, collector)
        _post(tc, "alpha")
        resp = tc.get("/metrics")

    p = resp.json()["profiles"]["alpha"]
    assert p["total_est_cost_usd"] > 0


def test_cost_matches_expected_formula():
    """total_est_cost_usd must equal input*rate_in + output*rate_out."""
    collector = MetricsCollector()
    registry = _registry("alpha")
    IN, OUT = 1_000_000, 1_000_000
    expected_cost = IN * _COST_PER_INPUT + OUT * _COST_PER_OUTPUT

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(IN, OUT)), registry, collector)
        _post(tc, "alpha")
        resp = tc.get("/metrics")

    p = resp.json()["profiles"]["alpha"]
    assert abs(p["total_est_cost_usd"] - expected_cost) < 1e-9


# ---------------------------------------------------------------------------
# AC: exclusions — /health and /metrics not self-counted
# ---------------------------------------------------------------------------


def test_health_calls_not_counted():
    """/health calls must not appear in any profile entry."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5)), registry, collector)
        # One real request to create a profile
        _post(tc, "alpha")
        saved_count = tc.get("/metrics").json()["profiles"]["alpha"]["request_count"]
        # Now hit /health repeatedly
        for _ in range(5):
            tc.get("/health")
        resp = tc.get("/metrics")

    profiles = resp.json()["profiles"]
    assert profiles["alpha"]["request_count"] == saved_count, \
        "/health calls must not increment any profile's request_count"


def test_metrics_calls_not_counted():
    """/metrics calls must not appear in any profile entry."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5)), registry, collector)
        _post(tc, "alpha")
        initial = tc.get("/metrics").json()["profiles"]["alpha"]["request_count"]
        for _ in range(5):
            tc.get("/metrics")
        final = tc.get("/metrics").json()["profiles"]["alpha"]["request_count"]

    assert final == initial, "/metrics calls must not increment any profile's request_count"


# ---------------------------------------------------------------------------
# AC: window — METRICS_WINDOW_SECONDS drops aged samples (unit-level)
# ---------------------------------------------------------------------------


def test_window_seconds_drops_expired_samples(monkeypatch):
    """Samples older than METRICS_WINDOW_SECONDS must be excluded from snapshot."""
    monkeypatch.setenv("METRICS_WINDOW_SECONDS", "0.001")  # 1 ms window
    collector = MetricsCollector()

    collector.record(
        profile="alpha", status=200, latency_ms=5.0,
        input_tokens=10, output_tokens=5, cost_usd=0.001,
    )
    time.sleep(0.05)  # 50 ms — well past the 1 ms window

    snapshot = collector.snapshot()
    assert "alpha" not in snapshot or snapshot["alpha"]["request_count"] == 0


def test_window_seconds_keeps_recent_samples(monkeypatch):
    """Samples within METRICS_WINDOW_SECONDS must be included in snapshot."""
    monkeypatch.setenv("METRICS_WINDOW_SECONDS", "60")  # 60 s window
    collector = MetricsCollector()

    collector.record(
        profile="alpha", status=200, latency_ms=5.0,
        input_tokens=10, output_tokens=5, cost_usd=0.001,
    )

    snapshot = collector.snapshot()
    assert "alpha" in snapshot
    assert snapshot["alpha"]["request_count"] == 1


# ---------------------------------------------------------------------------
# AC: in-memory — snapshot does not read JSONL (structural test)
# ---------------------------------------------------------------------------


def test_snapshot_does_not_read_jsonl(tmp_path, monkeypatch):
    """MetricsCollector.snapshot() must never open any file."""
    import builtins
    monkeypatch.setenv("CCPROXY_LOG_FILE", str(tmp_path / "requests.jsonl"))

    collector = MetricsCollector()
    collector.record(profile="alpha", status=200, latency_ms=1.0)

    opened_files = []
    original_open = builtins.open

    def tracking_open(name, *args, **kwargs):
        opened_files.append(str(name))
        return original_open(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    collector.snapshot()

    log_file = str(tmp_path / "requests.jsonl")
    assert log_file not in opened_files, "snapshot() must not read the JSONL log file"


# ---------------------------------------------------------------------------
# AC: latency percentiles (unit-level)
# ---------------------------------------------------------------------------


def test_p50_p95_latency_computed_correctly():
    """p50 and p95 must reflect the distribution of recorded latency_ms values."""
    collector = MetricsCollector()
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    for ms in latencies:
        collector.record(profile="alpha", status=200, latency_ms=ms)

    snap = collector.snapshot()["alpha"]
    assert 40.0 <= snap["p50_latency_ms"] <= 60.0, f"p50={snap['p50_latency_ms']}"
    assert 90.0 <= snap["p95_latency_ms"] <= 100.0, f"p95={snap['p95_latency_ms']}"
