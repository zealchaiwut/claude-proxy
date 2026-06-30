"""Tests for issue #55: tokenizer drift measurement and surfacing.

AC coverage:
- ac1: Each M5 request record includes token_drift_input and token_drift_output
- ac2: Drift computed as proxy_estimated - upstream_reported
- ac3: /metrics exposes per-profile drift aggregates (mean, abs-mean, input, output)
- ac4: Fixture with known estimated vs reported counts produces exact drift values
- ac5: Unit tests for drift computation; integration tests for drift aggregation
- ac6: Existing metrics fields and request-record fields unchanged
"""

import json
from io import StringIO
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig
from services.cost_accounting import (
    count_input_tokens,
    count_output_tokens,
    extract_text_from_sse,
    extract_upstream_usage_from_sse,
)
from services.metrics_collector import MetricsCollector
from services.request_logger import RequestLogger

STUB_ANTHROPIC = "http://stub-anthropic.test"
CLIENT_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_body(input_tokens: int = 10, output_tokens: int = 5, text: str = "Hi") -> bytes:
    return json.dumps({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": CLIENT_MODEL,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }).encode()


def _no_usage_body(text: str = "Hi") -> bytes:
    """Response body without usage field — simulates upstream that reports no token counts."""
    return json.dumps({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": CLIENT_MODEL,
        "stop_reason": "end_turn",
    }).encode()


def _request_body(content: str = "hello") -> bytes:
    return json.dumps({
        "model": CLIENT_MODEL,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": content}],
    }).encode()


def _registry(*names: str) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        n: ProfileConfig(kind="passthrough", upstream=STUB_ANTHROPIC) for n in names
    }))


def _http_client(body: bytes, status: int = 200):
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


def _setup(
    client,
    registry: ProfileRegistry,
    collector: MetricsCollector,
    capture: StringIO | None = None,
) -> RequestLogger | None:
    app.state.http_client = client
    app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
    app.state.profile_registry = registry
    app.state.config_from_file = True
    app.state.metrics_collector = collector
    if capture is not None:
        rl = RequestLogger(capture=capture)
        app.state.request_logger = rl
        return rl
    app.state.request_logger = None
    return None


def _post_request(tc, profile: str, body: bytes) -> None:
    tc.post(
        "/v1/messages",
        content=body,
        headers={"content-type": "application/json", "x-ccproxy-profile": profile},
    )


# ---------------------------------------------------------------------------
# AC4 + AC5 (unit): drift computation with known estimated vs reported counts
# ---------------------------------------------------------------------------


def test_drift_computation_exact_values():
    """AC4: fixture with known estimated vs reported produces exact drift values.

    content='hello' (5 chars) → est_input = max(1, 5//4) = 1
    upstream_input = 10  →  drift_input = 1 - 10 = -9
    response text='Hi' (1 word) → est_output = 1
    upstream_output = 5  →  drift_output = 1 - 5 = -4
    """
    body_json = {"messages": [{"role": "user", "content": "hello"}]}
    est_input = count_input_tokens(body_json)
    assert est_input == 1  # 5 chars // 4 = 1

    est_output = count_output_tokens("Hi")
    assert est_output == 1  # 1 word

    upstream_input, upstream_output = 10, 5

    drift_input = est_input - upstream_input
    drift_output = est_output - upstream_output

    assert drift_input == -9
    assert drift_output == -4


def test_extract_upstream_usage_from_sse_present():
    """AC5 (unit): extract_upstream_usage_from_sse returns (input, output) when SSE has usage."""
    sse = (
        b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 100}}}\n\n'
        b'data: {"type": "message_delta", "usage": {"output_tokens": 50}}\n\n'
    )
    result = extract_upstream_usage_from_sse(sse)
    assert result == (100, 50)


def test_extract_upstream_usage_from_sse_absent():
    """AC5 (unit): extract_upstream_usage_from_sse returns None when SSE has no usage."""
    sse = b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}\n\n'
    result = extract_upstream_usage_from_sse(sse)
    assert result is None


def test_extract_upstream_usage_from_sse_empty():
    """AC5 (unit): extract_upstream_usage_from_sse returns None for empty data."""
    assert extract_upstream_usage_from_sse(b"") is None


def test_extract_text_from_sse():
    """AC5 (unit): extract_text_from_sse concatenates text_delta content."""
    sse = (
        b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}}\n\n'
        b'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "world"}}\n\n'
    )
    text = extract_text_from_sse(sse)
    assert text == "Hello world"


def test_extract_text_from_sse_empty():
    """AC5 (unit): extract_text_from_sse returns empty string when no text deltas."""
    sse = b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 5}}}\n\n'
    text = extract_text_from_sse(sse)
    assert text == ""


# ---------------------------------------------------------------------------
# AC1 + AC2: request log record drift fields
# ---------------------------------------------------------------------------


def test_request_record_has_drift_fields_when_upstream_usage_present():
    """AC1: log record includes token_drift_input and token_drift_output (non-null)
    when upstream reports usage."""
    capture = StringIO()
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5, "Hi")), registry, collector, capture)
        _post_request(tc, "alpha", _request_body("hello"))

    lines = [l for l in capture.getvalue().splitlines() if l.startswith("{")]
    assert lines, "No JSON log records emitted"
    record = json.loads(lines[0])
    assert "token_drift_input" in record
    assert "token_drift_output" in record
    assert record["token_drift_input"] is not None
    assert record["token_drift_output"] is not None


def test_request_record_drift_null_when_no_upstream_usage():
    """AC1: token_drift_input and token_drift_output are null when upstream omits usage."""
    capture = StringIO()
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_no_usage_body("Hi")), registry, collector, capture)
        _post_request(tc, "alpha", _request_body("hello"))

    lines = [l for l in capture.getvalue().splitlines() if l.startswith("{")]
    assert lines, "No JSON log records emitted"
    record = json.loads(lines[0])
    assert "token_drift_input" in record
    assert "token_drift_output" in record
    assert record["token_drift_input"] is None
    assert record["token_drift_output"] is None


def test_request_record_drift_values_match_formula():
    """AC2: drift = proxy_estimated - upstream_reported for both fields.

    content='hello' → est_input=1, upstream_input=10 → drift_input=-9
    text='Hi'       → est_output=1, upstream_output=5 → drift_output=-4
    """
    capture = StringIO()
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5, "Hi")), registry, collector, capture)
        _post_request(tc, "alpha", _request_body("hello"))

    lines = [l for l in capture.getvalue().splitlines() if l.startswith("{")]
    record = json.loads(lines[0])
    assert record["token_drift_input"] == -9, f"Expected -9, got {record['token_drift_input']}"
    assert record["token_drift_output"] == -4, f"Expected -4, got {record['token_drift_output']}"


# ---------------------------------------------------------------------------
# AC3 + AC5 (integration): /metrics exposes drift aggregates
# ---------------------------------------------------------------------------


def test_metrics_has_drift_aggregate_fields():
    """AC3: /metrics response includes all four drift aggregate fields per profile."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5, "Hi")), registry, collector)
        _post_request(tc, "alpha", _request_body("hello"))
        resp = tc.get("/metrics")

    profile = resp.json()["profiles"]["alpha"]
    for field in ("mean_drift_input", "abs_mean_drift_input", "mean_drift_output", "abs_mean_drift_output"):
        assert field in profile, f"{field} missing from /metrics profile"


def test_metrics_drift_values_correct_single_request():
    """AC3+AC5: drift aggregate values correct after one request with known counts.

    drift_input=-9, drift_output=-4
    → mean_drift_input=-9.0, abs_mean_drift_input=9.0
    → mean_drift_output=-4.0, abs_mean_drift_output=4.0
    """
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5, "Hi")), registry, collector)
        _post_request(tc, "alpha", _request_body("hello"))
        resp = tc.get("/metrics")

    profile = resp.json()["profiles"]["alpha"]
    assert profile["mean_drift_input"] == -9.0
    assert profile["abs_mean_drift_input"] == 9.0
    assert profile["mean_drift_output"] == -4.0
    assert profile["abs_mean_drift_output"] == 4.0


def test_metrics_drift_null_when_no_upstream_usage():
    """AC3: drift aggregates are null when no request had upstream usage."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_no_usage_body("Hi")), registry, collector)
        _post_request(tc, "alpha", _request_body())
        resp = tc.get("/metrics")

    profile = resp.json()["profiles"]["alpha"]
    assert profile["mean_drift_input"] is None
    assert profile["abs_mean_drift_input"] is None
    assert profile["mean_drift_output"] is None
    assert profile["abs_mean_drift_output"] is None


def test_metrics_drift_aggregates_multiple_requests():
    """AC5 (integration): drift aggregates correctly averaged across multiple requests.

    Request 1: content='hello' → est_input=1, upstream_input=10 → drift_input=-9
    Request 2: content='hello' → est_input=1, upstream_input=2  → drift_input=-1
    mean_drift_input = (-9 + -1) / 2 = -5.0
    abs_mean_drift_input = (9 + 1) / 2 = 5.0
    """
    collector = MetricsCollector()
    registry = _registry("alpha")
    req_body = _request_body("hello")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5, "Hi")), registry, collector)
        _post_request(tc, "alpha", req_body)
        app.state.http_client = _http_client(_anthropic_body(2, 1, "Hi"))
        _post_request(tc, "alpha", req_body)
        resp = tc.get("/metrics")

    profile = resp.json()["profiles"]["alpha"]
    assert profile["mean_drift_input"] == -5.0    # (-9 + -1) / 2
    assert profile["abs_mean_drift_input"] == 5.0  # (9 + 1) / 2


# ---------------------------------------------------------------------------
# AC6: Existing metrics and request-record fields unchanged
# ---------------------------------------------------------------------------

EXISTING_METRICS_FIELDS = {
    "request_count",
    "error_count",
    "total_input_tokens",
    "total_output_tokens",
    "total_est_cost_usd",
    "p50_latency_ms",
    "p95_latency_ms",
}

EXISTING_RECORD_FIELDS = {
    "request_id",
    "timestamp",
    "profile_name",
    "profile_kind",
    "requested_model",
    "upstream_model",
    "upstream_host",
    "method",
    "path",
    "status",
    "latency_ms",
    "streamed",
    "run_id",
    "role",
    "ticket",
}


def test_existing_metrics_fields_preserved():
    """AC6: All existing /metrics profile fields are still present."""
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5)), registry, collector)
        _post_request(tc, "alpha", _request_body())
        resp = tc.get("/metrics")

    profile = resp.json()["profiles"]["alpha"]
    missing = EXISTING_METRICS_FIELDS - set(profile.keys())
    assert not missing, f"Existing metrics fields removed: {missing}"


def test_existing_record_fields_preserved():
    """AC6: All existing request-record fields are still present."""
    capture = StringIO()
    collector = MetricsCollector()
    registry = _registry("alpha")

    with TestClient(app) as tc:
        _setup(_http_client(_anthropic_body(10, 5)), registry, collector, capture)
        _post_request(tc, "alpha", _request_body())

    lines = [l for l in capture.getvalue().splitlines() if l.startswith("{")]
    assert lines, "No JSON records emitted"
    record = json.loads(lines[0])
    missing = EXISTING_RECORD_FIELDS - set(record.keys())
    assert not missing, f"Existing record fields removed: {missing}"


# ---------------------------------------------------------------------------
# MetricsCollector unit tests for drift storage and aggregation
# ---------------------------------------------------------------------------


def test_collector_record_stores_drift_and_aggregates():
    """AC5 (unit): MetricsCollector correctly stores and aggregates drift."""
    collector = MetricsCollector()
    collector.record(
        profile="alpha",
        status=200,
        latency_ms=10.0,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        token_drift_input=-5,
        token_drift_output=2,
    )
    snap = collector.snapshot()
    assert snap["alpha"]["mean_drift_input"] == -5.0
    assert snap["alpha"]["abs_mean_drift_input"] == 5.0
    assert snap["alpha"]["mean_drift_output"] == 2.0
    assert snap["alpha"]["abs_mean_drift_output"] == 2.0


def test_collector_drift_none_excluded_from_aggregation():
    """AC5 (unit): None drift values are excluded; only non-None values averaged."""
    collector = MetricsCollector()
    collector.record(
        profile="alpha", status=200, latency_ms=5.0,
        token_drift_input=-10, token_drift_output=3,
    )
    collector.record(
        profile="alpha", status=200, latency_ms=5.0,
        token_drift_input=None, token_drift_output=None,
    )
    snap = collector.snapshot()
    # Only the first sample has drift data
    assert snap["alpha"]["mean_drift_input"] == -10.0
    assert snap["alpha"]["abs_mean_drift_input"] == 10.0
    assert snap["alpha"]["mean_drift_output"] == 3.0
    assert snap["alpha"]["abs_mean_drift_output"] == 3.0


def test_collector_drift_aggregates_none_when_no_samples_have_drift():
    """AC5 (unit): drift aggregates are None when all samples have token_drift_input=None."""
    collector = MetricsCollector()
    collector.record(
        profile="alpha", status=200, latency_ms=5.0,
        token_drift_input=None, token_drift_output=None,
    )
    snap = collector.snapshot()
    assert snap["alpha"]["mean_drift_input"] is None
    assert snap["alpha"]["abs_mean_drift_input"] is None
    assert snap["alpha"]["mean_drift_output"] is None
    assert snap["alpha"]["abs_mean_drift_output"] is None


def test_collector_drift_multi_sample_mean():
    """AC5 (unit): mean and abs-mean are correct across multiple samples."""
    collector = MetricsCollector()
    # drift values: -10, -2, 6
    for d in (-10, -2, 6):
        collector.record(
            profile="alpha", status=200, latency_ms=1.0,
            token_drift_input=d, token_drift_output=-d,
        )
    snap = collector.snapshot()
    # mean = (-10 + -2 + 6) / 3 = -6/3 = -2.0
    assert abs(snap["alpha"]["mean_drift_input"] - (-2.0)) < 1e-9
    # abs_mean = (10 + 2 + 6) / 3 = 18/3 = 6.0
    assert abs(snap["alpha"]["abs_mean_drift_input"] - 6.0) < 1e-9
    # output drift is negated: 10, 2, -6 → mean = (10+2-6)/3 = 2.0, abs_mean = (10+2+6)/3 = 6.0
    assert abs(snap["alpha"]["mean_drift_output"] - 2.0) < 1e-9
    assert abs(snap["alpha"]["abs_mean_drift_output"] - 6.0) < 1e-9


def test_collector_existing_fields_unchanged():
    """AC6: MetricsCollector.snapshot() still returns all seven existing fields."""
    collector = MetricsCollector()
    collector.record(profile="alpha", status=200, latency_ms=5.0)
    snap = collector.snapshot()
    for field in EXISTING_METRICS_FIELDS:
        assert field in snap["alpha"], f"Existing field '{field}' missing from snapshot"
