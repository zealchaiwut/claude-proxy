"""Tests for issue #58: Surface cache effectiveness in logging and metrics.

AC coverage:
- ac1-anthropic-log: M5 records for Anthropic path include cache_read_input_tokens / cache_creation_input_tokens
- ac2-openai-log: M5 records for OpenAI path include cache_miss_estimate
- ac3-metrics-anthropic: /metrics includes per-profile cache_hit_ratio for Anthropic profiles
- ac4-metrics-openai: /metrics includes total_cache_miss_tokens and est_cache_gap_cost_usd for OpenAI profiles
- ac5-anthropic-extraction: unit tests for Anthropic cache token extraction with mocked payloads
- ac6-openai-estimation: unit tests for OpenAI cache_miss_estimate with mocked tokenizer output
"""
import io
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig
from services.cost_accounting import (
    extract_anthropic_cache_usage_from_response,
    extract_anthropic_cache_usage_from_sse,
)
from services.metrics_collector import MetricsCollector, _COST_PER_INPUT_TOKEN
from services.request_logger import RequestLogger

STUB_ANTHROPIC = "http://stub-anthropic.test"
STUB_OPENAI = "http://stub-openai.test"
CLIENT_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _anthropic_response(
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> bytes:
    usage: dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    if cache_creation:
        usage["cache_creation_input_tokens"] = cache_creation
    return json.dumps({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "model": CLIENT_MODEL,
        "stop_reason": "end_turn",
        "usage": usage,
    }).encode()


def _openai_response() -> bytes:
    return json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": OPENAI_MODEL,
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }).encode()


def _request_body(content: str = "hello", model: str = CLIENT_MODEL) -> bytes:
    return json.dumps({
        "model": model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": content}],
    }).encode()


def _passthrough_registry(name: str = "anthr") -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        name: ProfileConfig(kind="passthrough", upstream=STUB_ANTHROPIC),
    }))


def _openai_registry(name: str = "oai", api_key_env: str = "TEST_OAI_KEY") -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        name: ProfileConfig(kind="openai", upstream=STUB_OPENAI, api_key_env=api_key_env, model=OPENAI_MODEL),
    }))


def _sync_client(body: bytes, status: int = 200) -> object:
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
    registry,
    capture=None,
    collector=None,
) -> None:
    app.state.http_client = client
    app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
    app.state.profile_registry = registry
    app.state.config_from_file = True
    app.state.request_logger = RequestLogger(capture=capture) if capture is not None else None
    app.state.metrics_collector = collector


def _read_record(capture: io.StringIO) -> dict:
    capture.seek(0)
    lines = [ln.strip() for ln in capture.readlines() if ln.strip()]
    assert len(lines) >= 1, "No log records captured"
    return json.loads(lines[0])


# ---------------------------------------------------------------------------
# AC5: Unit tests — extract_anthropic_cache_usage_from_response
# ---------------------------------------------------------------------------


def test_extract_cache_from_response_both_fields():
    """AC5: extract both cache_read and cache_creation tokens from Anthropic response JSON."""
    response = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25,
        }
    }
    read, creation = extract_anthropic_cache_usage_from_response(response)
    assert read == 50
    assert creation == 25


def test_extract_cache_from_response_absent():
    """AC5: when cache fields absent, extraction returns (0, 0)."""
    response = {"usage": {"input_tokens": 100, "output_tokens": 20}}
    read, creation = extract_anthropic_cache_usage_from_response(response)
    assert read == 0
    assert creation == 0


def test_extract_cache_from_response_only_read():
    """AC5: only cache_read_input_tokens present → creation = 0."""
    response = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_input_tokens": 80,
        }
    }
    read, creation = extract_anthropic_cache_usage_from_response(response)
    assert read == 80
    assert creation == 0


def test_extract_cache_from_response_no_usage():
    """AC5: missing usage dict returns (0, 0) without error."""
    read, creation = extract_anthropic_cache_usage_from_response({})
    assert read == 0
    assert creation == 0


def test_extract_cache_from_sse_with_cache_read():
    """AC5: extract cache_read_input_tokens from Anthropic SSE message_start event."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 75,
        "cache_creation_input_tokens": 0,
    }
    evt = {"type": "message_start", "message": {"usage": usage}}
    data = f"data: {json.dumps(evt)}\n\n".encode()
    read, creation = extract_anthropic_cache_usage_from_sse(data)
    assert read == 75
    assert creation == 0


def test_extract_cache_from_sse_with_cache_creation():
    """AC5: extract cache_creation_input_tokens from Anthropic SSE message_start."""
    usage = {
        "input_tokens": 100,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 200,
    }
    evt = {"type": "message_start", "message": {"usage": usage}}
    data = f"data: {json.dumps(evt)}\n\n".encode()
    read, creation = extract_anthropic_cache_usage_from_sse(data)
    assert read == 0
    assert creation == 200


def test_extract_cache_from_sse_absent():
    """AC5: SSE with no cache fields in message_start returns (0, 0)."""
    usage = {"input_tokens": 100, "output_tokens": 0}
    evt = {"type": "message_start", "message": {"usage": usage}}
    data = f"data: {json.dumps(evt)}\n\n".encode()
    read, creation = extract_anthropic_cache_usage_from_sse(data)
    assert read == 0
    assert creation == 0


def test_extract_cache_from_sse_no_message_start():
    """AC5: SSE with no message_start event returns (0, 0)."""
    evt = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}
    data = f"data: {json.dumps(evt)}\n\n".encode()
    read, creation = extract_anthropic_cache_usage_from_sse(data)
    assert read == 0
    assert creation == 0


# ---------------------------------------------------------------------------
# AC6: Unit tests — cache_miss_estimate from M8 tokenizer
# ---------------------------------------------------------------------------


def test_cache_miss_estimate_computed_from_tokenizer():
    """AC6: count_messages_tokens returns a positive integer for any non-empty input."""
    from services.tokenizer import count_messages_tokens

    body = {"messages": [{"role": "user", "content": "hello world"}]}
    estimate = count_messages_tokens(body)
    assert isinstance(estimate, int)
    assert estimate >= 1


def test_cache_miss_estimate_proportional_to_input_length():
    """AC6: longer inputs produce larger cache_miss_estimate values."""
    from services.tokenizer import count_messages_tokens

    short_body = {"messages": [{"role": "user", "content": "hi"}]}
    long_body = {"messages": [{"role": "user", "content": "hi " * 100}]}

    short_est = count_messages_tokens(short_body)
    long_est = count_messages_tokens(long_body)
    assert long_est > short_est


# ---------------------------------------------------------------------------
# AC1: M5 log records — Anthropic passthrough path includes cache fields
# ---------------------------------------------------------------------------


def test_anthropic_log_record_includes_cache_read_tokens():
    """AC1: passthrough (Anthropic) log record includes cache_read_input_tokens."""
    capture = io.StringIO()
    registry = _passthrough_registry("anthr")
    client = _sync_client(_anthropic_response(cache_read=75))

    with TestClient(app) as tc:
        _setup(client, registry, capture=capture)
        resp = tc.post(
            "/v1/messages",
            content=_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthr"},
        )

    assert resp.status_code == 200
    record = _read_record(capture)
    assert "cache_read_input_tokens" in record
    assert record["cache_read_input_tokens"] == 75


def test_anthropic_log_record_includes_cache_creation_tokens():
    """AC1: passthrough log record includes cache_creation_input_tokens."""
    capture = io.StringIO()
    registry = _passthrough_registry("anthr")
    client = _sync_client(_anthropic_response(cache_creation=50))

    with TestClient(app) as tc:
        _setup(client, registry, capture=capture)
        resp = tc.post(
            "/v1/messages",
            content=_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthr"},
        )

    assert resp.status_code == 200
    record = _read_record(capture)
    assert "cache_creation_input_tokens" in record
    assert record["cache_creation_input_tokens"] == 50


def test_anthropic_log_record_cache_fields_zero_when_absent():
    """AC1: cache fields present as 0 when no cache activity occurred."""
    capture = io.StringIO()
    registry = _passthrough_registry("anthr")
    client = _sync_client(_anthropic_response(cache_read=0, cache_creation=0))

    with TestClient(app) as tc:
        _setup(client, registry, capture=capture)
        tc.post(
            "/v1/messages",
            content=_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthr"},
        )

    record = _read_record(capture)
    assert record.get("cache_read_input_tokens") == 0
    assert record.get("cache_creation_input_tokens") == 0


# ---------------------------------------------------------------------------
# AC2: M5 log records — OpenAI path includes cache_miss_estimate
# ---------------------------------------------------------------------------


def test_openai_log_record_includes_cache_miss_estimate(monkeypatch):
    """AC2: OpenAI log record includes cache_miss_estimate (non-negative integer)."""
    capture = io.StringIO()
    registry = _openai_registry("oai", "TEST_OAI_KEY")
    client = _sync_client(_openai_response())
    monkeypatch.setenv("TEST_OAI_KEY", "sk-test")

    with TestClient(app) as tc:
        _setup(client, registry, capture=capture)
        resp = tc.post(
            "/v1/messages",
            content=_request_body("hello world"),
            headers={"content-type": "application/json", "x-ccproxy-profile": "oai"},
        )

    assert resp.status_code == 200
    record = _read_record(capture)
    assert "cache_miss_estimate" in record
    assert isinstance(record["cache_miss_estimate"], int)
    assert record["cache_miss_estimate"] >= 1


def test_openai_log_record_cache_miss_estimate_nonnegative(monkeypatch):
    """AC2: cache_miss_estimate is always >= 0 regardless of input size."""
    capture = io.StringIO()
    registry = _openai_registry("oai", "TEST_OAI_KEY2")
    client = _sync_client(_openai_response())
    monkeypatch.setenv("TEST_OAI_KEY2", "sk-test2")

    with TestClient(app) as tc:
        _setup(client, registry, capture=capture)
        tc.post(
            "/v1/messages",
            content=_request_body("x"),
            headers={"content-type": "application/json", "x-ccproxy-profile": "oai"},
        )

    record = _read_record(capture)
    assert record["cache_miss_estimate"] >= 0


# ---------------------------------------------------------------------------
# AC3: /metrics — Anthropic profiles expose cache_hit_ratio
# ---------------------------------------------------------------------------


def test_metrics_anthropic_cache_hit_ratio_two_thirds():
    """AC3: cache_hit_ratio = (requests with cache_read > 0) / total."""
    collector = MetricsCollector()
    collector.record(
        profile="anthr", profile_kind="passthrough", status=200, latency_ms=1.0,
        cache_read_input_tokens=100,
    )
    collector.record(
        profile="anthr", profile_kind="passthrough", status=200, latency_ms=1.0,
        cache_read_input_tokens=50,
    )
    collector.record(
        profile="anthr", profile_kind="passthrough", status=200, latency_ms=1.0,
        cache_read_input_tokens=0,
    )

    snap = collector.snapshot()
    assert "cache_hit_ratio" in snap["anthr"]
    assert abs(snap["anthr"]["cache_hit_ratio"] - 2 / 3) < 1e-6


def test_metrics_anthropic_cache_hit_ratio_all_hits():
    """AC3: cache_hit_ratio = 1.0 when every request had a cache read."""
    collector = MetricsCollector()
    for _ in range(3):
        collector.record(
            profile="anthr", profile_kind="passthrough", status=200, latency_ms=1.0,
            cache_read_input_tokens=80,
        )

    snap = collector.snapshot()
    assert snap["anthr"]["cache_hit_ratio"] == 1.0


def test_metrics_anthropic_cache_hit_ratio_zero():
    """AC3: cache_hit_ratio = 0.0 when no request had a cache read."""
    collector = MetricsCollector()
    for _ in range(2):
        collector.record(
            profile="anthr", profile_kind="passthrough", status=200, latency_ms=1.0,
            cache_read_input_tokens=0,
        )

    snap = collector.snapshot()
    assert snap["anthr"]["cache_hit_ratio"] == 0.0


# ---------------------------------------------------------------------------
# AC4: /metrics — OpenAI profiles expose total_cache_miss_tokens + est_cache_gap_cost_usd
# ---------------------------------------------------------------------------


def test_metrics_openai_total_cache_miss_tokens_sum():
    """AC4: total_cache_miss_tokens = sum of all cache_miss_estimate values."""
    collector = MetricsCollector()
    collector.record(
        profile="oai", profile_kind="openai", status=200, latency_ms=1.0,
        cache_miss_estimate=500,
    )
    collector.record(
        profile="oai", profile_kind="openai", status=200, latency_ms=1.0,
        cache_miss_estimate=300,
    )

    snap = collector.snapshot()
    assert "total_cache_miss_tokens" in snap["oai"]
    assert snap["oai"]["total_cache_miss_tokens"] == 800


def test_metrics_openai_est_cache_gap_cost_positive():
    """AC4: est_cache_gap_cost_usd > 0 when cache_miss tokens are present."""
    collector = MetricsCollector()
    collector.record(
        profile="oai", profile_kind="openai", status=200, latency_ms=1.0,
        cache_miss_estimate=1_000_000,
    )

    snap = collector.snapshot()
    assert "est_cache_gap_cost_usd" in snap["oai"]
    expected = 1_000_000 * _COST_PER_INPUT_TOKEN
    assert abs(snap["oai"]["est_cache_gap_cost_usd"] - expected) < 1e-9


def test_metrics_openai_cache_gap_zero_when_no_misses():
    """AC4: total_cache_miss_tokens = 0 when no cache_miss_estimate recorded."""
    collector = MetricsCollector()
    collector.record(
        profile="oai", profile_kind="openai", status=200, latency_ms=1.0,
    )

    snap = collector.snapshot()
    assert snap["oai"].get("total_cache_miss_tokens", 0) == 0
