"""Tests for issue #41: token and cost accounting in request log records."""
from __future__ import annotations

import contextlib
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from config import Settings
from main import app
from services.cost_accounting import (
    PricingConfig,
    compute_est_cost,
    count_input_tokens,
    extract_usage_from_response,
)


# ---------------------------------------------------------------------------
# AC7(a): correct est_cost_usd for a known token count and price
# ---------------------------------------------------------------------------


def test_compute_est_cost_correct_math():
    """AC7(a): formula is (input/1M)*input_rate + (output/1M)*output_rate."""
    pricing = PricingConfig(input_per_mtok=3.0, output_per_mtok=15.0)
    cost = compute_est_cost(1_000_000, 1_000_000, pricing)
    assert cost == pytest.approx(3.0 + 15.0)


def test_compute_est_cost_small_values():
    """AC7(a): correct math for small token counts (UAT step 1 scenario)."""
    pricing = PricingConfig(input_per_mtok=3.0, output_per_mtok=15.0)
    cost = compute_est_cost(100, 50, pricing)
    expected = (100 / 1_000_000) * 3.0 + (50 / 1_000_000) * 15.0
    assert cost == pytest.approx(expected)


def test_compute_est_cost_uat_scenario():
    """AC7(a): reproduce UAT step 1 — cost matches formula with known counts."""
    pricing = PricingConfig(input_per_mtok=3.00, output_per_mtok=15.00)
    input_t, output_t = 500, 200
    cost = compute_est_cost(input_t, output_t, pricing)
    assert cost == pytest.approx((input_t / 1_000_000) * 3.00 + (output_t / 1_000_000) * 15.00)


# ---------------------------------------------------------------------------
# AC7(b): est_cost_usd is null when no pricing is configured
# ---------------------------------------------------------------------------


def test_compute_est_cost_no_pricing_returns_none():
    """AC7(b): est_cost_usd is None (null) when no pricing block is configured."""
    assert compute_est_cost(100, 50, None) is None


def test_compute_est_cost_zero_tokens_no_pricing():
    """AC7(b): None regardless of token counts."""
    assert compute_est_cost(0, 0, None) is None


# ---------------------------------------------------------------------------
# AC1/AC2: usage extraction helpers
# ---------------------------------------------------------------------------


def test_extract_usage_from_response_present():
    """AC1: extracts (input_tokens, output_tokens) when usage field is present."""
    resp = {"usage": {"input_tokens": 42, "output_tokens": 17}}
    assert extract_usage_from_response(resp) == (42, 17)


def test_extract_usage_from_response_absent():
    """AC2: returns None when upstream omits the usage field."""
    assert extract_usage_from_response({}) is None


def test_extract_usage_from_response_partial_usage():
    """AC2: returns None when usage is present but missing sub-fields."""
    assert extract_usage_from_response({"usage": {"input_tokens": 5}}) is None
    assert extract_usage_from_response({"usage": {"output_tokens": 5}}) is None
    assert extract_usage_from_response({"usage": {}}) is None


def test_count_input_tokens_heuristic():
    """AC2: fallback count_input_tokens returns a positive int from message content."""
    body = {"messages": [{"role": "user", "content": "Hello world this is a test"}]}
    count = count_input_tokens(body)
    assert isinstance(count, int)
    assert count >= 1


# ---------------------------------------------------------------------------
# Integration: non-streaming request with pricing logs est_cost_usd
# ---------------------------------------------------------------------------


def _make_mock_client(response_body: bytes, status: int = 200):
    captured: dict = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["url"] = url
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.content = response_body
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()
    return client, captured


def _make_proxy_config_with_pricing():
    from profiles import ProfileConfig, ProxyConfig, ServerConfig
    from services.cost_accounting import PricingConfig as PC

    return ProxyConfig(
        server=ServerConfig(),
        profiles={
            "priced": ProfileConfig(
                kind="passthrough",
                upstream="http://upstream.test",
                pricing=PC(input_per_mtok=3.0, output_per_mtok=15.0),
            )
        },
    )


def _make_proxy_config_no_pricing():
    from profiles import ProfileConfig, ProxyConfig, ServerConfig

    return ProxyConfig(
        server=ServerConfig(),
        profiles={
            "unpriced": ProfileConfig(
                kind="passthrough",
                upstream="http://upstream.test",
            )
        },
    )


def _setup_app(mock_client, proxy_config=None):
    from profiles import ProfileRegistry

    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url="http://upstream.test")
    if proxy_config is not None:
        app.state.proxy_config = proxy_config
        app.state.config_from_file = True
        app.state.profile_registry = ProfileRegistry(proxy_config)
    else:
        app.state.config_from_file = False


_UPSTREAM_RESPONSE_WITH_USAGE = json.dumps(
    {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
).encode()

_REQUEST_BODY = json.dumps(
    {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello"}],
    }
).encode()


def test_non_streaming_with_pricing_logs_record(caplog):
    """AC3/AC4: non-streaming request logs input_tokens, output_tokens, and est_cost_usd."""
    mock_client, _ = _make_mock_client(_UPSTREAM_RESPONSE_WITH_USAGE)
    proxy_config = _make_proxy_config_with_pricing()

    with caplog.at_level(logging.INFO, logger="routers.messages"):
        with TestClient(app) as tc:
            _setup_app(mock_client, proxy_config)
            resp = tc.post(
                "/v1/messages",
                content=_REQUEST_BODY,
                headers={
                    "content-type": "application/json",
                    "x-ccproxy-profile": "priced",
                },
            )

    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "routers.messages"]
    assert records, "Expected a log record from routers.messages"
    rec = records[-1]
    assert rec.input_tokens == 10
    assert rec.output_tokens == 5
    expected_cost = (10 / 1_000_000) * 3.0 + (5 / 1_000_000) * 15.0
    assert rec.est_cost_usd == pytest.approx(expected_cost)


def test_non_streaming_no_pricing_logs_null_cost(caplog):
    """AC5: est_cost_usd is None when no pricing block is configured."""
    mock_client, _ = _make_mock_client(_UPSTREAM_RESPONSE_WITH_USAGE)
    proxy_config = _make_proxy_config_no_pricing()

    with caplog.at_level(logging.INFO, logger="routers.messages"):
        with TestClient(app) as tc:
            _setup_app(mock_client, proxy_config)
            resp = tc.post(
                "/v1/messages",
                content=_REQUEST_BODY,
                headers={
                    "content-type": "application/json",
                    "x-ccproxy-profile": "unpriced",
                },
            )

    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "routers.messages"]
    assert records
    rec = records[-1]
    assert rec.est_cost_usd is None


def test_non_streaming_fallback_when_no_upstream_usage(caplog):
    """AC2/AC4(UAT step 4): when upstream omits usage, fallback counts are used (non-zero)."""
    response_no_usage = json.dumps(
        {
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-3-haiku-20240307",
            "stop_reason": "end_turn",
        }
    ).encode()

    mock_client, _ = _make_mock_client(response_no_usage)
    proxy_config = _make_proxy_config_with_pricing()

    with caplog.at_level(logging.INFO, logger="routers.messages"):
        with TestClient(app) as tc:
            _setup_app(mock_client, proxy_config)
            resp = tc.post(
                "/v1/messages",
                content=_REQUEST_BODY,
                headers={
                    "content-type": "application/json",
                    "x-ccproxy-profile": "priced",
                },
            )

    assert resp.status_code == 200

    records = [r for r in caplog.records if r.name == "routers.messages"]
    assert records
    rec = records[-1]
    assert rec.input_tokens >= 1, "input_tokens must be non-zero (fallback)"
    assert rec.output_tokens >= 1, "output_tokens must be non-zero (fallback)"


# ---------------------------------------------------------------------------
# AC6/AC7(c): streaming path populates output_tokens
# ---------------------------------------------------------------------------


def _make_anthropic_sse_stream():
    """Build a realistic Anthropic SSE response as a list of byte chunks."""
    msg_start = json.dumps({
        "type": "message_start",
        "message": {
            "id": "msg_s1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-haiku-20240307",
            "content": [],
            "usage": {"input_tokens": 15},
        },
    })
    cb_start = json.dumps({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    cb_delta = json.dumps({
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello there!"},
    })
    cb_stop = json.dumps({"type": "content_block_stop", "index": 0})
    msg_delta = json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 3},
    })
    msg_stop = json.dumps({"type": "message_stop"})
    frames = [
        f"event: message_start\ndata: {msg_start}\n\n",
        f"event: content_block_start\ndata: {cb_start}\n\n",
        f"event: content_block_delta\ndata: {cb_delta}\n\n",
        f"event: content_block_stop\ndata: {cb_stop}\n\n",
        f"event: message_delta\ndata: {msg_delta}\n\n",
        f"event: message_stop\ndata: {msg_stop}\n\n",
    ]
    return [f.encode() for f in frames]


def test_streaming_passthrough_logs_non_zero_output_tokens(caplog):
    """AC6/AC7(c): streaming passthrough populates output_tokens in the log record."""
    sse_chunks = _make_anthropic_sse_stream()

    class _StreamResp:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            for chunk in sse_chunks:
                yield chunk

    @contextlib.asynccontextmanager
    async def _stream_ctx(method, url, *, content, headers, **kwargs):
        yield _StreamResp()

    mock_client = MagicMock()
    mock_client.stream = _stream_ctx
    mock_client.aclose = AsyncMock()

    proxy_config = _make_proxy_config_no_pricing()

    stream_body = json.dumps(
        {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
    ).encode()

    with caplog.at_level(logging.INFO, logger="routers.messages"):
        with TestClient(app) as tc:
            _setup_app(mock_client, proxy_config)
            with tc.stream(
                "POST",
                "/v1/messages",
                content=stream_body,
                headers={
                    "content-type": "application/json",
                    "x-ccproxy-profile": "unpriced",
                },
            ) as resp:
                resp.read()

    records = [r for r in caplog.records if r.name == "routers.messages"]
    assert records, "Expected a log record from routers.messages after streaming"
    rec = records[-1]
    assert rec.output_tokens > 0, f"output_tokens must be non-zero, got {rec.output_tokens}"


def test_streaming_with_pricing_logs_est_cost(caplog):
    """AC3/AC6: streaming request with pricing logs a non-None est_cost_usd."""
    sse_chunks = _make_anthropic_sse_stream()

    class _StreamResp:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            for chunk in sse_chunks:
                yield chunk

    @contextlib.asynccontextmanager
    async def _stream_ctx(method, url, *, content, headers, **kwargs):
        yield _StreamResp()

    mock_client = MagicMock()
    mock_client.stream = _stream_ctx
    mock_client.aclose = AsyncMock()

    proxy_config = _make_proxy_config_with_pricing()

    stream_body = json.dumps(
        {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
    ).encode()

    with caplog.at_level(logging.INFO, logger="routers.messages"):
        with TestClient(app) as tc:
            _setup_app(mock_client, proxy_config)
            with tc.stream(
                "POST",
                "/v1/messages",
                content=stream_body,
                headers={
                    "content-type": "application/json",
                    "x-ccproxy-profile": "priced",
                },
            ) as resp:
                resp.read()

    records = [r for r in caplog.records if r.name == "routers.messages"]
    assert records
    rec = records[-1]
    assert rec.output_tokens > 0
    assert rec.est_cost_usd is not None
    assert rec.est_cost_usd > 0
