"""Tests for issue #54: real tokenizer in count_tokens and streaming fallback."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app
from config import Settings
from services.tokenizer import count_text_tokens, count_messages_tokens


# ---------------------------------------------------------------------------
# Unit: tokenizer service helpers
# ---------------------------------------------------------------------------


def test_count_text_tokens_known_string():
    """Tokenizer returns the correct cl100k_base count for a known string."""
    # tiktoken cl100k_base: "Hello world" -> 2 tokens
    assert count_text_tokens("Hello world") == 2


def test_count_text_tokens_empty_returns_zero():
    """Empty string returns 0."""
    assert count_text_tokens("") == 0


def test_count_text_tokens_matches_tiktoken_directly():
    """count_text_tokens output matches tiktoken cl100k_base directly."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    text = "Say something interesting about Python programming"
    assert count_text_tokens(text) == len(enc.encode(text))


def test_count_messages_tokens_string_content():
    """count_messages_tokens handles string content fields."""
    body = {"messages": [{"role": "user", "content": "Hello world"}]}
    # "Hello world" -> 2 tokens
    assert count_messages_tokens(body) == 2


def test_count_messages_tokens_list_content():
    """count_messages_tokens handles list content with text blocks."""
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello world"}],
            }
        ]
    }
    assert count_messages_tokens(body) == 2


def test_count_messages_tokens_empty_messages_returns_one():
    """count_messages_tokens returns 1 for empty messages (minimum)."""
    assert count_messages_tokens({"messages": []}) == 1


# ---------------------------------------------------------------------------
# AC1: count_tokens in OpenAI mode uses real tokenizer, not chars/4
# ---------------------------------------------------------------------------


def test_count_tokens_openai_mode_uses_real_tokenizer(monkeypatch):
    """AC1: count_tokens in OpenAI mode returns real tokenizer count, not chars/4."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.test")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    # Use a string where heuristic (chars/4=25) and tokenizer diverge
    prompt = "a" * 100
    body = json.dumps({"messages": [{"role": "user", "content": prompt}]}).encode()
    expected = len(enc.encode(prompt))
    heuristic = len(prompt) // 4  # 25

    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()

    with TestClient(app) as tc:
        app.state.http_client = mock_client
        app.state.settings = Settings(upstream_base_url="http://upstream.test")
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "input_tokens" in data
    result = data["input_tokens"]
    assert isinstance(result, int)
    assert result >= 1
    assert result == expected, (
        f"Expected tokenizer count {expected}, got {result}. "
        f"chars/4 heuristic would give {heuristic}."
    )
    assert result != heuristic, "Real tokenizer result must differ from chars/4 heuristic"


def test_count_tokens_openai_mode_no_upstream_call(monkeypatch):
    """AC1: count_tokens in OpenAI mode never calls upstream."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.test")

    called = {}

    async def _post(url, *, content, headers, **kwargs):
        called["hit"] = url
        raise AssertionError("Should not call upstream for OpenAI count_tokens")

    mock_client = MagicMock()
    mock_client.post = _post
    mock_client.aclose = AsyncMock()

    body = json.dumps({"messages": [{"role": "user", "content": "Hello"}]}).encode()

    with TestClient(app) as tc:
        app.state.http_client = mock_client
        app.state.settings = Settings(upstream_base_url="http://upstream.test")
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert "hit" not in called


# ---------------------------------------------------------------------------
# AC2: count_tokens in Anthropic mode is unmodified passthrough
# ---------------------------------------------------------------------------


def test_count_tokens_anthropic_mode_passthrough(monkeypatch):
    """AC2: count_tokens in Anthropic mode passes through to upstream unchanged."""
    monkeypatch.setenv("CCPROXY_PROFILE", "anthropic")
    upstream_response = json.dumps({"input_tokens": 42}).encode()
    captured = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["url"] = url
        import httpx as _httpx
        mock_resp = MagicMock(spec=_httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = upstream_response
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock_client = MagicMock()
    mock_client.post = _post
    mock_client.aclose = AsyncMock()

    body = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()

    with TestClient(app) as tc:
        app.state.http_client = mock_client
        app.state.settings = Settings(upstream_base_url="http://anthropic.test")
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.content == upstream_response
    assert "anthropic.test" in captured.get("url", "")


def test_count_tokens_anthropic_mode_returns_upstream_value(monkeypatch):
    """AC2: Anthropic-mode count_tokens returns exactly what the upstream returned."""
    monkeypatch.setenv("CCPROXY_PROFILE", "anthropic")
    arbitrary_count = 99
    upstream_response = json.dumps({"input_tokens": arbitrary_count}).encode()

    async def _post(url, *, content, headers, **kwargs):
        import httpx as _httpx
        mock_resp = MagicMock(spec=_httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = upstream_response
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock_client = MagicMock()
    mock_client.post = _post
    mock_client.aclose = AsyncMock()

    body = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()

    with TestClient(app) as tc:
        app.state.http_client = mock_client
        app.state.settings = Settings(upstream_base_url="http://anthropic.test")
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.json()["input_tokens"] == arbitrary_count


# ---------------------------------------------------------------------------
# AC3: Streaming fallback uses tokenizer when upstream omits usage
# ---------------------------------------------------------------------------


def _make_sse_stream_without_usage(output_text: str) -> list[bytes]:
    """SSE stream that does NOT include usage in message_delta."""
    msg_start = json.dumps({
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant",
                    "content": [], "usage": {"input_tokens": 10}},
    })
    cb_start = json.dumps({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}})
    cb_delta = json.dumps({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": output_text}})
    cb_stop = json.dumps({"type": "content_block_stop", "index": 0})
    msg_delta = json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        # No "usage" key here — forces fallback
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


def _make_sse_stream_with_usage(output_text: str, output_tokens: int) -> list[bytes]:
    """SSE stream that includes usage in message_delta."""
    msg_start = json.dumps({
        "type": "message_start",
        "message": {"id": "msg_1", "type": "message", "role": "assistant",
                    "content": [], "usage": {"input_tokens": 10}},
    })
    cb_start = json.dumps({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}})
    cb_delta = json.dumps({"type": "content_block_delta", "index": 0,
                           "delta": {"type": "text_delta", "text": output_text}})
    cb_stop = json.dumps({"type": "content_block_stop", "index": 0})
    msg_delta = json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": output_tokens},
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


def test_streaming_fallback_without_usage_uses_tokenizer():
    """AC3: streaming fallback computes output_tokens via tokenizer when upstream omits usage."""
    from services.cost_accounting import parse_anthropic_sse_usage
    import tiktoken

    output_text = "Hello there!"
    enc = tiktoken.get_encoding("cl100k_base")
    expected_tokens = len(enc.encode(output_text))  # 3

    sse_bytes = b"".join(_make_sse_stream_without_usage(output_text))
    body_json: dict = {"messages": [{"role": "user", "content": "Hi"}]}

    _, out_tok = parse_anthropic_sse_usage(sse_bytes, body_json)

    assert out_tok == expected_tokens, (
        f"Expected tokenizer count {expected_tokens} for {repr(output_text)}, got {out_tok}"
    )


def test_streaming_fallback_with_usage_uses_upstream_value():
    """AC3: when upstream provides output_tokens in usage, that value is used (no tokenizer)."""
    from services.cost_accounting import parse_anthropic_sse_usage

    output_text = "Hello there!"
    upstream_output_tokens = 999  # deliberately different from tokenizer count

    sse_bytes = b"".join(_make_sse_stream_with_usage(output_text, upstream_output_tokens))
    body_json: dict = {"messages": [{"role": "user", "content": "Hi"}]}

    _, out_tok = parse_anthropic_sse_usage(sse_bytes, body_json)

    assert out_tok == upstream_output_tokens, (
        f"Expected upstream value {upstream_output_tokens}, got {out_tok}"
    )
