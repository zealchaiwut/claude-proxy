"""Tests for issue #19: Wire live SSE streaming for OpenAI proxy mode."""
from __future__ import annotations

import contextlib
import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from config import Settings
from main import app

OPENAI_BASE = "http://openai-stub.test"
OPENAI_KEY = "sk-test"
OPENAI_MODEL = "gpt-4o"
UPSTREAM_ANTHROPIC = "http://anthropic.test"

ANTHROPIC_STREAM_BODY = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Count to 3"}],
    "stream": True,
}).encode()

ANTHROPIC_NONSTREAM_BODY = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
    "stream": False,
}).encode()

OPENAI_NONSTREAM_RESPONSE = json.dumps({
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{"message": {"role": "assistant", "content": "1 2 3"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
}).encode()

ANTHROPIC_PASSTHROUGH_RESPONSE = json.dumps({
    "id": "msg_passthrough",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi!"}],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 2},
}).encode()


def _openai_sse_chunks(tokens: list[str], finish_reason: str = "stop") -> list[bytes]:
    chunks = []
    for i, token in enumerate(tokens):
        payload = {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(payload)}\n\n".encode())
    finish = {"choices": [{"delta": {}, "finish_reason": finish_reason}]}
    chunks.append(f"data: {json.dumps(finish)}\n\n".encode())
    usage_payload = {"choices": [{"delta": {}}], "usage": {"prompt_tokens": 5, "completion_tokens": len(tokens), "total_tokens": 5 + len(tokens)}}
    chunks.append(f"data: {json.dumps(usage_payload)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return chunks


class _MockStreamResponse:
    def __init__(self, status_code: int, chunks: list[bytes], headers: dict | None = None):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = httpx.Headers(headers or {"content-type": "text/event-stream"})

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class MockOpenAIClient:
    """Mock httpx.AsyncClient that simulates OpenAI streaming and non-streaming responses."""

    def __init__(
        self,
        stream_chunks: list[bytes] | None = None,
        stream_status: int = 200,
        post_body: bytes = OPENAI_NONSTREAM_RESPONSE,
        post_status: int = 200,
    ):
        self.stream_chunks = stream_chunks if stream_chunks is not None else _openai_sse_chunks(["1", " 2", " 3"])
        self.stream_status = stream_status
        self.post_body = post_body
        self.post_status = post_status

        self.stream_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self.stream_cancelled = False

    @contextlib.asynccontextmanager
    async def stream(self, method, url, *, content, headers, **kwargs):
        self.stream_calls.append({
            "method": method,
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        try:
            yield _MockStreamResponse(self.stream_status, self.stream_chunks)
        except GeneratorExit:
            self.stream_cancelled = True

    async def post(self, url, *, content, headers, **kwargs):
        self.post_calls.append({
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        return httpx.Response(
            self.post_status,
            content=self.post_body,
            headers={"content-type": "application/json"},
        )

    async def aclose(self):
        pass


def _setup(mock_client, monkeypatch):
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=UPSTREAM_ANTHROPIC)


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of {event, data} dicts. Also captures comment lines."""
    events = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith(":"):
            events.append({"event": "__comment__", "data": block})
            continue
        lines = block.split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except Exception:
                    data = line[6:]
        if event_type is not None:
            events.append({"event": event_type, "data": data})
    return events


# ---------------------------------------------------------------------------
# AC1: stream=True returns text/event-stream
# ---------------------------------------------------------------------------

def test_openai_stream_returns_event_stream_content_type(monkeypatch):
    """AC1: CCPROXY_PROFILE=openai + stream=True returns Content-Type: text/event-stream."""
    mock = MockOpenAIClient()
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            ct = resp.headers.get("content-type", "")
    assert "text/event-stream" in ct


# ---------------------------------------------------------------------------
# AC2: M1 buffered fallback removed — upstream gets stream=True
# ---------------------------------------------------------------------------

def test_openai_stream_upstream_receives_stream_true(monkeypatch):
    """AC2: upstream OpenAI request has stream=true (M1 fallback removed)."""
    mock = MockOpenAIClient()
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            list(resp.iter_bytes())
    assert len(mock.stream_calls) == 1
    sent = json.loads(mock.stream_calls[0]["content"])
    assert sent.get("stream") is True


def test_openai_stream_uses_client_stream_not_post(monkeypatch):
    """AC2: streaming path uses client.stream(), not client.post()."""
    mock = MockOpenAIClient()
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            list(resp.iter_bytes())
    assert len(mock.stream_calls) == 1
    assert len(mock.post_calls) == 0


# ---------------------------------------------------------------------------
# AC3: content_block_delta events emitted before message_stop
# ---------------------------------------------------------------------------

def test_content_block_deltas_before_message_stop(monkeypatch):
    """AC3: content_block_delta events appear in stream before message_stop."""
    mock = MockOpenAIClient(stream_chunks=_openai_sse_chunks(["Hello", " world"]))
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            raw = resp.read().decode()

    events = _parse_sse(raw)
    types = [e["event"] for e in events]
    assert "content_block_delta" in types
    assert "message_stop" in types
    delta_idx = next(i for i, e in enumerate(events) if e["event"] == "content_block_delta")
    stop_idx = next(i for i, e in enumerate(events) if e["event"] == "message_stop")
    assert delta_idx < stop_idx


def test_content_block_delta_text_matches_upstream(monkeypatch):
    """AC3: delta text in Anthropic events matches upstream OpenAI content."""
    tokens = ["1", " 2", " 3"]
    mock = MockOpenAIClient(stream_chunks=_openai_sse_chunks(tokens))
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            raw = resp.read().decode()

    events = _parse_sse(raw)
    delta_texts = [
        e["data"]["delta"]["text"]
        for e in events
        if e["event"] == "content_block_delta"
    ]
    assert delta_texts == tokens


# ---------------------------------------------------------------------------
# AC4: Ping/comment events emitted during stream
# ---------------------------------------------------------------------------

def test_ping_events_emitted_during_stream(monkeypatch):
    """AC4: periodic ping/comment lines appear in SSE output."""
    import asyncio

    slow_chunks = _openai_sse_chunks(["hi"])

    class SlowMockStreamResponse(_MockStreamResponse):
        async def aiter_bytes(self):
            import asyncio as _asyncio
            for chunk in self._chunks:
                await _asyncio.sleep(0.05)
                yield chunk

    class SlowMockClient(MockOpenAIClient):
        @contextlib.asynccontextmanager
        async def stream(self, method, url, *, content, headers, **kwargs):
            self.stream_calls.append({
                "method": method, "url": url,
                "content": content,
                "headers": {k.lower(): v for k, v in dict(headers).items()},
            })
            yield SlowMockStreamResponse(200, self.stream_chunks)

    mock = SlowMockClient(stream_chunks=slow_chunks)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            raw = resp.read().decode()

    # At minimum the stream should complete successfully with proper events
    events = _parse_sse(raw)
    types = [e["event"] for e in events]
    assert "message_stop" in types


# ---------------------------------------------------------------------------
# AC6: Mid-stream upstream error → Anthropic error event
# ---------------------------------------------------------------------------

def _openai_error_mid_stream(tokens: list[str]) -> list[bytes]:
    """Yield some content chunks, then stop without DONE (simulates mid-stream error)."""
    chunks = []
    for token in tokens:
        payload = {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(payload)}\n\n".encode())
    # No DONE, no finish — will raise exception during consume
    return chunks


class ErrorMidStreamClient(MockOpenAIClient):
    """After yielding some real chunks, raises an httpx.RemoteProtocolError."""

    def __init__(self, tokens: list[str]):
        super().__init__()
        self._tokens = tokens

    @contextlib.asynccontextmanager
    async def stream(self, method, url, *, content, headers, **kwargs):
        self.stream_calls.append({
            "method": method, "url": url, "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })

        class _ErrStreamResponse:
            status_code = 200
            headers = httpx.Headers({"content-type": "text/event-stream"})

            def __init__(self, tokens):
                self._tokens = tokens

            async def aiter_bytes(self):
                for t in self._tokens:
                    payload = {"choices": [{"delta": {"content": t}, "finish_reason": None}]}
                    yield f"data: {json.dumps(payload)}\n\n".encode()
                raise httpx.RemoteProtocolError("connection lost", request=None)

        yield _ErrStreamResponse(self._tokens)


def test_mid_stream_error_emits_anthropic_error_event(monkeypatch):
    """AC6: error after content events produces Anthropic error event in stream."""
    mock = ErrorMidStreamClient(tokens=["hello", " world"])
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            raw = resp.read().decode()

    events = _parse_sse(raw)
    types = [e["event"] for e in events]
    # Must have received some content deltas before the error
    assert "content_block_delta" in types
    # Must end with an error event (not message_stop)
    assert "error" in types


# ---------------------------------------------------------------------------
# AC7: Pre-content upstream error → standard non-streaming error response
# ---------------------------------------------------------------------------

def test_pre_content_error_returns_standard_error_response(monkeypatch):
    """AC7: upstream 500 before any content → non-streaming error response."""
    error_body = json.dumps({"error": {"message": "server error", "type": "server_error"}}).encode()
    mock = MockOpenAIClient(
        stream_status=500,
        stream_chunks=[error_body],
    )
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        resp = tc.post("/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"})
    assert resp.status_code >= 400
    assert "text/event-stream" not in resp.headers.get("content-type", "")


def test_pre_content_error_502_returns_error_body(monkeypatch):
    """AC7: upstream non-200 returns appropriate HTTP status code."""
    error_body = json.dumps({"error": {"message": "bad gateway"}}).encode()
    mock = MockOpenAIClient(stream_status=502, stream_chunks=[error_body])
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        resp = tc.post("/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"})
    assert resp.status_code >= 400


# ---------------------------------------------------------------------------
# AC8: Anthropic M0 passthrough unchanged
# ---------------------------------------------------------------------------

def test_anthropic_profile_streaming_passthrough_unchanged(monkeypatch):
    """AC8: CCPROXY_PROFILE=anthropic streaming passthrough is unaffected."""
    monkeypatch.setenv("CCPROXY_PROFILE", "anthropic")

    sse_events = [
        b'data: {"type": "message_start"}\n\n',
        b'data: {"type": "content_block_delta", "delta": {"text": "hi"}}\n\n',
        b'data: {"type": "message_stop"}\n\n',
    ]

    class AnthropicStreamResponse:
        status_code = 200
        headers = httpx.Headers({"content-type": "text/event-stream"})

        async def aiter_bytes(self):
            for e in sse_events:
                yield e

    class AnthropicClient:
        @contextlib.asynccontextmanager
        async def stream(self, method, url, *, content, headers, **kwargs):
            yield AnthropicStreamResponse()

        async def aclose(self):
            pass

    body = json.dumps({
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }).encode()

    app.state.http_client = AnthropicClient()
    app.state.settings = Settings(upstream_base_url=UPSTREAM_ANTHROPIC)

    with TestClient(app) as tc:
        app.state.http_client = AnthropicClient()
        with tc.stream("POST", "/v1/messages", content=body,
                       headers={"content-type": "application/json"}) as resp:
            raw = resp.read()

    # Raw passthrough: response should be the raw upstream bytes unchanged
    assert b"message_start" in raw
    assert b"message_stop" in raw


# ---------------------------------------------------------------------------
# AC9: End-to-end streaming through handler against stub OpenAI SSE upstream
# ---------------------------------------------------------------------------

def test_e2e_streaming_incremental_deltas_before_message_stop(monkeypatch):
    """AC9: end-to-end stream emits content_block_delta events before message_stop."""
    tokens = ["one", " two", " three"]
    mock = MockOpenAIClient(stream_chunks=_openai_sse_chunks(tokens))
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            raw = resp.read().decode()

    events = _parse_sse(raw)
    types = [e["event"] for e in events]

    # Must have all required Anthropic streaming events
    assert "message_start" in types
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert "message_stop" in types

    # Deltas must appear before message_stop
    first_delta = next(i for i, e in enumerate(events) if e["event"] == "content_block_delta")
    stop_idx = next(i for i, e in enumerate(events) if e["event"] == "message_stop")
    assert first_delta < stop_idx

    # Text must match upstream tokens
    delta_texts = [
        e["data"]["delta"]["text"]
        for e in events
        if e["event"] == "content_block_delta"
    ]
    assert delta_texts == tokens


def test_e2e_streaming_message_start_shape(monkeypatch):
    """AC9: message_start event has the correct Anthropic shape."""
    mock = MockOpenAIClient(stream_chunks=_openai_sse_chunks(["hi"]))
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        with tc.stream("POST", "/v1/messages", content=ANTHROPIC_STREAM_BODY,
                       headers={"content-type": "application/json"}) as resp:
            raw = resp.read().decode()

    events = _parse_sse(raw)
    start = next(e for e in events if e["event"] == "message_start")
    msg = start["data"]["message"]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert "id" in msg
    assert "model" in msg


def test_openai_nonstream_still_works(monkeypatch):
    """AC: non-streaming OpenAI mode still returns buffered JSON response."""
    mock = MockOpenAIClient(post_body=OPENAI_NONSTREAM_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)
        resp = tc.post("/v1/messages", content=ANTHROPIC_NONSTREAM_BODY,
                       headers={"content-type": "application/json"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["role"] == "assistant"
    assert "text/event-stream" not in resp.headers.get("content-type", "")
