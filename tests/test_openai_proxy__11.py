"""Tests for issue #11: OpenAI proxy mode behind CCPROXY_PROFILE env switch."""
from __future__ import annotations

import contextlib
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient

from config import Settings
from main import app

UPSTREAM_ANTHROPIC = "http://anthropic.test"
OPENAI_BASE = "http://openai.test"
OPENAI_KEY = "sk-test-key"
OPENAI_MODEL = "gpt-4o"

ANTHROPIC_REQUEST_BODY = json.dumps(
    {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello"}],
    }
).encode()

ANTHROPIC_REQUEST_WITH_STREAM = json.dumps(
    {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": True,
    }
).encode()

OPENAI_RESPONSE = json.dumps(
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
).encode()

ANTHROPIC_PASSTHROUGH_RESPONSE = json.dumps(
    {
        "id": "msg_passthrough01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi from Anthropic!"}],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
).encode()


def _make_openai_mock_client(
    status: int = 200,
    response_body: bytes = OPENAI_RESPONSE,
) -> tuple[object, dict]:
    captured: dict = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = {k.lower(): v for k, v in dict(headers).items()}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status
        mock_resp.content = response_body
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    def _make_sse_chunks_from_content(text: str) -> list[bytes]:
        chunks = [
            f'data: {json.dumps({"choices":[{"delta":{"content":text},"finish_reason":None}]})}\n\n'.encode(),
            f'data: {json.dumps({"choices":[{"delta":{},"finish_reason":"stop"}]})}\n\n'.encode(),
            f'data: {json.dumps({"choices":[{"delta":{}}],"usage":{"prompt_tokens":10,"completion_tokens":5}})}\n\n'.encode(),
            b"data: [DONE]\n\n",
        ]
        return chunks

    class _StreamResp:
        status_code = status
        headers = httpx.Headers({"content-type": "text/event-stream"})

        async def aiter_bytes(self):
            # Extract text from the non-stream OPENAI_RESPONSE for SSE simulation
            try:
                parsed = json.loads(response_body)
                text = parsed["choices"][0]["message"]["content"] or ""
            except Exception:
                text = ""
            for chunk in _make_sse_chunks_from_content(text):
                yield chunk

    @contextlib.asynccontextmanager
    async def _stream(method, url, *, content, headers, **kwargs):
        captured["stream_url"] = url
        captured["stream_content"] = content
        captured["stream_headers"] = {k.lower(): v for k, v in dict(headers).items()}
        yield _StreamResp()

    client = MagicMock()
    client.post = _post
    client.stream = _stream
    client.aclose = AsyncMock()
    return client, captured


def _setup_openai(mock_client) -> None:
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=UPSTREAM_ANTHROPIC)


# ---------------------------------------------------------------------------
# AC (a): CCPROXY_PROFILE routing unit tests
# ---------------------------------------------------------------------------

def test_anthropic_profile_passthrough_byte_for_byte(monkeypatch):
    """AC(a): when CCPROXY_PROFILE=anthropic, response is byte-for-byte passthrough."""
    monkeypatch.setenv("CCPROXY_PROFILE", "anthropic")

    mock_client, _ = _make_openai_mock_client(response_body=ANTHROPIC_PASSTHROUGH_RESPONSE)
    # Override post to return anthropic response
    captured = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["url"] = url
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = ANTHROPIC_PASSTHROUGH_RESPONSE
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock_client.post = _post
    _setup_openai(mock_client)

    with TestClient(app) as tc:
        _setup_openai(mock_client)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.content == ANTHROPIC_PASSTHROUGH_RESPONSE
    assert captured["url"] == f"{UPSTREAM_ANTHROPIC}/v1/messages"


def test_openai_profile_routes_to_openai_upstream(monkeypatch):
    """AC(a): when CCPROXY_PROFILE=openai, request goes to ${OPENAI_BASE_URL}/chat/completions."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    mock_client, captured = _make_openai_mock_client()
    with TestClient(app) as tc:
        _setup_openai(mock_client)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert captured["url"] == f"{OPENAI_BASE}/chat/completions"
    assert resp.status_code == 200


def test_missing_profile_defaults_to_anthropic_passthrough(monkeypatch):
    """AC(a): absent CCPROXY_PROFILE defaults to anthropic passthrough."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)

    captured = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["url"] = url
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = ANTHROPIC_PASSTHROUGH_RESPONSE
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock_client = MagicMock()
    mock_client.post = _post
    mock_client.aclose = AsyncMock()

    with TestClient(app) as tc:
        _setup_openai(mock_client)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert captured["url"] == f"{UPSTREAM_ANTHROPIC}/v1/messages"
    assert resp.content == ANTHROPIC_PASSTHROUGH_RESPONSE


# ---------------------------------------------------------------------------
# AC (b): full openai round-trip integration test against stub upstream
# ---------------------------------------------------------------------------

def test_openai_roundtrip_returns_valid_anthropic_response(monkeypatch):
    """AC(b): full round-trip: Anthropic request → OpenAI → Anthropic MessagesResponse."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    mock_client, captured = _make_openai_mock_client(response_body=OPENAI_RESPONSE)
    with TestClient(app) as tc:
        _setup_openai(mock_client)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    body = resp.json()

    # Must be a valid Anthropic MessagesResponse shape
    assert body["role"] == "assistant"
    assert isinstance(body["content"], list)
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Hello!"
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["output_tokens"] == 5


def test_openai_request_carries_bearer_auth(monkeypatch):
    """AC(b): upstream request has Authorization: Bearer ${OPENAI_API_KEY}."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    mock_client, captured = _make_openai_mock_client()
    with TestClient(app) as tc:
        _setup_openai(mock_client)
        tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert captured["headers"]["authorization"] == f"Bearer {OPENAI_KEY}"


def test_openai_request_uses_configured_model(monkeypatch):
    """AC(b): translated request uses OPENAI_MODEL, not the client's model."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    mock_client, captured = _make_openai_mock_client()
    with TestClient(app) as tc:
        _setup_openai(mock_client)
        tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    sent_body = json.loads(captured["content"])
    assert sent_body["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# AC (c) [updated for M2/issue#19]: stream=true in openai mode returns live SSE
# The M1 "force non-stream" fallback is removed; stream=true now returns text/event-stream.
# ---------------------------------------------------------------------------

def test_stream_true_openai_mode_returns_sse_response(monkeypatch):
    """AC(c) M2: stream=true in openai mode returns text/event-stream (M1 fallback removed)."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    mock_client, captured = _make_openai_mock_client(response_body=OPENAI_RESPONSE)
    with TestClient(app) as tc:
        _setup_openai(mock_client)
        with tc.stream(
            "POST", "/v1/messages",
            content=ANTHROPIC_REQUEST_WITH_STREAM,
            headers={"content-type": "application/json"},
        ) as resp:
            assert "text/event-stream" in resp.headers.get("content-type", "")
            resp.read()

    assert resp.status_code == 200


def test_stream_true_openai_mode_returns_event_stream_content_type(monkeypatch):
    """AC(c) M2: stream=true in openai mode returns Content-Type: text/event-stream."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    mock_client, _ = _make_openai_mock_client(response_body=OPENAI_RESPONSE)
    with TestClient(app) as tc:
        _setup_openai(mock_client)
        with tc.stream(
            "POST", "/v1/messages",
            content=ANTHROPIC_REQUEST_WITH_STREAM,
            headers={"content-type": "application/json"},
        ) as resp:
            ct = resp.headers.get("content-type", "")
            resp.read()

    assert "text/event-stream" in ct


def test_stream_true_openai_upstream_receives_streaming_request(monkeypatch):
    """AC(c) M2: the upstream OpenAI call has stream=true (M1 non-stream fallback removed)."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    mock_client, captured = _make_openai_mock_client(response_body=OPENAI_RESPONSE)
    with TestClient(app) as tc:
        _setup_openai(mock_client)
        with tc.stream(
            "POST", "/v1/messages",
            content=ANTHROPIC_REQUEST_WITH_STREAM,
            headers={"content-type": "application/json"},
        ) as resp:
            resp.read()

    sent_body = json.loads(captured["stream_content"])
    assert sent_body.get("stream") is True


# ---------------------------------------------------------------------------
# AC (d)+(e): count_tokens endpoint
# ---------------------------------------------------------------------------

def test_count_tokens_openai_mode_local_heuristic(monkeypatch):
    """AC(d): count_tokens in openai mode returns local {input_tokens: N} without upstream call."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    called = {}

    async def _post(url, *, content, headers, **kwargs):
        called["hit"] = True
        raise AssertionError("Should not call upstream in openai count_tokens mode")

    mock_client = MagicMock()
    mock_client.post = _post
    mock_client.aclose = AsyncMock()

    body = json.dumps(
        {"messages": [{"role": "user", "content": "Hello world"}]}
    ).encode()

    with TestClient(app) as tc:
        _setup_openai(mock_client)
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "input_tokens" in data
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0
    assert "hit" not in called


def test_count_tokens_openai_mode_returns_positive_integer(monkeypatch):
    """AC(d): heuristic produces a positive integer for a non-empty messages payload."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)

    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()

    body = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "Say something interesting"},
                {"role": "assistant", "content": "Sure!"},
            ]
        }
    ).encode()

    with TestClient(app) as tc:
        _setup_openai(mock_client)
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.json()["input_tokens"] >= 1


def test_count_tokens_anthropic_mode_passthrough(monkeypatch):
    """AC(e): count_tokens in anthropic mode passes through to upstream unchanged."""
    monkeypatch.setenv("CCPROXY_PROFILE", "anthropic")

    count_tokens_response = json.dumps({"input_tokens": 42}).encode()
    captured = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["url"] = url
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = count_tokens_response
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock_client = MagicMock()
    mock_client.post = _post
    mock_client.aclose = AsyncMock()

    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    with TestClient(app) as tc:
        _setup_openai(mock_client)
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.content == count_tokens_response
    assert captured["url"] == f"{UPSTREAM_ANTHROPIC}/v1/messages/count_tokens"


# ---------------------------------------------------------------------------
# AC (f): credentials never logged
# ---------------------------------------------------------------------------

def test_openai_credentials_not_logged(monkeypatch, caplog):
    """AC(f): OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL never appear in log output."""
    secret_key = "sk-super-secret-key-12345"
    secret_base = "http://secret-openai.internal"
    secret_model = "gpt-secret-model"

    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", secret_base)
    monkeypatch.setenv("OPENAI_API_KEY", secret_key)
    monkeypatch.setenv("OPENAI_MODEL", secret_model)

    mock_client, _ = _make_openai_mock_client(response_body=OPENAI_RESPONSE)

    with caplog.at_level(logging.DEBUG):
        with TestClient(app) as tc:
            _setup_openai(mock_client)
            tc.post(
                "/v1/messages",
                content=ANTHROPIC_REQUEST_BODY,
                headers={"content-type": "application/json"},
            )

    log_text = caplog.text
    assert secret_key not in log_text
    assert secret_base not in log_text
    assert secret_model not in log_text
