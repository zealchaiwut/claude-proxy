"""Tests for issue #3: streaming passthrough for POST /v1/messages."""
import asyncio
import contextlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from config import Settings
from main import app

# ---------------------------------------------------------------------------
# Stub SSE events used across tests
# ---------------------------------------------------------------------------

STUB_EVENTS = [
    b"data: {\"type\": \"message_start\"}\n\n",
    b"data: {\"type\": \"content_block_delta\", \"delta\": {\"text\": \"Hi\"}}\n\n",
    b"data: [DONE]\n\n",
]

STREAMING_BODY = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
}).encode()

NON_STREAMING_BODY = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hello"}],
}).encode()

SAMPLE_BUFFERED_RESP = json.dumps({
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi!"}],
}).encode()

# ---------------------------------------------------------------------------
# Mock httpx client
# ---------------------------------------------------------------------------


class _MockStreamResponse:
    """Simulates an httpx.Response in streaming mode."""

    def __init__(self, status_code: int, events: list[bytes], headers: dict | None = None):
        self.status_code = status_code
        self._events = events
        self.headers = httpx.Headers(headers or {"content-type": "text/event-stream"})

    async def aiter_bytes(self):
        for event in self._events:
            yield event


class MockSSEClient:
    """
    Mock for httpx.AsyncClient that supports both stream() and post().

    stream() yields STUB_EVENTS one chunk at a time via an async context manager.
    post()   returns a buffered JSON response.
    """

    def __init__(
        self,
        events: list[bytes] | None = None,
        stream_status: int = 200,
        stream_headers: dict | None = None,
        buffered_body: bytes = SAMPLE_BUFFERED_RESP,
        buffered_status: int = 200,
    ):
        self.events = events if events is not None else list(STUB_EVENTS)
        self.stream_status = stream_status
        self.stream_headers = stream_headers or {"content-type": "text/event-stream"}
        self.buffered_body = buffered_body
        self.buffered_status = buffered_status

        self.stream_calls: list[dict] = []
        self.post_calls: list[dict] = []
        self.aclose_called = False

    @contextlib.asynccontextmanager
    async def stream(self, method, url, *, content, headers, **kwargs):
        self.stream_calls.append({
            "method": method,
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        yield _MockStreamResponse(self.stream_status, self.events, self.stream_headers)

    async def post(self, url, *, content, headers, **kwargs):
        self.post_calls.append({
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        mock_resp = httpx.Response(
            self.buffered_status,
            content=self.buffered_body,
            headers={"content-type": "application/json"},
        )
        return mock_resp

    async def aclose(self):
        self.aclose_called = True


def _setup(mock_client: MockSSEClient, upstream: str = "http://stub") -> None:
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=upstream)


# ---------------------------------------------------------------------------
# AC1: stream:true in body → text/event-stream response
# ---------------------------------------------------------------------------


def test_stream_true_returns_event_stream_content_type():
    """AC1: POST /v1/messages with stream:true returns Content-Type: text/event-stream."""
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            content_type = resp.headers.get("content-type", "")
    assert "text/event-stream" in content_type


def test_stream_true_uses_streaming_client_path():
    """AC1: stream:true routes through client.stream(), not client.post()."""
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            list(resp.iter_bytes())
    assert len(mock.stream_calls) == 1
    assert len(mock.post_calls) == 0


def test_stream_true_preserves_upstream_status_code():
    """AC1: upstream status code is passed through for streaming responses."""
    mock = MockSSEClient(stream_status=200)
    with TestClient(app) as tc:
        _setup(mock)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            status = resp.status_code
            list(resp.iter_bytes())
    assert status == 200


# ---------------------------------------------------------------------------
# AC2: Accept: text/event-stream header → streaming
# ---------------------------------------------------------------------------


def test_accept_event_stream_header_triggers_streaming():
    """AC2: Accept: text/event-stream header routes to streaming path."""
    body = json.dumps({
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
    }).encode()
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        with tc.stream(
            "POST", "/v1/messages",
            content=body,
            headers={
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
        ) as resp:
            content_type = resp.headers.get("content-type", "")
            list(resp.iter_bytes())
    assert "text/event-stream" in content_type
    assert len(mock.stream_calls) == 1


# ---------------------------------------------------------------------------
# AC3 / AC6: Incremental delivery — generator yields one event at a time
# ---------------------------------------------------------------------------


def _make_http_scope(body: bytes) -> tuple[dict, object]:
    """Return (scope, receive) for a minimal POST /v1/messages request."""
    _sent = []

    async def receive():
        if not _sent:
            _sent.append(True)
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.sleep(3600)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/messages",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "root_path": "",
        "app": app,
    }
    return scope, receive


@pytest.mark.asyncio
async def test_events_arrive_incrementally_not_buffered():
    """AC3/AC6: each upstream event is a separate yield — no full-body buffering."""
    from fastapi.responses import StreamingResponse
    from routers.messages import messages_passthrough

    mock = MockSSEClient()
    app.state.http_client = mock
    app.state.settings = Settings(upstream_base_url="http://stub")

    scope, receive = _make_http_scope(STREAMING_BODY)
    request = Request(scope, receive)

    response = await messages_passthrough(request)

    assert isinstance(response, StreamingResponse), (
        "Streaming request must return StreamingResponse, not a buffered Response"
    )

    # Iterate the body generator directly — each upstream event must be its own chunk.
    chunks = [chunk async for chunk in response.body_iterator]
    assert len(chunks) == len(STUB_EVENTS), (
        f"Expected {len(STUB_EVENTS)} chunks (one per event), "
        f"got {len(chunks)}"
    )


# ---------------------------------------------------------------------------
# AC6: All event bytes are byte-intact
# ---------------------------------------------------------------------------


def test_all_event_bytes_received_intact():
    """AC6: every byte from the upstream SSE stream arrives byte-for-byte."""
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        received = b""
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            for chunk in resp.iter_bytes():
                received += chunk

    expected = b"".join(STUB_EVENTS)
    assert received == expected


# ---------------------------------------------------------------------------
# AC4: Client disconnect closes upstream connection cleanly
# ---------------------------------------------------------------------------


def test_client_disconnect_closes_upstream_generator():
    """AC4: when the client stops reading, the upstream generator is cleaned up."""
    stream_exhausted = False
    close_called = False

    class _TrackingStreamResp:
        status_code = 200
        headers = httpx.Headers({"content-type": "text/event-stream"})

        async def aiter_bytes(self):
            nonlocal stream_exhausted
            try:
                for event in STUB_EVENTS:
                    yield event
                stream_exhausted = True
            except GeneratorExit:
                pass

    class _TrackingClient(MockSSEClient):
        @contextlib.asynccontextmanager
        async def stream(self, method, url, *, content, headers, **kwargs):
            nonlocal close_called
            try:
                yield _TrackingStreamResp()
            finally:
                close_called = True

    mock = _TrackingClient()
    with TestClient(app, raise_server_exceptions=False) as tc:
        _setup(mock)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            # Read only first chunk then stop (simulates disconnect)
            for chunk in resp.iter_bytes():
                if chunk:
                    break

    # The upstream context manager's finally block must have run.
    assert close_called, "Upstream stream context must be closed on client disconnect"


# ---------------------------------------------------------------------------
# AC5: Request headers forwarded via _filter_headers (no duplication)
# ---------------------------------------------------------------------------


def test_request_headers_forwarded_to_upstream():
    """AC5: non-hop-by-hop request headers reach the upstream."""
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={
                "content-type": "application/json",
                "authorization": "Bearer sk-test",
                "x-api-key": "key-123",
                "anthropic-version": "2023-06-01",
            },
        ) as resp:
            list(resp.iter_bytes())

    fwd = mock.stream_calls[0]["headers"]
    assert fwd.get("authorization") == "Bearer sk-test"
    assert fwd.get("x-api-key") == "key-123"
    assert fwd.get("anthropic-version") == "2023-06-01"


def test_hop_by_hop_headers_not_forwarded_on_stream():
    """AC5: hop-by-hop headers are stripped before forwarding."""
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            list(resp.iter_bytes())

    fwd = set(mock.stream_calls[0]["headers"].keys())
    for hop in ("connection", "keep-alive", "transfer-encoding", "host"):
        assert hop not in fwd, f"Hop-by-hop header '{hop}' must not reach upstream"


# ---------------------------------------------------------------------------
# AC7: Non-streaming requests are unaffected
# ---------------------------------------------------------------------------


def test_non_streaming_body_uses_buffered_path():
    """AC7: body without stream key uses client.post(), not client.stream()."""
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        resp = tc.post(
            "/v1/messages",
            content=NON_STREAMING_BODY,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200
    assert len(mock.post_calls) == 1
    assert len(mock.stream_calls) == 0


def test_stream_false_uses_buffered_path():
    """AC7: stream:false explicitly uses client.post()."""
    body = json.dumps({
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }).encode()
    mock = MockSSEClient()
    with TestClient(app) as tc:
        _setup(mock)
        resp = tc.post(
            "/v1/messages",
            content=body,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200
    assert len(mock.post_calls) == 1
    assert len(mock.stream_calls) == 0


def test_non_streaming_response_body_intact():
    """AC7: buffered response body is returned unchanged."""
    mock = MockSSEClient(buffered_body=SAMPLE_BUFFERED_RESP)
    with TestClient(app) as tc:
        _setup(mock)
        resp = tc.post(
            "/v1/messages",
            content=NON_STREAMING_BODY,
            headers={"content-type": "application/json"},
        )
    assert resp.content == SAMPLE_BUFFERED_RESP


def test_non_streaming_upstream_url_correct():
    """AC7: non-streaming path forwards to {upstream_base_url}/v1/messages."""
    mock = MockSSEClient()
    custom_upstream = "https://proxy.internal.example.com"
    with TestClient(app) as tc:
        _setup(mock, upstream=custom_upstream)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_BODY,
            headers={"content-type": "application/json"},
        )
    assert mock.post_calls[0]["url"] == f"{custom_upstream}/v1/messages"


def test_streaming_upstream_url_correct():
    """AC6: streaming path forwards to {upstream_base_url}/v1/messages."""
    mock = MockSSEClient()
    custom_upstream = "https://proxy.internal.example.com"
    with TestClient(app) as tc:
        _setup(mock, upstream=custom_upstream)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            list(resp.iter_bytes())
    assert mock.stream_calls[0]["url"] == f"{custom_upstream}/v1/messages"
