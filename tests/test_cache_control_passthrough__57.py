"""Tests for issue #57: preserve cache_control markers on anthropic passthrough.

AC coverage:
- AC1: cache_control fields in system blocks forwarded to upstream with block order intact.
- AC2: cache_read_input_tokens and cache_creation_input_tokens from upstream preserved verbatim.
- AC3: forwarded body is structurally equivalent to the original (all cache_control fields and
       block ordering intact) — covers the direct passthrough and model_map rewrite paths.
- AC4: no existing passthrough test regresses (verified by running the full suite).
- AC5: no intentional transformation touches cache_control; the hop-by-hop filter only strips
       HTTP headers, never JSON body fields.
"""
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig

UPSTREAM = "http://stub-anthropic.test"

# ---------------------------------------------------------------------------
# Request fixture: two-block system prompt, first block has cache_control
# ---------------------------------------------------------------------------

CACHE_SYSTEM = [
    {
        "type": "text",
        "text": "You are a helpful assistant with a large knowledge base.",
        "cache_control": {"type": "ephemeral"},
    },
    {
        "type": "text",
        "text": "Additional context that must follow the cached block.",
    },
]

CACHE_REQUEST_BODY = json.dumps(
    {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "system": CACHE_SYSTEM,
        "messages": [{"role": "user", "content": "Hello"}],
    }
).encode()

# Response fixture with cache token fields populated (second-request scenario)
CACHE_HIT_RESPONSE = json.dumps(
    {
        "id": "msg_cache_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I help?"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 8,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 347,
        },
    }
).encode()

# First-request response where cache is being created
CACHE_CREATION_RESPONSE = json.dumps(
    {
        "id": "msg_cache_02",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Sure!"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 357,
            "output_tokens": 5,
            "cache_creation_input_tokens": 347,
            "cache_read_input_tokens": 0,
        },
    }
).encode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(response_body: bytes, status: int = 200):
    """Return (mock_client, forwarded) where forwarded is populated by the first POST call."""
    forwarded: dict = {}

    async def _post(url, *, content, headers, **kwargs):
        forwarded["url"] = url
        forwarded["content"] = content
        forwarded["headers"] = {k.lower(): v for k, v in dict(headers).items()}
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status
        resp.content = response_body
        resp.headers = {"content-type": "application/json"}
        return resp

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()
    return client, forwarded


def _setup_legacy(mock_client, upstream: str = UPSTREAM) -> None:
    """Legacy (env-var) path: no registry, no config_from_file."""
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=upstream)
    app.state.config_from_file = False
    if hasattr(app.state, "profile_registry"):
        del app.state.profile_registry


def _setup_registry(mock_client, registry: ProfileRegistry, upstream: str = UPSTREAM) -> None:
    """Registry path: config loaded from file."""
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=upstream)
    app.state.profile_registry = registry
    app.state.config_from_file = True


def _make_registry(**profiles: ProfileConfig) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles=profiles))


# ---------------------------------------------------------------------------
# SSE mock for streaming tests
# ---------------------------------------------------------------------------

# SSE stream that includes a message_start with cache_read_input_tokens
CACHE_SSE_STREAM = [
    b'data: {"type": "message_start", "message": {"id": "msg_s01", "role": "assistant", '
    b'"content": [], "model": "claude-3-5-sonnet-20241022", "stop_reason": null, '
    b'"usage": {"input_tokens": 10, "output_tokens": 0, '
    b'"cache_read_input_tokens": 347, "cache_creation_input_tokens": 0}}}\n\n',
    b'data: {"type": "content_block_start", "index": 0, '
    b'"content_block": {"type": "text", "text": ""}}\n\n',
    b'data: {"type": "content_block_delta", "index": 0, '
    b'"delta": {"type": "text_delta", "text": "Hello!"}}\n\n',
    b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
    b'"usage": {"output_tokens": 5}}\n\n',
    b"data: [DONE]\n\n",
]


class _MockStreamResponse:
    def __init__(self, events: list[bytes], status_code: int = 200):
        self.status_code = status_code
        self._events = events
        self.headers = httpx.Headers({"content-type": "text/event-stream"})

    async def aiter_bytes(self):
        for event in self._events:
            yield event


class MockSSEClient:
    def __init__(self, events: list[bytes], stream_status: int = 200):
        self.events = events
        self.stream_status = stream_status
        self.stream_calls: list[dict] = []

    @contextlib.asynccontextmanager
    async def stream(self, method, url, *, content, headers, **kwargs):
        self.stream_calls.append(
            {
                "method": method,
                "url": url,
                "content": content,
                "headers": {k.lower(): v for k, v in dict(headers).items()},
            }
        )
        yield _MockStreamResponse(self.events, self.stream_status)

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# AC1 + AC3: cache_control in system blocks forwarded — legacy passthrough path
# ---------------------------------------------------------------------------


def test_cache_control_forwarded_legacy_path():
    """AC1/AC3: cache_control on system block reaches upstream unchanged (legacy path)."""
    client, forwarded = _make_mock_client(CACHE_HIT_RESPONSE)
    with TestClient(app) as tc:
        _setup_legacy(client)
        tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    fwd_body = json.loads(forwarded["content"])
    system = fwd_body["system"]
    assert system[0].get("cache_control") == {"type": "ephemeral"}, (
        "First system block must have cache_control: {type: ephemeral}"
    )
    assert "cache_control" not in system[1], (
        "Second system block must not have cache_control"
    )


def test_system_block_order_preserved_legacy_path():
    """AC1: system blocks forwarded in original order (legacy path)."""
    client, forwarded = _make_mock_client(CACHE_HIT_RESPONSE)
    with TestClient(app) as tc:
        _setup_legacy(client)
        tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    fwd_body = json.loads(forwarded["content"])
    system = fwd_body["system"]
    assert len(system) == 2, "Both system blocks must be forwarded"
    assert system[0]["text"] == CACHE_SYSTEM[0]["text"], "First block text must be at index 0"
    assert system[1]["text"] == CACHE_SYSTEM[1]["text"], "Second block text must be at index 1"


def test_forwarded_body_structurally_equivalent_to_original():
    """AC3: forwarded body is structurally equivalent — all fields including cache_control intact."""
    client, forwarded = _make_mock_client(CACHE_HIT_RESPONSE)
    with TestClient(app) as tc:
        _setup_legacy(client)
        tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    original = json.loads(CACHE_REQUEST_BODY)
    fwd_body = json.loads(forwarded["content"])
    assert fwd_body == original, (
        "Forwarded body must be structurally equivalent to the original request body"
    )


# ---------------------------------------------------------------------------
# AC1 + AC3: cache_control forwarded — registry anthropic profile path
# ---------------------------------------------------------------------------


def test_cache_control_forwarded_registry_path():
    """AC1/AC3: cache_control on system block reaches upstream unchanged (registry path)."""
    registry = _make_registry(
        anthropic=ProfileConfig(kind="passthrough", upstream=UPSTREAM),
    )
    client, forwarded = _make_mock_client(CACHE_HIT_RESPONSE)
    with TestClient(app) as tc:
        _setup_registry(client, registry)
        tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
            },
        )

    fwd_body = json.loads(forwarded["content"])
    system = fwd_body["system"]
    assert system[0].get("cache_control") == {"type": "ephemeral"}
    assert len(system) == 2
    assert system[0]["text"] == CACHE_SYSTEM[0]["text"]
    assert system[1]["text"] == CACHE_SYSTEM[1]["text"]


# ---------------------------------------------------------------------------
# AC3: cache_control preserved through model_map rewrite path
# ---------------------------------------------------------------------------


def test_cache_control_preserved_through_model_map_rewrite():
    """AC3: model_map rewrite re-serialises the body but must keep cache_control intact."""
    client_model = "claude-3-5-sonnet-20241022"
    upstream_model = "claude-3-5-sonnet-20241022-v2"

    registry = _make_registry(
        anthropic=ProfileConfig(
            kind="passthrough",
            upstream=UPSTREAM,
            model_map={client_model: upstream_model},
        ),
    )
    client, forwarded = _make_mock_client(CACHE_HIT_RESPONSE)
    with TestClient(app) as tc:
        _setup_registry(client, registry)
        tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
            },
        )

    fwd_body = json.loads(forwarded["content"])
    # Model is correctly rewritten
    assert fwd_body["model"] == upstream_model, "model_map must rewrite the model name"

    # cache_control and system block order are preserved after re-serialisation
    system = fwd_body["system"]
    assert len(system) == 2
    assert system[0].get("cache_control") == {"type": "ephemeral"}, (
        "cache_control must survive model_map re-serialisation"
    )
    assert system[0]["text"] == CACHE_SYSTEM[0]["text"]
    assert system[1]["text"] == CACHE_SYSTEM[1]["text"]
    assert "cache_control" not in system[1]


# ---------------------------------------------------------------------------
# AC2: cache token fields in response preserved verbatim
# ---------------------------------------------------------------------------


def test_cache_read_input_tokens_preserved_in_response():
    """AC2: cache_read_input_tokens from upstream is NOT stripped from the client response."""
    client, _ = _make_mock_client(CACHE_HIT_RESPONSE)
    with TestClient(app) as tc:
        _setup_legacy(client)
        resp = tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    body = resp.json()
    usage = body.get("usage", {})
    assert "cache_read_input_tokens" in usage, (
        "cache_read_input_tokens must be present in the response usage"
    )
    assert usage["cache_read_input_tokens"] == 347, (
        "cache_read_input_tokens value must be forwarded verbatim from upstream"
    )


def test_cache_creation_input_tokens_preserved_in_response():
    """AC2: cache_creation_input_tokens from upstream is NOT stripped from the client response."""
    client, _ = _make_mock_client(CACHE_CREATION_RESPONSE)
    with TestClient(app) as tc:
        _setup_legacy(client)
        resp = tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    body = resp.json()
    usage = body.get("usage", {})
    assert "cache_creation_input_tokens" in usage, (
        "cache_creation_input_tokens must be present in the response usage"
    )
    assert usage["cache_creation_input_tokens"] == 347, (
        "cache_creation_input_tokens value must be forwarded verbatim from upstream"
    )


def test_response_body_byte_identical_to_upstream():
    """AC2/AC3: response body bytes are identical to what upstream returned."""
    client, _ = _make_mock_client(CACHE_HIT_RESPONSE)
    with TestClient(app) as tc:
        _setup_legacy(client)
        resp = tc.post(
            "/v1/messages",
            content=CACHE_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.content == CACHE_HIT_RESPONSE, (
        "Response body must be byte-identical to upstream response — no field stripping"
    )


# ---------------------------------------------------------------------------
# AC2: cache token fields preserved in streaming response
# ---------------------------------------------------------------------------

STREAMING_CACHE_BODY = json.dumps(
    {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "system": CACHE_SYSTEM,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
).encode()


def test_cache_tokens_preserved_in_streaming_response():
    """AC2: cache_read_input_tokens in SSE stream passes through to client byte-for-byte."""
    mock = MockSSEClient(events=CACHE_SSE_STREAM)

    received = b""
    with TestClient(app) as tc:
        app.state.http_client = mock
        app.state.settings = Settings(upstream_base_url=UPSTREAM)
        app.state.config_from_file = False
        if hasattr(app.state, "profile_registry"):
            del app.state.profile_registry

        with tc.stream(
            "POST",
            "/v1/messages",
            content=STREAMING_CACHE_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            for chunk in resp.iter_bytes():
                received += chunk

    # The message_start SSE event (first chunk) must contain cache_read_input_tokens
    assert b"cache_read_input_tokens" in received, (
        "cache_read_input_tokens must survive the streaming passthrough intact"
    )
    assert b'"cache_read_input_tokens": 347' in received, (
        "cache_read_input_tokens value 347 must be forwarded verbatim in the SSE stream"
    )


def test_streaming_cache_control_forwarded():
    """AC1: cache_control in system blocks is forwarded correctly on the streaming path."""
    mock = MockSSEClient(events=CACHE_SSE_STREAM)

    with TestClient(app) as tc:
        app.state.http_client = mock
        app.state.settings = Settings(upstream_base_url=UPSTREAM)
        app.state.config_from_file = False
        if hasattr(app.state, "profile_registry"):
            del app.state.profile_registry

        with tc.stream(
            "POST",
            "/v1/messages",
            content=STREAMING_CACHE_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            list(resp.iter_bytes())

    assert len(mock.stream_calls) == 1
    fwd_body = json.loads(mock.stream_calls[0]["content"])
    system = fwd_body["system"]
    assert system[0].get("cache_control") == {"type": "ephemeral"}, (
        "cache_control must be present on the first system block in the streaming request"
    )
    assert len(system) == 2
    assert system[0]["text"] == CACHE_SYSTEM[0]["text"]
    assert system[1]["text"] == CACHE_SYSTEM[1]["text"]
