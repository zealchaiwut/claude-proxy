"""Tests for issue #59: per-request capture of proxied exchanges.

AC coverage:
- non_streaming: file written, correct request_id filename, no auth material, response matches
- streaming: captured response is reassembled single object (not list of chunks), no auth material
- redaction: helper replaces Authorization and api_key fields with [REDACTED]
- no_capture_default: no file written when CCPROXY_CAPTURE unset and no profile capture=true
- per_profile: capture=true in profile writes file even without env var
- dir_autocreate: captures directory is created automatically
"""
import contextlib
import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig
from services.capture import CaptureService, redact_credentials, reassemble_anthropic_sse

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

STUB_UPSTREAM = "http://stub-anthropic.test"
CLIENT_MODEL = "claude-haiku-4-5-20251001"

NON_STREAMING_REQ_BODY = json.dumps({
    "model": CLIENT_MODEL,
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "hello"}],
}).encode()

STREAMING_REQ_BODY = json.dumps({
    "model": CLIENT_MODEL,
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "hello"}],
    "stream": True,
}).encode()

STUB_ANTHROPIC_RESPONSE = {
    "id": "msg_test123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi there"}],
    "model": CLIENT_MODEL,
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

STUB_SSE_EVENTS = [
    b'data: {"type":"message_start","message":{"id":"msg_sse1","type":"message","role":"assistant",'
    b'"model":"claude-haiku-4-5-20251001","usage":{"input_tokens":5,"output_tokens":0}}}\n\n',
    b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
    b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
    b'data: {"type":"content_block_stop","index":0}\n\n',
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
    b'data: {"type":"message_stop"}\n\n',
]


# ---------------------------------------------------------------------------
# Mock httpx clients
# ---------------------------------------------------------------------------


def _make_passthrough_client(response_body: bytes):
    async def _post(url, *, content, headers, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = response_body
        resp.headers = {"content-type": "application/json"}
        return resp

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()
    return client


class _MockStreamResponse:
    def __init__(self, events: list[bytes]):
        self.status_code = 200
        self._events = events
        self.headers = httpx.Headers({"content-type": "text/event-stream"})

    async def aiter_bytes(self):
        for event in self._events:
            yield event


class _MockSSEClient:
    def __init__(self, events: list[bytes]):
        self._events = events

    @contextlib.asynccontextmanager
    async def stream(self, method, url, *, content, headers, **kwargs):
        yield _MockStreamResponse(self._events)

    async def post(self, url, *, content, headers, **kwargs):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = b"{}"
        resp.headers = {"content-type": "application/json"}
        return resp

    async def aclose(self):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry(capture: bool = False) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        "anthropic": ProfileConfig(
            kind="passthrough",
            upstream=STUB_UPSTREAM,
            capture=capture,
        ),
    }))


def _setup_app(mock_client, registry: ProfileRegistry, capture_service: CaptureService) -> None:
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=STUB_UPSTREAM)
    app.state.profile_registry = registry
    app.state.config_from_file = True
    app.state.capture_service = capture_service
    # No request_logger for these tests; cost-only path is irrelevant here
    app.state.request_logger = None


# ---------------------------------------------------------------------------
# AC: redact_credentials helper
# ---------------------------------------------------------------------------


def test_redaction_replaces_authorization():
    """AC: redact_credentials replaces Authorization field with [REDACTED]."""
    data = {"Authorization": "Bearer sk-secret-123", "model": "claude-3"}
    result = redact_credentials(data)
    assert result["Authorization"] == "[REDACTED]"
    assert result["model"] == "claude-3"


def test_redaction_replaces_api_key():
    """AC: redact_credentials replaces api_key field with [REDACTED]."""
    data = {"api_key": "sk-secret", "messages": []}
    result = redact_credentials(data)
    assert result["api_key"] == "[REDACTED]"
    assert result["messages"] == []


def test_redaction_replaces_x_api_key():
    """AC: redact_credentials replaces x-api-key field with [REDACTED]."""
    data = {"x-api-key": "key-val", "other": "safe"}
    result = redact_credentials(data)
    assert result["x-api-key"] == "[REDACTED]"
    assert result["other"] == "safe"


def test_redaction_nested_dict():
    """AC: redaction recurses into nested dicts."""
    data = {
        "headers": {
            "Authorization": "Bearer token",
            "content-type": "application/json",
        }
    }
    result = redact_credentials(data)
    assert result["headers"]["Authorization"] == "[REDACTED]"
    assert result["headers"]["content-type"] == "application/json"


def test_redaction_in_list():
    """AC: redaction recurses into lists."""
    data = [{"api_key": "secret"}, {"safe": "value"}]
    result = redact_credentials(data)
    assert result[0]["api_key"] == "[REDACTED]"
    assert result[1]["safe"] == "value"


# ---------------------------------------------------------------------------
# AC: non-streaming capture
# ---------------------------------------------------------------------------


def test_non_streaming_capture_writes_one_file(tmp_path, monkeypatch):
    """AC: non-streaming exchange with capture enabled writes exactly one file."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry(capture=False)
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        resp = tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
            },
        )

    assert resp.status_code == 200
    files = list(capture_dir.glob("*.json"))
    assert len(files) == 1, f"Expected 1 capture file, got {len(files)}"


def test_non_streaming_capture_filename_is_uuid_request_id(tmp_path, monkeypatch):
    """AC: capture filename is <request_id>.json; request_id inside matches filename."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    files = list(capture_dir.glob("*.json"))
    assert len(files) == 1
    stem = files[0].stem
    # Stem must be a valid UUID4
    parsed = uuid.UUID(stem, version=4)
    assert str(parsed) == stem
    # request_id field inside file must match
    data = json.loads(files[0].read_text())
    assert data["request_id"] == stem


def test_non_streaming_capture_no_auth_material(tmp_path, monkeypatch):
    """AC: capture file contains no Authorization or api-key credential values."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                "authorization": "Bearer sk-super-secret",
                "x-api-key": "key-super-secret",
            },
        )

    files = list(capture_dir.glob("*.json"))
    raw = files[0].read_text()
    assert "sk-super-secret" not in raw
    assert "key-super-secret" not in raw


def test_non_streaming_capture_response_body_matches(tmp_path, monkeypatch):
    """AC: capture file response field matches upstream response body."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    files = list(capture_dir.glob("*.json"))
    data = json.loads(files[0].read_text())
    assert data["response"]["id"] == STUB_ANTHROPIC_RESPONSE["id"]
    assert data["response"]["stop_reason"] == "end_turn"


def test_non_streaming_capture_contains_timing(tmp_path, monkeypatch):
    """AC: capture file contains start timestamp and duration_ms."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    files = list(capture_dir.glob("*.json"))
    data = json.loads(files[0].read_text())
    assert "timing" in data
    assert "start" in data["timing"]
    assert "duration_ms" in data["timing"]
    assert isinstance(data["timing"]["duration_ms"], (int, float))
    assert data["timing"]["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# AC: streaming capture
# ---------------------------------------------------------------------------


def test_streaming_capture_writes_reassembled_object(tmp_path, monkeypatch):
    """AC: streaming exchange writes a single reassembled response object, not chunks."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _MockSSEClient(STUB_SSE_EVENTS)

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        ) as resp:
            list(resp.iter_bytes())

    files = list(capture_dir.glob("*.json"))
    assert len(files) == 1, f"Expected 1 capture file, got {len(files)}"
    data = json.loads(files[0].read_text())

    # response must be a single dict, not a list of SSE chunks
    assert isinstance(data["response"], dict), "response must be a single assembled object"
    assert data["response"].get("role") == "assistant"


def test_streaming_capture_assembles_text_from_deltas(tmp_path, monkeypatch):
    """AC: streaming capture file has complete text from assembled content_block_deltas."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _MockSSEClient(STUB_SSE_EVENTS)

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        ) as resp:
            list(resp.iter_bytes())

    files = list(capture_dir.glob("*.json"))
    data = json.loads(files[0].read_text())
    content_blocks = data["response"].get("content", [])
    full_text = "".join(
        b.get("text", "") for b in content_blocks if isinstance(b, dict) and b.get("type") == "text"
    )
    assert "Hello" in full_text
    assert "world" in full_text


def test_streaming_capture_no_auth_material(tmp_path, monkeypatch):
    """AC: streaming capture file contains no auth or api-key credential values."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _MockSSEClient(STUB_SSE_EVENTS)

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        with tc.stream(
            "POST", "/v1/messages",
            content=STREAMING_REQ_BODY,
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                "authorization": "Bearer sk-stream-secret",
            },
        ) as resp:
            list(resp.iter_bytes())

    files = list(capture_dir.glob("*.json"))
    raw = files[0].read_text()
    assert "sk-stream-secret" not in raw


# ---------------------------------------------------------------------------
# AC: per-profile capture (env var absent)
# ---------------------------------------------------------------------------


def test_per_profile_capture_writes_file_without_env_var(tmp_path, monkeypatch):
    """AC: capture=true in profile config writes file even when CCPROXY_CAPTURE is unset."""
    monkeypatch.delenv("CCPROXY_CAPTURE", raising=False)
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry(capture=True)
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    files = list(capture_dir.glob("*.json"))
    assert len(files) == 1, "profile capture=true must write file without env var"


def test_profile_without_capture_flag_writes_no_file(tmp_path, monkeypatch):
    """AC: profile without capture=true and no env var writes no capture file."""
    monkeypatch.delenv("CCPROXY_CAPTURE", raising=False)
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = ProfileRegistry(ProxyConfig(profiles={
        "with-capture": ProfileConfig(kind="passthrough", upstream=STUB_UPSTREAM, capture=True),
        "no-capture": ProfileConfig(kind="passthrough", upstream=STUB_UPSTREAM, capture=False),
    }))
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "no-capture"},
        )

    files = list(capture_dir.glob("*.json")) if capture_dir.exists() else []
    assert len(files) == 0, "profile without capture=true must not write any file"


# ---------------------------------------------------------------------------
# AC: no file written when capture is fully disabled
# ---------------------------------------------------------------------------


def test_no_file_written_when_capture_fully_off(tmp_path, monkeypatch):
    """AC: no capture file written when CCPROXY_CAPTURE is unset and profile has no capture=true."""
    monkeypatch.delenv("CCPROXY_CAPTURE", raising=False)
    capture_dir = tmp_path / "captures"
    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry(capture=False)
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        resp = tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    assert resp.status_code == 200
    files = list(capture_dir.glob("*.json")) if capture_dir.exists() else []
    assert len(files) == 0, "No capture file must be written when capture is fully off"


# ---------------------------------------------------------------------------
# AC: capture directory created automatically
# ---------------------------------------------------------------------------


def test_capture_dir_created_automatically(tmp_path, monkeypatch):
    """AC: captures directory (including parents) is created automatically on first write."""
    monkeypatch.setenv("CCPROXY_CAPTURE", "1")
    capture_dir = tmp_path / "deep" / "nested" / "captures"
    assert not capture_dir.exists()

    svc = CaptureService(capture_dir=capture_dir)
    registry = _make_registry()
    client = _make_passthrough_client(json.dumps(STUB_ANTHROPIC_RESPONSE).encode())

    with TestClient(app) as tc:
        _setup_app(client, registry, svc)
        tc.post(
            "/v1/messages",
            content=NON_STREAMING_REQ_BODY,
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    assert capture_dir.exists(), "Capture directory must be created automatically"
    assert len(list(capture_dir.glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Unit tests for reassemble_anthropic_sse
# ---------------------------------------------------------------------------


def test_reassemble_sse_returns_dict():
    """reassemble_anthropic_sse returns a single dict, not a list."""
    buf = b"".join(STUB_SSE_EVENTS)
    result = reassemble_anthropic_sse(buf)
    assert isinstance(result, dict)


def test_reassemble_sse_assembles_text_deltas():
    """reassemble_anthropic_sse concatenates all text_delta chunks."""
    buf = b"".join(STUB_SSE_EVENTS)
    result = reassemble_anthropic_sse(buf)
    content = result.get("content", [])
    text = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    assert "Hello" in text
    assert "world" in text


def test_reassemble_sse_includes_stop_reason():
    """reassemble_anthropic_sse includes stop_reason from message_delta."""
    buf = b"".join(STUB_SSE_EVENTS)
    result = reassemble_anthropic_sse(buf)
    assert result.get("stop_reason") == "end_turn"


def test_reassemble_sse_merges_usage():
    """reassemble_anthropic_sse merges usage from message_start and message_delta."""
    buf = b"".join(STUB_SSE_EVENTS)
    result = reassemble_anthropic_sse(buf)
    usage = result.get("usage", {})
    assert usage.get("input_tokens") == 5
    assert usage.get("output_tokens") == 2
