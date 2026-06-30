"""Tests for issue #2: transparent non-streaming POST /v1/messages passthrough."""
import json
from unittest.mock import MagicMock, AsyncMock
import httpx
from fastapi.testclient import TestClient

from main import app
from config import Settings

UPSTREAM = "http://upstream.test"

SAMPLE_BODY = json.dumps(
    {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
    }
).encode()

SAMPLE_RESPONSE = json.dumps(
    {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi!"}],
        "model": "claude-3-haiku-20240307",
        "stop_reason": "end_turn",
    }
).encode()


def _make_mock_client(
    status: int = 200,
    response_body: bytes = SAMPLE_RESPONSE,
    response_headers: dict | None = None,
) -> tuple[object, dict]:
    """Return (mock_client, captured_dict).  captured_dict is populated on first call."""
    captured: dict = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["url"] = url
        captured["content"] = content
        captured["headers"] = {k.lower(): v for k, v in dict(headers).items()}
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status
        mock_resp.content = response_body
        mock_resp.headers = response_headers or {"content-type": "application/json"}
        return mock_resp

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()
    return client, captured


def _setup(mock_client, upstream: str = UPSTREAM) -> None:
    """Install mock client and settings into app.state."""
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=upstream)


# --- AC (a): body and non-hop-by-hop headers pass through unchanged ---

def test_body_forwarded_byte_for_byte():
    """AC(a): raw request body is forwarded unchanged."""
    mock_client, captured = _make_mock_client()
    with TestClient(app) as tc:
        _setup(mock_client)
        tc.post("/v1/messages", content=SAMPLE_BODY, headers={"content-type": "application/json"})
    assert captured["content"] == SAMPLE_BODY


def test_non_hop_by_hop_headers_forwarded():
    """AC(a): non-hop-by-hop headers pass through unchanged."""
    mock_client, captured = _make_mock_client()
    with TestClient(app) as tc:
        _setup(mock_client)
        tc.post(
            "/v1/messages",
            content=SAMPLE_BODY,
            headers={
                "authorization": "Bearer test-token",
                "x-api-key": "sk-test",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "tools-2024-04-04",
                "content-type": "application/json",
                "x-custom-header": "my-value",
            },
        )
    h = captured["headers"]
    assert h.get("authorization") == "Bearer test-token"
    assert h.get("x-api-key") == "sk-test"
    assert h.get("anthropic-version") == "2023-06-01"
    assert h.get("anthropic-beta") == "tools-2024-04-04"
    assert h.get("x-custom-header") == "my-value"


# --- AC (b): auth headers arrive exactly as received ---

def test_auth_headers_arrive_exactly():
    """AC(b): Authorization, x-api-key, anthropic-version, anthropic-beta are verbatim."""
    mock_client, captured = _make_mock_client()
    with TestClient(app) as tc:
        _setup(mock_client)
        tc.post(
            "/v1/messages",
            content=SAMPLE_BODY,
            headers={
                "Authorization": "Bearer sk-ant-api03-secret",
                "x-api-key": "sk-ant-api03-key",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "tools-2024-04-04",
                "content-type": "application/json",
            },
        )
    h = captured["headers"]
    assert h["authorization"] == "Bearer sk-ant-api03-secret"
    assert h["x-api-key"] == "sk-ant-api03-key"
    assert h["anthropic-version"] == "2023-06-01"
    assert h["anthropic-beta"] == "tools-2024-04-04"


# --- AC (c): hop-by-hop headers absent from upstream request ---

def test_hop_by_hop_headers_stripped_from_upstream_request():
    """AC(c): hop-by-hop headers are NOT forwarded to upstream."""
    mock_client, captured = _make_mock_client()
    with TestClient(app) as tc:
        _setup(mock_client)
        tc.post(
            "/v1/messages",
            content=SAMPLE_BODY,
            headers={"content-type": "application/json", "authorization": "Bearer token"},
        )
    forwarded = set(captured["headers"].keys())
    hop_by_hop = {
        "host",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
    for h in hop_by_hop:
        assert h not in forwarded, f"Hop-by-hop header '{h}' must not reach upstream"


# --- AC (d): upstream status code and JSON body propagated verbatim ---

def test_upstream_status_code_propagated():
    """AC(d): HTTP status from upstream is returned unchanged."""
    mock_client, _ = _make_mock_client(
        status=422,
        response_body=json.dumps({"type": "error", "error": {"type": "invalid_request_error"}}).encode(),
    )
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.post("/v1/messages", content=SAMPLE_BODY, headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_upstream_json_body_propagated_verbatim():
    """AC(d): response JSON body from upstream is byte-for-byte identical."""
    mock_client, _ = _make_mock_client(status=200, response_body=SAMPLE_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.post("/v1/messages", content=SAMPLE_BODY, headers={"content-type": "application/json"})
    assert resp.status_code == 200
    assert resp.content == SAMPLE_RESPONSE


# --- Additional AC checks ---

def test_upstream_url_is_upstream_base_url_plus_path():
    """AC(1): request forwarded to ${UPSTREAM_BASE_URL}/v1/messages."""
    mock_client, captured = _make_mock_client()
    custom_upstream = "https://proxy.internal.example.com"
    with TestClient(app) as tc:
        _setup(mock_client, upstream=custom_upstream)
        tc.post("/v1/messages", content=SAMPLE_BODY, headers={"content-type": "application/json"})
    assert captured["url"] == f"{custom_upstream}/v1/messages"


def test_hop_by_hop_response_headers_stripped():
    """AC(8): hop-by-hop headers from upstream response are NOT returned to caller."""
    mock_client, _ = _make_mock_client(
        response_headers={
            "content-type": "application/json",
            "x-request-id": "abc123",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
        }
    )
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.post("/v1/messages", content=SAMPLE_BODY, headers={"content-type": "application/json"})
    resp_headers_lower = {k.lower() for k in resp.headers}
    assert "transfer-encoding" not in resp_headers_lower
    assert "connection" not in resp_headers_lower
    assert "x-request-id" in resp_headers_lower
