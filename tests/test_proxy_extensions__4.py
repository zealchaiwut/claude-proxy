"""Tests for issue #4: extend proxy passthrough and harden upstream error handling."""
import json
from unittest.mock import MagicMock, AsyncMock

import httpx
from fastapi.testclient import TestClient

from main import app
from config import Settings

UPSTREAM = "http://upstream.test"

COUNT_TOKENS_BODY = json.dumps(
    {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "hello"}],
    }
).encode()

COUNT_TOKENS_RESPONSE = json.dumps({"input_tokens": 10}).encode()

MODELS_RESPONSE = json.dumps(
    {"data": [{"id": "claude-3-haiku-20240307", "type": "model"}]}
).encode()


def _make_mock_client(
    status: int = 200,
    response_body: bytes = COUNT_TOKENS_RESPONSE,
    response_headers: dict | None = None,
) -> tuple[object, dict]:
    captured: dict = {}

    async def _post(url, *, content, headers, **kwargs):
        captured["method"] = "POST"
        captured["url"] = url
        captured["content"] = content
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status
        mock_resp.content = response_body
        mock_resp.headers = response_headers or {"content-type": "application/json"}
        return mock_resp

    async def _get(url, *, headers, **kwargs):
        captured["method"] = "GET"
        captured["url"] = url
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status
        mock_resp.content = response_body
        mock_resp.headers = response_headers or {"content-type": "application/json"}
        return mock_resp

    async def _request(method, url, *, content=None, headers, **kwargs):
        captured["method"] = method
        captured["url"] = url
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status
        mock_resp.content = response_body
        mock_resp.headers = response_headers or {"content-type": "application/json"}
        return mock_resp

    client = MagicMock()
    client.post = _post
    client.get = _get
    client.request = _request
    client.aclose = AsyncMock()
    return client, captured


def _make_connect_error_client() -> object:
    async def _raise_connect(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    async def _raise_connect_request(method, url, **kwargs):
        raise httpx.ConnectError("connection refused")

    client = MagicMock()
    client.post = _raise_connect
    client.get = _raise_connect
    client.request = _raise_connect_request
    client.aclose = AsyncMock()
    return client


def _make_timeout_client() -> object:
    async def _raise_timeout(*args, **kwargs):
        raise httpx.ReadTimeout("read timed out")

    async def _raise_timeout_request(method, url, **kwargs):
        raise httpx.ReadTimeout("read timed out")

    client = MagicMock()
    client.post = _raise_timeout
    client.get = _raise_timeout
    client.request = _raise_timeout_request
    client.aclose = AsyncMock()
    return client


def _setup(mock_client, upstream: str = UPSTREAM) -> None:
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=upstream)


# --- AC: POST /v1/messages/count_tokens transparent proxy ---

def test_count_tokens_returns_upstream_response():
    """AC: POST /v1/messages/count_tokens returns upstream's response body and status unchanged."""
    mock_client, _ = _make_mock_client(status=200, response_body=COUNT_TOKENS_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=COUNT_TOKENS_BODY,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200
    assert resp.content == COUNT_TOKENS_RESPONSE


def test_count_tokens_upstream_error_status_propagated():
    """AC: POST /v1/messages/count_tokens propagates non-200 upstream status."""
    error_body = json.dumps({"type": "error", "error": {"type": "invalid_request_error"}}).encode()
    mock_client, _ = _make_mock_client(status=400, response_body=error_body)
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=COUNT_TOKENS_BODY,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 400


# --- AC: GET /v1/models transparent proxy ---

def test_models_returns_upstream_response():
    """AC: GET /v1/models returns upstream's response body and status unchanged."""
    mock_client, _ = _make_mock_client(status=200, response_body=MODELS_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.get("/v1/models")
    assert resp.status_code == 200
    assert resp.content == MODELS_RESPONSE


# --- AC: Catch-all /v1/{path} forwarding ---

def test_catchall_forwards_unknown_path():
    """AC: Any request to unrecognised /v1/{path} is forwarded upstream, not rejected 404."""
    mock_client, captured = _make_mock_client(
        status=200,
        response_body=json.dumps({"ok": True}).encode(),
    )
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.get("/v1/beta/some-future-feature")
    assert resp.status_code != 404
    assert resp.status_code == 200


def test_catchall_does_not_override_known_routes():
    """AC: Known routes like /v1/messages still work as before."""
    response_body = json.dumps({"id": "msg_01", "type": "message"}).encode()
    mock_client, captured = _make_mock_client(status=200, response_body=response_body)
    with TestClient(app) as tc:
        _setup(mock_client)
        resp = tc.post(
            "/v1/messages",
            content=COUNT_TOKENS_BODY,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200


# --- AC: 502 on upstream connection failure ---

def test_connect_error_returns_502():
    """AC: Upstream TCP connection failure returns 502 with structured JSON body."""
    with TestClient(app) as tc:
        _setup(_make_connect_error_client())
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=COUNT_TOKENS_BODY,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 502
    body = resp.json()
    assert body["error"] == "bad_gateway"
    assert body["message"] == "upstream unreachable"


def test_connect_error_no_stack_trace():
    """AC: 502 response body contains no stack trace or secrets."""
    with TestClient(app) as tc:
        _setup(_make_connect_error_client())
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=COUNT_TOKENS_BODY,
            headers={"content-type": "application/json"},
        )
    body_text = resp.text
    assert "Traceback" not in body_text
    assert "traceback" not in body_text


# --- AC: 504 on upstream timeout ---

def test_read_timeout_returns_504():
    """AC: Upstream read/response timeout returns 504 with structured JSON body."""
    with TestClient(app) as tc:
        _setup(_make_timeout_client())
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=COUNT_TOKENS_BODY,
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 504
    body = resp.json()
    assert body["error"] == "gateway_timeout"
    assert body["message"] == "upstream timed out"


def test_read_timeout_no_stack_trace():
    """AC: 504 response body contains no stack trace."""
    with TestClient(app) as tc:
        _setup(_make_timeout_client())
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=COUNT_TOKENS_BODY,
            headers={"content-type": "application/json"},
        )
    body_text = resp.text
    assert "Traceback" not in body_text
    assert "traceback" not in body_text


def test_catchall_connect_error_returns_502():
    """AC: Connection error on catch-all route also returns 502."""
    with TestClient(app) as tc:
        _setup(_make_connect_error_client())
        resp = tc.get("/v1/beta/future")
    assert resp.status_code == 502
    assert resp.json()["error"] == "bad_gateway"


def test_catchall_timeout_returns_504():
    """AC: Timeout on catch-all route also returns 504."""
    with TestClient(app) as tc:
        _setup(_make_timeout_client())
        resp = tc.get("/v1/beta/future")
    assert resp.status_code == 504
    assert resp.json()["error"] == "gateway_timeout"
