"""Tests for issue #42: inbound correlation headers stored on request records.

AC coverage:
- (a) all-three-headers: all three headers flow through to the record when supplied
- (b) absent-headers: all three fields are null when headers are omitted
- (c) nonexistent-role: X-CCProxy-Role with non-existent profile does not alter routing
- extra: partial headers, extremely long value, no 5xx on any combination
"""
import io
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig
from services.request_logger import RequestLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUB_ANTHROPIC = "http://stub-anthropic.test"
CLIENT_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_response(msg_id: str = "msg_test") -> bytes:
    return json.dumps({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello"}],
        "model": CLIENT_MODEL,
        "stop_reason": "end_turn",
    }).encode()


def _make_request_body(model: str = CLIENT_MODEL) -> bytes:
    return json.dumps({
        "model": model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hello"}],
    }).encode()


def _make_passthrough_registry(upstream: str = STUB_ANTHROPIC) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        "anthropic": ProfileConfig(kind="passthrough", upstream=upstream),
    }))


def _make_http_client(response_body: bytes) -> object:
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


def _setup(mock_client, registry: ProfileRegistry, capture: io.StringIO) -> None:
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
    app.state.profile_registry = registry
    app.state.config_from_file = True
    app.state.request_logger = RequestLogger(capture=capture)


def _read_record(capture: io.StringIO) -> dict:
    capture.seek(0)
    lines = [ln.strip() for ln in capture.readlines() if ln.strip()]
    assert len(lines) >= 1, f"Expected at least 1 log line, got {len(lines)}"
    return json.loads(lines[0])


# ---------------------------------------------------------------------------
# AC (a): all three headers flow through to the record when supplied
# ---------------------------------------------------------------------------


def test_all_three_headers_stored_in_record():
    """AC (a): X-CCProxy-Run, X-CCProxy-Role, X-CCProxy-Ticket are stored on the record."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_http_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup(client, registry, capture)
        resp = tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                "x-ccproxy-run": "run-42",
                "x-ccproxy-role": "summarizer",
                "x-ccproxy-ticket": "PROJ-99",
            },
        )

    assert resp.status_code == 200
    record = _read_record(capture)
    assert record.get("run_id") == "run-42", f"run_id mismatch: {record.get('run_id')!r}"
    assert record.get("role") == "summarizer", f"role mismatch: {record.get('role')!r}"
    assert record.get("ticket") == "PROJ-99", f"ticket mismatch: {record.get('ticket')!r}"


# ---------------------------------------------------------------------------
# AC (b): all three fields are null when headers are omitted
# ---------------------------------------------------------------------------


def test_absent_headers_yield_null_fields():
    """AC (b): when correlation headers are absent, run_id/role/ticket are null in the record."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_http_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup(client, registry, capture)
        resp = tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                # No correlation headers
            },
        )

    assert resp.status_code == 200
    record = _read_record(capture)
    assert "run_id" in record, "run_id field must be present in the record"
    assert "role" in record, "role field must be present in the record"
    assert "ticket" in record, "ticket field must be present in the record"
    assert record["run_id"] is None, f"run_id must be null when header absent, got {record['run_id']!r}"
    assert record["role"] is None, f"role must be null when header absent, got {record['role']!r}"
    assert record["ticket"] is None, f"ticket must be null when header absent, got {record['ticket']!r}"


# ---------------------------------------------------------------------------
# AC (c): non-existent profile in X-CCProxy-Role does not alter routing
# ---------------------------------------------------------------------------


def test_nonexistent_role_does_not_alter_routing():
    """AC (c): X-CCProxy-Role with a non-existent profile name does not change routing."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_http_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup(client, registry, capture)
        resp = tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                "x-ccproxy-role": "nonexistent-profile",
            },
        )

    # Must still succeed — profile selected normally via x-ccproxy-profile
    assert resp.status_code == 200
    record = _read_record(capture)
    # Raw header value stored in role field
    assert record.get("role") == "nonexistent-profile"
    # Profile routing was not influenced — profile_name is still "anthropic"
    assert record.get("profile_name") == "anthropic"
    assert record.get("profile_kind") == "passthrough"


# ---------------------------------------------------------------------------
# Extra: partial headers — only run_id set
# ---------------------------------------------------------------------------


def test_partial_headers_only_run_id():
    """Extra: when only X-CCProxy-Run is supplied, role and ticket are null."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_http_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup(client, registry, capture)
        resp = tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                "x-ccproxy-run": "run-partial",
            },
        )

    assert resp.status_code == 200
    record = _read_record(capture)
    assert record.get("run_id") == "run-partial"
    assert record["role"] is None
    assert record["ticket"] is None


# ---------------------------------------------------------------------------
# Extra: extremely long header value does not crash
# ---------------------------------------------------------------------------


def test_extremely_long_ticket_header_no_crash():
    """Extra: 4 KB ticket header value does not produce a 5xx."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_http_client(_make_anthropic_response())
    big_value = "X" * 4096

    with TestClient(app) as tc:
        _setup(client, registry, capture)
        resp = tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                "x-ccproxy-ticket": big_value,
            },
        )

    assert resp.status_code < 500, f"Expected non-5xx, got {resp.status_code}"
    record = _read_record(capture)
    # Value is stored as-is (or truncated) — must not be absent
    assert "ticket" in record
    assert record["ticket"] is not None
