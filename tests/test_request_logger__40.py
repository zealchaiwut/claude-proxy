"""Tests for issue #40: structured per-request JSONL logging layer.

AC coverage:
- schema-passthrough: passthrough request emits all 12 required fields with correct types
- schema-openai: OpenAI-style request emits all 12 required fields with correct types
- no-secrets-passthrough: no sensitive fields (api_key, body, content, headers) in passthrough record
- no-secrets-openai: no sensitive fields in OpenAI record
- log-file-override: CCPROXY_LOG_FILE env var writes to custom path
- injectable-logger: tests substitute capture buffer without file I/O
"""
import io
import json
import re
from unittest.mock import AsyncMock, MagicMock

import httpx
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig
from services.request_logger import RequestLogger, _REQUIRED_FIELDS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STUB_ANTHROPIC = "http://stub-anthropic.test"
STUB_OPENAI = "http://stub-openai.test"

CLIENT_MODEL = "claude-haiku-4-5-20251001"
OPENAI_MODEL = "gpt-4o"

_SENSITIVE_FIELD_PATTERN = re.compile(
    r"api_key|authorization|secret|token(?!_drift)|bearer|password|body|content|messages|prompt|header",
    re.IGNORECASE,
)

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


def _make_openai_response() -> bytes:
    return json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": OPENAI_MODEL,
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }).encode()


def _make_request_body(model: str = CLIENT_MODEL, stream: bool = False) -> bytes:
    payload: dict = {
        "model": model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hello"}],
    }
    if stream:
        payload["stream"] = True
    return json.dumps(payload).encode()


def _make_passthrough_registry(upstream: str = STUB_ANTHROPIC) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        "anthropic": ProfileConfig(kind="passthrough", upstream=upstream),
    }))


def _make_openai_registry(
    upstream: str = STUB_OPENAI,
    api_key_env: str = "TEST_OPENAI_KEY",
    model: str = OPENAI_MODEL,
) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles={
        "openai-profile": ProfileConfig(
            kind="openai",
            upstream=upstream,
            api_key_env=api_key_env,
            model=model,
        ),
    }))


def _make_passthrough_client(response_body: bytes) -> object:
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


def _make_openai_client(response_body: bytes) -> object:
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


def _setup_passthrough(mock_client, registry: ProfileRegistry, capture: io.StringIO) -> None:
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
    app.state.profile_registry = registry
    app.state.config_from_file = True
    app.state.request_logger = RequestLogger(capture=capture)


def _setup_openai(mock_client, registry: ProfileRegistry, capture: io.StringIO, monkeypatch) -> None:
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-test-key-do-not-log")
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
    app.state.profile_registry = registry
    app.state.config_from_file = True
    app.state.request_logger = RequestLogger(capture=capture)


def _read_record(capture: io.StringIO) -> dict:
    capture.seek(0)
    lines = [ln.strip() for ln in capture.readlines() if ln.strip()]
    assert len(lines) == 1, f"Expected exactly 1 log line, got {len(lines)}: {lines}"
    return json.loads(lines[0])


# ---------------------------------------------------------------------------
# AC: schema fields — passthrough request
# ---------------------------------------------------------------------------


def test_passthrough_request_emits_all_required_fields():
    """AC: passthrough request produces a record with all 12 required schema fields."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_passthrough_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup_passthrough(client, registry, capture)
        resp = tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
            },
        )

    assert resp.status_code == 200
    record = _read_record(capture)

    missing = _REQUIRED_FIELDS - set(record.keys())
    assert not missing, f"Missing required fields: {missing}"


def test_passthrough_request_field_types():
    """AC: passthrough record fields have correct types."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_passthrough_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup_passthrough(client, registry, capture)
        tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    record = _read_record(capture)
    assert isinstance(record["request_id"], str) and record["request_id"]
    assert isinstance(record["timestamp"], str)
    assert isinstance(record["profile_name"], str)
    assert isinstance(record["profile_kind"], str)
    assert isinstance(record["requested_model"], str)
    assert isinstance(record["upstream_model"], str)
    assert isinstance(record["upstream_host"], str)
    assert isinstance(record["method"], str)
    assert isinstance(record["path"], str)
    assert isinstance(record["status"], int)
    assert isinstance(record["latency_ms"], (int, float))
    assert isinstance(record["streamed"], bool)


def test_passthrough_record_correct_values():
    """AC: passthrough record contains correct field values."""
    capture = io.StringIO()
    registry = _make_passthrough_registry(STUB_ANTHROPIC)
    client = _make_passthrough_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup_passthrough(client, registry, capture)
        tc.post(
            "/v1/messages",
            content=_make_request_body(CLIENT_MODEL),
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    record = _read_record(capture)
    assert record["profile_name"] == "anthropic"
    assert record["profile_kind"] == "passthrough"
    assert record["requested_model"] == CLIENT_MODEL
    assert record["upstream_model"] == CLIENT_MODEL
    assert record["upstream_host"] == "stub-anthropic.test"
    assert record["method"] == "POST"
    assert record["path"] == "/v1/messages"
    assert record["status"] == 200
    assert record["latency_ms"] >= 0
    assert record["streamed"] is False


# ---------------------------------------------------------------------------
# AC: schema fields — OpenAI-style request
# ---------------------------------------------------------------------------


def test_openai_request_emits_all_required_fields(monkeypatch):
    """AC: OpenAI-style request produces a record with all 12 required schema fields."""
    capture = io.StringIO()
    registry = _make_openai_registry()
    client = _make_openai_client(_make_openai_response())

    with TestClient(app) as tc:
        _setup_openai(client, registry, capture, monkeypatch)
        resp = tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "openai-profile",
            },
        )

    assert resp.status_code == 200
    record = _read_record(capture)

    missing = _REQUIRED_FIELDS - set(record.keys())
    assert not missing, f"Missing required fields: {missing}"


def test_openai_request_field_types(monkeypatch):
    """AC: OpenAI record fields have correct types."""
    capture = io.StringIO()
    registry = _make_openai_registry()
    client = _make_openai_client(_make_openai_response())

    with TestClient(app) as tc:
        _setup_openai(client, registry, capture, monkeypatch)
        tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "openai-profile"},
        )

    record = _read_record(capture)
    assert isinstance(record["request_id"], str) and record["request_id"]
    assert isinstance(record["timestamp"], str)
    assert isinstance(record["profile_name"], str)
    assert isinstance(record["profile_kind"], str)
    assert isinstance(record["requested_model"], str)
    assert isinstance(record["upstream_model"], str)
    assert isinstance(record["upstream_host"], str)
    assert isinstance(record["method"], str)
    assert isinstance(record["path"], str)
    assert isinstance(record["status"], int)
    assert isinstance(record["latency_ms"], (int, float))
    assert isinstance(record["streamed"], bool)


def test_openai_record_correct_values(monkeypatch):
    """AC: OpenAI record contains correct field values."""
    capture = io.StringIO()
    registry = _make_openai_registry(STUB_OPENAI, "TEST_OPENAI_KEY", OPENAI_MODEL)
    client = _make_openai_client(_make_openai_response())

    with TestClient(app) as tc:
        _setup_openai(client, registry, capture, monkeypatch)
        tc.post(
            "/v1/messages",
            content=_make_request_body(CLIENT_MODEL),
            headers={"content-type": "application/json", "x-ccproxy-profile": "openai-profile"},
        )

    record = _read_record(capture)
    assert record["profile_name"] == "openai-profile"
    assert record["profile_kind"] == "openai"
    assert record["requested_model"] == CLIENT_MODEL
    assert record["upstream_model"] == OPENAI_MODEL
    assert record["upstream_host"] == "stub-openai.test"
    assert record["method"] == "POST"
    assert record["path"] == "/v1/messages"
    assert record["status"] == 200
    assert record["latency_ms"] >= 0
    assert record["streamed"] is False


# ---------------------------------------------------------------------------
# AC: no secrets in log records
# ---------------------------------------------------------------------------


def test_no_secrets_in_passthrough_record():
    """AC: no API key, header value, body, or content field in passthrough log record."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_passthrough_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup_passthrough(client, registry, capture)
        tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={
                "content-type": "application/json",
                "x-ccproxy-profile": "anthropic",
                "authorization": "Bearer sk-secret-key",
            },
        )

    record = _read_record(capture)
    raw_json = json.dumps(record)

    # No field names that suggest secrets
    for field_name in record.keys():
        assert not _SENSITIVE_FIELD_PATTERN.match(field_name), (
            f"Sensitive field name '{field_name}' found in log record"
        )

    # The actual secret value must not appear anywhere in the serialized record
    assert "sk-secret-key" not in raw_json
    assert "Bearer" not in raw_json
    assert "hello" not in raw_json, "Request body content must not appear in log"


def test_no_secrets_in_openai_record(monkeypatch):
    """AC: no API key, header value, body, or content field in OpenAI log record."""
    capture = io.StringIO()
    registry = _make_openai_registry()
    client = _make_openai_client(_make_openai_response())
    monkeypatch.setenv("TEST_OPENAI_KEY", "sk-openai-do-not-log")

    with TestClient(app) as tc:
        _setup_openai(client, registry, capture, monkeypatch)
        tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "openai-profile"},
        )

    record = _read_record(capture)
    raw_json = json.dumps(record)

    for field_name in record.keys():
        assert not _SENSITIVE_FIELD_PATTERN.match(field_name), (
            f"Sensitive field name '{field_name}' found in log record"
        )

    assert "sk-openai-do-not-log" not in raw_json, "API key must not appear in log record"
    assert "hello" not in raw_json, "Request body content must not appear in log"


# ---------------------------------------------------------------------------
# AC: CCPROXY_LOG_FILE override
# ---------------------------------------------------------------------------


def test_ccproxy_log_file_env_var_override(tmp_path, monkeypatch):
    """AC: CCPROXY_LOG_FILE env var directs output to the specified path."""
    custom_log = tmp_path / "proxy-test.jsonl"
    monkeypatch.setenv("CCPROXY_LOG_FILE", str(custom_log))

    registry = _make_passthrough_registry()
    client = _make_passthrough_client(_make_anthropic_response())

    with TestClient(app) as tc:
        # Set state AFTER lifespan runs so our values are not overwritten
        app.state.http_client = client
        app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
        app.state.profile_registry = registry
        app.state.config_from_file = True
        # Logger with no capture and no log_path — reads CCPROXY_LOG_FILE from env
        app.state.request_logger = RequestLogger()

        tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    assert custom_log.exists(), "Log file must be created at CCPROXY_LOG_FILE path"
    lines = [ln for ln in custom_log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, f"Expected 1 log line, got {len(lines)}"
    record = json.loads(lines[0])
    missing = _REQUIRED_FIELDS - set(record.keys())
    assert not missing, f"Missing required fields in file-written record: {missing}"


def test_ccproxy_log_file_directory_created_automatically(tmp_path, monkeypatch):
    """AC: directory for log file is created automatically if absent."""
    nested_log = tmp_path / "deep" / "nested" / "dir" / "requests.jsonl"
    monkeypatch.setenv("CCPROXY_LOG_FILE", str(nested_log))

    registry = _make_passthrough_registry()
    client = _make_passthrough_client(_make_anthropic_response())

    with TestClient(app) as tc:
        # Set state AFTER lifespan runs so our values are not overwritten
        app.state.http_client = client
        app.state.settings = Settings(upstream_base_url=STUB_ANTHROPIC)
        app.state.profile_registry = registry
        app.state.config_from_file = True
        app.state.request_logger = RequestLogger()

        tc.post(
            "/v1/messages",
            content=_make_request_body(),
            headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
        )

    assert nested_log.exists(), "Log file must be created even when parent dirs don't exist"


# ---------------------------------------------------------------------------
# AC: exactly one JSONL record per request
# ---------------------------------------------------------------------------


def test_exactly_one_record_per_request():
    """AC: each request produces exactly one JSONL record."""
    capture = io.StringIO()
    registry = _make_passthrough_registry()
    client = _make_passthrough_client(_make_anthropic_response())

    with TestClient(app) as tc:
        _setup_passthrough(client, registry, capture)
        for _ in range(3):
            tc.post(
                "/v1/messages",
                content=_make_request_body(),
                headers={"content-type": "application/json", "x-ccproxy-profile": "anthropic"},
            )

    capture.seek(0)
    lines = [ln for ln in capture.readlines() if ln.strip()]
    assert len(lines) == 3, f"Expected 3 records for 3 requests, got {len(lines)}"
