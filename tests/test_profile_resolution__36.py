"""Tests for issue #36: per-request profile resolution on /v1/messages endpoints."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from config import Settings
from main import app
from profiles import (
    ProfileConfig,
    ProfileRegistry,
    ProxyConfig,
    resolve_profile_name,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANTHROPIC_REQUEST = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}],
}).encode()

ANTHROPIC_RESPONSE_A = json.dumps({
    "id": "msg_a",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "From A"}],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
}).encode()

ANTHROPIC_RESPONSE_B = json.dumps({
    "id": "msg_b",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "From B"}],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
}).encode()

OPENAI_RESPONSE = json.dumps({
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}).encode()

STUB_A = "http://stub-a.test"
STUB_B = "http://stub-b.test"
STUB_C = "http://stub-c.test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*profiles_spec: tuple) -> ProfileRegistry:
    """Build a ProfileRegistry from (name, kind, upstream[, api_key_env, model]) tuples."""
    built: dict[str, ProfileConfig] = {}
    for spec in profiles_spec:
        name, kind, upstream = spec[0], spec[1], spec[2]
        api_key_env = spec[3] if len(spec) > 3 else None
        model = spec[4] if len(spec) > 4 else None
        built[name] = ProfileConfig(kind=kind, upstream=upstream, api_key_env=api_key_env, model=model)
    return ProfileRegistry(ProxyConfig(profiles=built))


def _make_dispatch_client(routes: dict[str, bytes]) -> tuple[object, list]:
    """Mock client routing POST calls to different responses by URL prefix."""
    calls: list[str] = []
    lock = threading.Lock()

    async def _post(url, *, content, headers, **kwargs):
        with lock:
            calls.append(url)
        for prefix, body in routes.items():
            if url.startswith(prefix):
                mock_resp = MagicMock(spec=httpx.Response)
                mock_resp.status_code = 200
                mock_resp.content = body
                mock_resp.headers = {"content-type": "application/json"}
                return mock_resp
        raise AssertionError(f"No route configured for {url!r}")

    mock = MagicMock()
    mock.post = _post
    mock.aclose = AsyncMock()
    return mock, calls


def _setup(mock_client, registry: ProfileRegistry, *, upstream: str = STUB_A) -> None:
    """Install mock client, settings, registry, and enable registry-based routing."""
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=upstream)
    app.state.profile_registry = registry
    app.state.config_from_file = True  # enable registry routing path


def _setup_legacy(mock_client, *, upstream: str = STUB_A) -> None:
    """Install mock client and settings without a registry (legacy mode)."""
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=upstream)
    app.state.config_from_file = False


# ---------------------------------------------------------------------------
# Unit tests: resolve_profile_name (4-level precedence chain)
# ---------------------------------------------------------------------------

def test_resolve_level1_header_wins(monkeypatch):
    """AC 1: X-CCProxy-Profile header wins over all other selectors."""
    monkeypatch.setenv("CCPROXY_PROFILE", "from-env")
    result = resolve_profile_name(
        header="from-header",
        query_param="from-query",
        state_json_path=Path("/nonexistent"),
    )
    assert result == "from-header"


def test_resolve_level2_query_param_over_env(monkeypatch):
    """AC 2: query param wins when no header, even with env var set."""
    monkeypatch.setenv("CCPROXY_PROFILE", "from-env")
    result = resolve_profile_name(
        header=None,
        query_param="from-query",
        state_json_path=Path("/nonexistent"),
    )
    assert result == "from-query"


def test_resolve_level3_env_var(monkeypatch):
    """AC 3: CCPROXY_PROFILE env var used when no header or query param."""
    monkeypatch.setenv("CCPROXY_PROFILE", "from-env")
    result = resolve_profile_name(
        header=None,
        query_param=None,
        state_json_path=Path("/nonexistent"),
    )
    assert result == "from-env"


def test_resolve_level4_state_json(monkeypatch, tmp_path):
    """AC 4: state.json active_profile used when levels 1-3 are absent."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"active_profile": "from-state"}))
    result = resolve_profile_name(header=None, query_param=None, state_json_path=state)
    assert result == "from-state"


def test_resolve_level5_builtin_default(monkeypatch):
    """AC 5: built-in 'anthropic' default used when nothing else is set."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    result = resolve_profile_name(
        header=None,
        query_param=None,
        state_json_path=Path("/nonexistent"),
    )
    assert result == "anthropic"


def test_resolve_state_json_missing_falls_through(monkeypatch):
    """AC 5: missing state.json silently falls through to built-in default."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    result = resolve_profile_name(
        header=None,
        query_param=None,
        state_json_path=Path("/definitely/does/not/exist.json"),
    )
    assert result == "anthropic"


def test_resolve_state_json_no_active_profile_key(monkeypatch, tmp_path):
    """AC 5: state.json without 'active_profile' key falls through to default."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"other_key": "ignored"}))
    result = resolve_profile_name(header=None, query_param=None, state_json_path=state)
    assert result == "anthropic"


def test_resolve_state_json_malformed_falls_through(monkeypatch, tmp_path):
    """AC 5: malformed state.json silently falls through to default."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    state = tmp_path / "state.json"
    state.write_text("not valid json {{{")
    result = resolve_profile_name(header=None, query_param=None, state_json_path=state)
    assert result == "anthropic"


# ---------------------------------------------------------------------------
# Integration: header resolves named profile (AC 1, 7)
# ---------------------------------------------------------------------------

def test_header_routes_to_profile_upstream(monkeypatch):
    """AC 1, 7: X-CCProxy-Profile header routes to that profile's upstream."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    registry = _make_registry(("profile-b", "passthrough", STUB_B))
    mock_client, calls = _make_dispatch_client({STUB_B: ANTHROPIC_RESPONSE_B})

    with TestClient(app) as tc:
        _setup(mock_client, registry, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST,
            headers={"content-type": "application/json", "X-CCProxy-Profile": "profile-b"},
        )

    assert resp.status_code == 200
    assert any(c.startswith(STUB_B) for c in calls), f"Expected call to {STUB_B}, got {calls}"
    assert not any(c.startswith(STUB_A) for c in calls)


# ---------------------------------------------------------------------------
# Integration: query param resolves named profile (AC 2)
# ---------------------------------------------------------------------------

def test_query_param_routes_to_profile_upstream(monkeypatch):
    """AC 2: ?profile=<name> routes to that profile's upstream."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    registry = _make_registry(("profile-b", "passthrough", STUB_B))
    mock_client, calls = _make_dispatch_client({STUB_B: ANTHROPIC_RESPONSE_B})

    with TestClient(app) as tc:
        _setup(mock_client, registry, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages?profile=profile-b",
            content=ANTHROPIC_REQUEST,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert any(c.startswith(STUB_B) for c in calls)


# ---------------------------------------------------------------------------
# Integration: header beats query param (AC 7)
# ---------------------------------------------------------------------------

def test_header_wins_over_query_param(monkeypatch):
    """AC 7: when header and query param both present, header wins."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    registry = _make_registry(
        ("profile-a", "passthrough", STUB_A),
        ("profile-b", "passthrough", STUB_B),
    )
    mock_client, calls = _make_dispatch_client({
        STUB_A: ANTHROPIC_RESPONSE_A,
        STUB_B: ANTHROPIC_RESPONSE_B,
    })

    with TestClient(app) as tc:
        _setup(mock_client, registry, upstream=STUB_C)
        resp = tc.post(
            "/v1/messages?profile=profile-a",    # query → profile-a
            content=ANTHROPIC_REQUEST,
            headers={
                "content-type": "application/json",
                "X-CCProxy-Profile": "profile-b",  # header → profile-b (wins)
            },
        )

    assert resp.status_code == 200
    assert any(c.startswith(STUB_B) for c in calls), "Header profile-b should win over query profile-a"
    assert not any(c.startswith(STUB_A) for c in calls)


# ---------------------------------------------------------------------------
# Integration: env var resolves profile (AC 3)
# ---------------------------------------------------------------------------

def test_env_var_routes_to_profile_upstream(monkeypatch):
    """AC 3: CCPROXY_PROFILE env var routes to that profile's upstream."""
    monkeypatch.setenv("CCPROXY_PROFILE", "env-profile")
    registry = _make_registry(("env-profile", "passthrough", STUB_B))
    mock_client, calls = _make_dispatch_client({STUB_B: ANTHROPIC_RESPONSE_B})

    with TestClient(app) as tc:
        _setup(mock_client, registry, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert any(c.startswith(STUB_B) for c in calls)


# ---------------------------------------------------------------------------
# Integration: state.json active default (AC 4)
# ---------------------------------------------------------------------------

def test_state_json_default_routes_to_profile_upstream(monkeypatch):
    """AC 4: state.json active_profile is used when levels 1-3 are absent."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    registry = _make_registry(("state-profile", "passthrough", STUB_B))
    mock_client, calls = _make_dispatch_client({STUB_B: ANTHROPIC_RESPONSE_B})

    monkeypatch.setattr("profiles._state_json_path", Path("/nonexistent"))  # will be overridden below

    # Write state.json with active_profile
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"active_profile": "state-profile"}, f)
        tmp_name = f.name

    try:
        monkeypatch.setattr("profiles._state_json_path", Path(tmp_name))
        with TestClient(app) as tc:
            _setup(mock_client, registry, upstream=STUB_A)
            resp = tc.post(
                "/v1/messages",
                content=ANTHROPIC_REQUEST,
                headers={"content-type": "application/json"},
            )
    finally:
        os.unlink(tmp_name)

    assert resp.status_code == 200
    assert any(c.startswith(STUB_B) for c in calls)


# ---------------------------------------------------------------------------
# Integration: profile kind determines routing (AC 6)
# ---------------------------------------------------------------------------

def test_profile_kind_openai_translates_request(monkeypatch):
    """AC 6: profile with kind='openai' translates the request to OpenAI wire format."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    monkeypatch.setenv("STUB_B_KEY", "test-api-key")

    captured_url = []

    async def _post(url, *, content, headers, **kwargs):
        captured_url.append(url)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = OPENAI_RESPONSE
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock = MagicMock()
    mock.post = _post
    mock.aclose = AsyncMock()

    registry = _make_registry(("my-openai", "openai", STUB_B, "STUB_B_KEY", "gpt-4o"))

    with TestClient(app) as tc:
        _setup(mock, registry, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST,
            headers={"content-type": "application/json", "X-CCProxy-Profile": "my-openai"},
        )

    assert resp.status_code == 200
    # Must have called the OpenAI chat/completions endpoint, not Anthropic
    assert any(STUB_B in u and "chat/completions" in u for u in captured_url), \
        f"Expected OpenAI endpoint at {STUB_B}, got: {captured_url}"
    # Response must be translated back to Anthropic format
    body = resp.json()
    assert body["role"] == "assistant"
    assert body["content"][0]["type"] == "text"


def test_profile_kind_openai_uses_profile_api_key(monkeypatch):
    """AC 7: profile's api_key (from api_key_env) is used for the outbound call."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    monkeypatch.setenv("STUB_B_KEY", "my-secret-key-xyz")

    captured_headers: list[dict] = []

    async def _post(url, *, content, headers, **kwargs):
        captured_headers.append({k.lower(): v for k, v in headers.items()})
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = OPENAI_RESPONSE
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock = MagicMock()
    mock.post = _post
    mock.aclose = AsyncMock()

    registry = _make_registry(("my-openai", "openai", STUB_B, "STUB_B_KEY", "gpt-4o"))

    with TestClient(app) as tc:
        _setup(mock, registry, upstream=STUB_A)
        tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST,
            headers={"content-type": "application/json", "X-CCProxy-Profile": "my-openai"},
        )

    assert len(captured_headers) == 1
    assert captured_headers[0].get("authorization") == "Bearer my-secret-key-xyz"


def test_profile_kind_openai_uses_profile_model(monkeypatch):
    """AC 7: profile's model is used for the outbound call."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    monkeypatch.setenv("STUB_B_KEY", "key")

    captured_body: list[bytes] = []

    async def _post(url, *, content, headers, **kwargs):
        captured_body.append(content)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = OPENAI_RESPONSE
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock = MagicMock()
    mock.post = _post
    mock.aclose = AsyncMock()

    registry = _make_registry(("my-openai", "openai", STUB_B, "STUB_B_KEY", "gpt-4o-mini"))

    with TestClient(app) as tc:
        _setup(mock, registry, upstream=STUB_A)
        tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST,
            headers={"content-type": "application/json", "X-CCProxy-Profile": "my-openai"},
        )

    assert len(captured_body) == 1
    sent = json.loads(captured_body[0])
    assert sent["model"] == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# AC 8, 9: Concurrent isolation — no cross-contamination
# ---------------------------------------------------------------------------

def test_concurrent_requests_no_cross_contamination(monkeypatch):
    """AC 8, 9: concurrent requests with different profiles route independently,
    with no cross-contamination and no mutable global state written in the hot path."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)

    calls: list[str] = []
    responses: dict[str, bytes] = {
        STUB_A: ANTHROPIC_RESPONSE_A,
        STUB_B: ANTHROPIC_RESPONSE_B,
    }
    lock = threading.Lock()

    async def _post(url, *, content, headers, **kwargs):
        with lock:
            calls.append(url)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        for prefix, body in responses.items():
            if url.startswith(prefix):
                mock_resp.content = body
                break
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock = MagicMock()
    mock.post = _post
    mock.aclose = AsyncMock()

    registry = _make_registry(
        ("profile-a", "passthrough", STUB_A),
        ("profile-b", "passthrough", STUB_B),
    )

    with TestClient(app) as tc:
        _setup(mock, registry)

        def req_a():
            return tc.post(
                "/v1/messages",
                content=ANTHROPIC_REQUEST,
                headers={"content-type": "application/json", "X-CCProxy-Profile": "profile-a"},
            )

        def req_b():
            return tc.post(
                "/v1/messages",
                content=ANTHROPIC_REQUEST,
                headers={"content-type": "application/json", "X-CCProxy-Profile": "profile-b"},
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(req_a)
            fut_b = pool.submit(req_b)
            resp_a = fut_a.result(timeout=10)
            resp_b = fut_b.result(timeout=10)

    assert resp_a.status_code == 200
    assert resp_b.status_code == 200

    # Each upstream called exactly once
    assert sum(1 for c in calls if c.startswith(STUB_A)) == 1, f"stub-a calls: {calls}"
    assert sum(1 for c in calls if c.startswith(STUB_B)) == 1, f"stub-b calls: {calls}"

    # No cross-contamination: each response carries its own content
    data_a = resp_a.json()
    data_b = resp_b.json()
    assert data_a["id"] == "msg_a", "Response A must come from stub-a"
    assert data_b["id"] == "msg_b", "Response B must come from stub-b"


# ---------------------------------------------------------------------------
# AC 11: No regression for requests without any selector
# ---------------------------------------------------------------------------

def test_no_regression_without_selector_passthrough(monkeypatch):
    """AC 11: requests without any selector continue to behave identically (passthrough)."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)

    mock_client, calls = _make_dispatch_client({STUB_A: ANTHROPIC_RESPONSE_A})
    # Use legacy mode (no registry routing) — mimics existing behaviour
    with TestClient(app) as tc:
        _setup_legacy(mock_client, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages",
            content=ANTHROPIC_REQUEST,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    assert resp.content == ANTHROPIC_RESPONSE_A
    assert calls == [f"{STUB_A}/v1/messages"]


# ---------------------------------------------------------------------------
# UAT 6: count_tokens respects per-request profile selector
# ---------------------------------------------------------------------------

def test_count_tokens_query_param_selects_openai_profile(monkeypatch):
    """UAT 6: /v1/messages/count_tokens with ?profile=openai uses the openai profile."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    registry = _make_registry(("openai-profile", "openai", STUB_B, None, "gpt-4o"))

    mock = MagicMock()
    mock.aclose = AsyncMock()

    body = json.dumps({"messages": [{"role": "user", "content": "Hello world"}]}).encode()

    with TestClient(app) as tc:
        _setup(mock, registry, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages/count_tokens?profile=openai-profile",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "input_tokens" in data
    assert isinstance(data["input_tokens"], int)
    assert data["input_tokens"] > 0


def test_count_tokens_header_selects_openai_profile(monkeypatch):
    """AC 1 on count_tokens: X-CCProxy-Profile header is respected on /count_tokens too."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    registry = _make_registry(("openai-profile", "openai", STUB_B, None, "gpt-4o"))

    mock = MagicMock()
    mock.aclose = AsyncMock()

    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    with TestClient(app) as tc:
        _setup(mock, registry, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json", "X-CCProxy-Profile": "openai-profile"},
        )

    assert resp.status_code == 200
    assert "input_tokens" in resp.json()


def test_count_tokens_passthrough_profile_uses_upstream(monkeypatch):
    """AC 7 on count_tokens: passthrough profile uses profile's upstream for count_tokens."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)

    captured_url: list[str] = []

    async def _post(url, *, content, headers, **kwargs):
        captured_url.append(url)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = json.dumps({"input_tokens": 42}).encode()
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    mock = MagicMock()
    mock.post = _post
    mock.aclose = AsyncMock()

    registry = _make_registry(("my-anthropic", "passthrough", STUB_B))

    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()

    with TestClient(app) as tc:
        _setup(mock, registry, upstream=STUB_A)
        resp = tc.post(
            "/v1/messages/count_tokens",
            content=body,
            headers={"content-type": "application/json", "X-CCProxy-Profile": "my-anthropic"},
        )

    assert resp.status_code == 200
    assert any(c.startswith(STUB_B) for c in captured_url), \
        f"Expected call to {STUB_B}, got {captured_url}"
