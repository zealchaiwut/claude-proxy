"""Tests for issue #38: per-dispatch profile selection and model_map routing.

AC coverage:
- header-wins-over-env: X-CCProxy-Profile header overrides CCPROXY_PROFILE env var
- env-wins-over-default: CCPROXY_PROFILE env wins over global state default
- default-fallback: no header, no env → global state default (state.json / 'anthropic')
- concurrent-isolation: two threads with different profiles route to different backends
- model_map rewrite: client model name rewritten via model_map before upstream forward
"""
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
from profiles import ProfileConfig, ProfileRegistry, ProxyConfig, resolve_profile_name


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

STUB_LOCAL = "http://stub-local.test"
STUB_ANTHROPIC = "http://stub-anthropic.test"

HAIKU_CLIENT_MODEL = "claude-haiku-4-5-20251001"
REWRITTEN_MODEL = "gpt-4o-mini"

def _make_body(model: str = HAIKU_CLIENT_MODEL) -> bytes:
    return json.dumps({
        "model": model,
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }).encode()


def _anthr_response(id_: str, model: str = HAIKU_CLIENT_MODEL) -> bytes:
    return json.dumps({
        "id": id_,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": f"response-{id_}"}],
        "model": model,
        "stop_reason": "end_turn",
    }).encode()


def _make_registry(**profiles: ProfileConfig) -> ProfileRegistry:
    return ProfileRegistry(ProxyConfig(profiles=profiles))


def _make_mock_client(routes: dict[str, bytes]) -> tuple[object, list]:
    """Mock HTTP client that routes by URL prefix; records call URLs."""
    calls: list[str] = []
    lock = threading.Lock()

    async def _post(url, *, content, headers, **kwargs):
        with lock:
            calls.append(url)
        for prefix, body in routes.items():
            if url.startswith(prefix):
                resp = MagicMock(spec=httpx.Response)
                resp.status_code = 200
                resp.content = body
                resp.headers = {"content-type": "application/json"}
                return resp
        raise AssertionError(f"Unmapped URL: {url!r}")

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()
    return client, calls


def _setup(client, registry: ProfileRegistry, upstream: str = STUB_LOCAL) -> None:
    app.state.http_client = client
    app.state.settings = Settings(upstream_base_url=upstream)
    app.state.profile_registry = registry
    app.state.config_from_file = True


# ---------------------------------------------------------------------------
# Unit tests: resolve_profile_name precedence (AC 1, 2, 3)
# ---------------------------------------------------------------------------

def test_header_wins_over_env(monkeypatch):
    """AC: X-CCProxy-Profile header overrides CCPROXY_PROFILE env var."""
    monkeypatch.setenv("CCPROXY_PROFILE", "env-profile")
    result = resolve_profile_name(
        header="header-profile",
        query_param=None,
        state_json_path=Path("/nonexistent"),
    )
    assert result == "header-profile"


def test_env_wins_over_default(monkeypatch):
    """AC: CCPROXY_PROFILE env var wins over global state default."""
    monkeypatch.setenv("CCPROXY_PROFILE", "env-profile")
    result = resolve_profile_name(
        header=None,
        query_param=None,
        state_json_path=Path("/nonexistent"),
    )
    assert result == "env-profile"


def test_default_fallback_to_state_json(monkeypatch, tmp_path):
    """AC: no header, no env → falls through to state.json active_profile."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"active_profile": "state-default"}))
    result = resolve_profile_name(header=None, query_param=None, state_json_path=state)
    assert result == "state-default"


def test_default_fallback_to_anthropic(monkeypatch):
    """AC: no header, no env, no state.json → built-in 'anthropic' default."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    result = resolve_profile_name(
        header=None,
        query_param=None,
        state_json_path=Path("/nonexistent"),
    )
    assert result == "anthropic"


# ---------------------------------------------------------------------------
# Integration: header overrides env var in actual request routing (AC 1)
# ---------------------------------------------------------------------------

def test_header_overrides_env_in_routing(monkeypatch):
    """AC: X-CCProxy-Profile header selects correct upstream even when CCPROXY_PROFILE is set."""
    monkeypatch.setenv("CCPROXY_PROFILE", "local")

    registry = _make_registry(
        local=ProfileConfig(kind="passthrough", upstream=STUB_LOCAL),
        anthropic=ProfileConfig(kind="passthrough", upstream=STUB_ANTHROPIC),
    )
    client, calls = _make_mock_client({
        STUB_LOCAL: _anthr_response("local-msg"),
        STUB_ANTHROPIC: _anthr_response("anthr-msg"),
    })

    with TestClient(app) as tc:
        _setup(client, registry)
        resp = tc.post(
            "/v1/messages",
            content=_make_body(),
            headers={"content-type": "application/json", "X-CCProxy-Profile": "anthropic"},
        )

    assert resp.status_code == 200
    assert resp.json()["id"] == "anthr-msg", "Header 'anthropic' should win over env 'local'"
    assert any(c.startswith(STUB_ANTHROPIC) for c in calls)
    assert not any(c.startswith(STUB_LOCAL) for c in calls)


# ---------------------------------------------------------------------------
# Integration: env var routes requests when no header present (AC 2)
# ---------------------------------------------------------------------------

def test_env_var_routes_without_header(monkeypatch):
    """AC: CCPROXY_PROFILE env var routes request when no header is set."""
    monkeypatch.setenv("CCPROXY_PROFILE", "local")

    registry = _make_registry(
        local=ProfileConfig(kind="passthrough", upstream=STUB_LOCAL),
        anthropic=ProfileConfig(kind="passthrough", upstream=STUB_ANTHROPIC),
    )
    client, calls = _make_mock_client({
        STUB_LOCAL: _anthr_response("local-msg"),
        STUB_ANTHROPIC: _anthr_response("anthr-msg"),
    })

    with TestClient(app) as tc:
        _setup(client, registry, upstream=STUB_ANTHROPIC)
        resp = tc.post(
            "/v1/messages",
            content=_make_body(),
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    # env var says 'local' → should hit STUB_LOCAL, not STUB_ANTHROPIC
    assert any(c.startswith(STUB_LOCAL) for c in calls)
    assert not any(c.startswith(STUB_ANTHROPIC) for c in calls)


# ---------------------------------------------------------------------------
# Concurrent isolation: two threads, different profiles, no cross-contamination (AC 5)
# ---------------------------------------------------------------------------

def test_concurrent_profile_isolation(monkeypatch):
    """AC: concurrent requests with distinct X-CCProxy-Profile headers route independently."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)

    registry = _make_registry(
        local=ProfileConfig(kind="passthrough", upstream=STUB_LOCAL),
        anthropic=ProfileConfig(kind="passthrough", upstream=STUB_ANTHROPIC),
    )
    client, calls = _make_mock_client({
        STUB_LOCAL: _anthr_response("local-msg"),
        STUB_ANTHROPIC: _anthr_response("anthr-msg"),
    })

    N = 5  # requests per profile

    with TestClient(app) as tc:
        _setup(client, registry)

        def send_local():
            return tc.post(
                "/v1/messages",
                content=_make_body(),
                headers={"content-type": "application/json", "X-CCProxy-Profile": "local"},
            )

        def send_anthropic():
            return tc.post(
                "/v1/messages",
                content=_make_body(),
                headers={"content-type": "application/json", "X-CCProxy-Profile": "anthropic"},
            )

        with ThreadPoolExecutor(max_workers=N * 2) as pool:
            local_futs = [pool.submit(send_local) for _ in range(N)]
            anthr_futs = [pool.submit(send_anthropic) for _ in range(N)]
            local_resps = [f.result(timeout=10) for f in local_futs]
            anthr_resps = [f.result(timeout=10) for f in anthr_futs]

    for r in local_resps:
        assert r.status_code == 200
        assert r.json()["id"] == "local-msg", "Local profile must route to STUB_LOCAL"

    for r in anthr_resps:
        assert r.status_code == 200
        assert r.json()["id"] == "anthr-msg", "Anthropic profile must route to STUB_ANTHROPIC"

    local_calls = sum(1 for c in calls if c.startswith(STUB_LOCAL))
    anthr_calls = sum(1 for c in calls if c.startswith(STUB_ANTHROPIC))
    assert local_calls == N, f"Expected {N} local calls, got {local_calls}"
    assert anthr_calls == N, f"Expected {N} anthropic calls, got {anthr_calls}"


# ---------------------------------------------------------------------------
# model_map rewrite: forwarded body contains rewritten model name (AC 4)
# ---------------------------------------------------------------------------

def test_model_map_rewrite_passthrough(monkeypatch):
    """AC: model_map rewrites client model name in the forwarded request body."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)

    registry = _make_registry(
        local=ProfileConfig(
            kind="passthrough",
            upstream=STUB_LOCAL,
            model_map={HAIKU_CLIENT_MODEL: REWRITTEN_MODEL},
        ),
    )

    forwarded_bodies: list[bytes] = []

    async def _post(url, *, content, headers, **kwargs):
        forwarded_bodies.append(content)
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = _anthr_response("msg1", model=REWRITTEN_MODEL)
        resp.headers = {"content-type": "application/json"}
        return resp

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()

    with TestClient(app) as tc:
        _setup(client, registry)
        resp = tc.post(
            "/v1/messages",
            content=_make_body(HAIKU_CLIENT_MODEL),
            headers={"content-type": "application/json", "X-CCProxy-Profile": "local"},
        )

    assert resp.status_code == 200
    assert len(forwarded_bodies) == 1, "Exactly one upstream call expected"
    forwarded = json.loads(forwarded_bodies[0])
    assert forwarded["model"] == REWRITTEN_MODEL, (
        f"model_map should rewrite '{HAIKU_CLIENT_MODEL}' → '{REWRITTEN_MODEL}', "
        f"but forwarded model was '{forwarded['model']}'"
    )


def test_model_map_passthrough_unknown_model_unchanged(monkeypatch):
    """AC: model not in model_map is forwarded unchanged."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)

    registry = _make_registry(
        local=ProfileConfig(
            kind="passthrough",
            upstream=STUB_LOCAL,
            model_map={"some-other-model": REWRITTEN_MODEL},
        ),
    )

    forwarded_bodies: list[bytes] = []

    async def _post(url, *, content, headers, **kwargs):
        forwarded_bodies.append(content)
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = _anthr_response("msg1")
        resp.headers = {"content-type": "application/json"}
        return resp

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()

    with TestClient(app) as tc:
        _setup(client, registry)
        resp = tc.post(
            "/v1/messages",
            content=_make_body(HAIKU_CLIENT_MODEL),
            headers={"content-type": "application/json", "X-CCProxy-Profile": "local"},
        )

    assert resp.status_code == 200
    forwarded = json.loads(forwarded_bodies[0])
    assert forwarded["model"] == HAIKU_CLIENT_MODEL, "Unknown model should pass through unchanged"


def test_model_map_rewrite_openai_profile(monkeypatch):
    """AC: model_map is applied for openai-kind profiles — client model mapped to upstream model."""
    monkeypatch.delenv("CCPROXY_PROFILE", raising=False)
    monkeypatch.setenv("OPENAI_KEY_TEST", "test-key")

    OPENAI_RESP = json.dumps({
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": REWRITTEN_MODEL,
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }).encode()

    registry = _make_registry(
        local=ProfileConfig(
            kind="openai",
            upstream=STUB_LOCAL,
            api_key_env="OPENAI_KEY_TEST",
            model="gpt-4o",
            model_map={HAIKU_CLIENT_MODEL: REWRITTEN_MODEL},
        ),
    )

    forwarded_bodies: list[bytes] = []

    async def _post(url, *, content, headers, **kwargs):
        forwarded_bodies.append(content)
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.content = OPENAI_RESP
        resp.headers = {"content-type": "application/json"}
        return resp

    client = MagicMock()
    client.post = _post
    client.aclose = AsyncMock()

    with TestClient(app) as tc:
        _setup(client, registry)
        resp = tc.post(
            "/v1/messages",
            content=_make_body(HAIKU_CLIENT_MODEL),
            headers={"content-type": "application/json", "X-CCProxy-Profile": "local"},
        )

    assert resp.status_code == 200
    assert len(forwarded_bodies) == 1
    forwarded = json.loads(forwarded_bodies[0])
    assert forwarded["model"] == REWRITTEN_MODEL, (
        f"model_map should rewrite '{HAIKU_CLIENT_MODEL}' → '{REWRITTEN_MODEL}' for openai profile"
    )
