"""Tests for issue #60: ccproxy replay command.

AC coverage:
- ac-cli-invocation: ccproxy replay exits 0 on success
- ac-profile-resolution: routes through existing profile resolution code path
- ac-two-profiles-two-artifacts: replaying against two profiles writes two artifact files
- ac-artifact-fields: artifact contains response, prompt_tokens, completion_tokens, est_cost, latency_ms
- ac-stream-flag: --stream issues a streaming request
- ac-no-stream-flag: --no-stream forces non-streaming
- ac-missing-file: missing capture file exits non-zero with descriptive error
- ac-missing-profile: unknown profile exits non-zero with descriptive error
"""
import json
import textwrap
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

TOML_TWO_PROFILES = textwrap.dedent("""\
    [profiles.alpha]
    kind = "passthrough"
    upstream = "http://alpha.test"
    pricing.input_per_mtok = 3.0
    pricing.output_per_mtok = 15.0

    [profiles.beta]
    kind = "passthrough"
    upstream = "http://beta.test"
    pricing.input_per_mtok = 1.0
    pricing.output_per_mtok = 5.0
""")

TOML_OPENAI_PROFILE = textwrap.dedent("""\
    [profiles.oai]
    kind = "openai"
    upstream = "http://oai.test"
    model = "gpt-4o"
    pricing.input_per_mtok = 2.5
    pricing.output_per_mtok = 10.0
""")

SAMPLE_REQUEST = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hello"}],
}

SAMPLE_CAPTURE = {
    "version": 1,
    "captured_at": "2025-01-01T00:00:00Z",
    "profile": "alpha",
    "stream": False,
    "request": SAMPLE_REQUEST,
}

ALPHA_RESPONSE = {
    "id": "msg_alpha_001",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from alpha!"}],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 10},
}

BETA_RESPONSE = {
    "id": "msg_beta_001",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from beta!"}],
    "model": "claude-3-haiku-20240307",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 8},
}


def _make_mock_http_client(response_body: bytes, status: int = 200):
    """Build a mock httpx.AsyncClient for non-streaming."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status
    mock_resp.content = response_body
    mock_resp.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


# ---------------------------------------------------------------------------
# AC: Two profiles produce two separate artifact files with required fields
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_profiles_two_artifacts(tmp_path):
    """ac-two-profiles-two-artifacts: replaying against alpha and beta writes separate artifacts."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    alpha_resp = json.dumps(ALPHA_RESPONSE).encode()
    beta_resp = json.dumps(BETA_RESPONSE).encode()

    call_index = [0]
    responses = [alpha_resp, beta_resp]

    async def _mock_post(url, *, content, headers, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = responses[call_index[0]]
        mock_resp.headers = {"content-type": "application/json"}
        call_index[0] += 1
        return mock_resp

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        out = StringIO()
        rc_alpha = await replay(
            capture_file, "alpha", config_path=config_path, stdout=out
        )
        assert rc_alpha == 0

        rc_beta = await replay(
            capture_file, "beta", config_path=config_path, stdout=out
        )
        assert rc_beta == 0

    alpha_artifact = tmp_path / "request_001.replay-alpha.json"
    beta_artifact = tmp_path / "request_001.replay-beta.json"
    assert alpha_artifact.exists(), "alpha artifact not written"
    assert beta_artifact.exists(), "beta artifact not written"


@pytest.mark.asyncio
async def test_artifact_contains_required_fields(tmp_path):
    """ac-artifact-fields: artifact has response, prompt_tokens, completion_tokens, est_cost, latency_ms."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    alpha_resp = json.dumps(ALPHA_RESPONSE).encode()

    async def _mock_post(url, *, content, headers, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = alpha_resp
        mock_resp.headers = {"content-type": "application/json"}
        return mock_resp

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        rc = await replay(capture_file, "alpha", config_path=config_path, stdout=StringIO())
        assert rc == 0

    artifact = tmp_path / "request_001.replay-alpha.json"
    data = json.loads(artifact.read_text())

    assert "response" in data, "artifact missing 'response'"
    assert "prompt_tokens" in data, "artifact missing 'prompt_tokens'"
    assert "completion_tokens" in data, "artifact missing 'completion_tokens'"
    assert "est_cost" in data, "artifact missing 'est_cost'"
    assert "latency_ms" in data, "artifact missing 'latency_ms'"

    assert data["prompt_tokens"] >= 0
    assert data["completion_tokens"] >= 0
    assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_artifact_tokens_match_upstream_usage(tmp_path):
    """ac-artifact-fields: prompt_tokens and completion_tokens come from upstream usage."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    alpha_resp = json.dumps(ALPHA_RESPONSE).encode()

    async def _mock_post(url, *, content, headers, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = alpha_resp
        return mock_resp

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        rc = await replay(capture_file, "alpha", config_path=config_path, stdout=StringIO())

    artifact = json.loads((tmp_path / "request_001.replay-alpha.json").read_text())
    # ALPHA_RESPONSE has usage: input=5, output=10
    assert artifact["prompt_tokens"] == 5
    assert artifact["completion_tokens"] == 10


@pytest.mark.asyncio
async def test_artifact_est_cost_is_float_or_none(tmp_path):
    """ac-artifact-fields: est_cost is a float (USD) when pricing is configured."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    alpha_resp = json.dumps(ALPHA_RESPONSE).encode()

    async def _mock_post(url, *, content, headers, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = alpha_resp
        return mock_resp

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        await replay(capture_file, "alpha", config_path=config_path, stdout=StringIO())

    artifact = json.loads((tmp_path / "request_001.replay-alpha.json").read_text())
    assert isinstance(artifact["est_cost"], float)
    # 5 input * 3.0/1M + 10 output * 15.0/1M = 0.000015 + 0.00015 = 0.000165
    assert artifact["est_cost"] == pytest.approx(0.000165, rel=1e-4)


# ---------------------------------------------------------------------------
# AC: --stream / --no-stream flag behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_stream_flag_forces_non_streaming(tmp_path):
    """ac-no-stream-flag: --no-stream forces non-streaming even if capture has stream=True."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture = {**SAMPLE_CAPTURE, "stream": True}
    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(capture))

    posted_body = {}

    async def _mock_post(url, *, content, headers, **kwargs):
        posted_body["data"] = json.loads(content)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(ALPHA_RESPONSE).encode()
        return mock_resp

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        rc = await replay(
            capture_file, "alpha",
            stream_override=False,
            config_path=config_path,
            stdout=StringIO(),
        )
        assert rc == 0

    assert posted_body["data"].get("stream") is False


@pytest.mark.asyncio
async def test_stream_flag_forces_streaming(tmp_path):
    """ac-stream-flag: --stream causes a streaming request."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture = {**SAMPLE_CAPTURE, "stream": False}
    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(capture))

    ALPHA_SSE = (
        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":5}}}\n\n"
        b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
        b"event: message_delta\ndata: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":1}}\n\n"
        b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
    )

    streamed_body = {}

    async def _aiter_bytes():
        yield ALPHA_SSE

    class MockStreamResponse:
        status_code = 200

        async def aiter_bytes(self):
            yield ALPHA_SSE

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = MagicMock()

        stream_ctx = MockStreamResponse()

        def _stream_method(method, url, *, content, headers, **kwargs):
            streamed_body["content"] = json.loads(content)
            return stream_ctx

        mock_instance.stream = _stream_method
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        rc = await replay(
            capture_file, "alpha",
            stream_override=True,
            config_path=config_path,
            stdout=StringIO(),
        )
        assert rc == 0

    assert streamed_body["content"].get("stream") is True
    assert (tmp_path / "request_001.replay-alpha.json").exists()


# ---------------------------------------------------------------------------
# AC: Default inherits stream setting from capture
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_default_inherits_capture_stream_false(tmp_path):
    """ac-stream-default: when no override, stream setting comes from capture file."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture = {**SAMPLE_CAPTURE, "stream": False}
    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(capture))

    posted_body = {}

    async def _mock_post(url, *, content, headers, **kwargs):
        posted_body["data"] = json.loads(content)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(ALPHA_RESPONSE).encode()
        return mock_resp

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        rc = await replay(capture_file, "alpha", config_path=config_path, stdout=StringIO())
        assert rc == 0

    assert posted_body["data"]["stream"] is False


# ---------------------------------------------------------------------------
# AC: Response is printed to stdout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_response_printed_to_stdout(tmp_path):
    """ac-cli-invocation: response is printed to stdout during replay."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    async def _mock_post(url, *, content, headers, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = json.dumps(ALPHA_RESPONSE).encode()
        return mock_resp

    with patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        out = StringIO()
        rc = await replay(capture_file, "alpha", config_path=config_path, stdout=out)
        assert rc == 0

    out_text = out.getvalue()
    assert len(out_text) > 0, "nothing printed to stdout"


# ---------------------------------------------------------------------------
# AC: Missing capture file exits non-zero with descriptive error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_missing_capture_file_exits_nonzero(tmp_path, capsys):
    """ac-missing-file: non-existent capture file exits 1 with error message."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    rc = await replay(
        tmp_path / "does_not_exist.json",
        "alpha",
        config_path=config_path,
        stdout=StringIO(),
    )
    assert rc != 0

    captured = capsys.readouterr()
    assert "error" in captured.err.lower() or "error" in captured.out.lower()


@pytest.mark.asyncio
async def test_malformed_capture_file_exits_nonzero(tmp_path, capsys):
    """ac-missing-file: malformed capture file exits 1 with error message."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json {{{")

    rc = await replay(bad_file, "alpha", config_path=config_path, stdout=StringIO())
    assert rc != 0

    captured = capsys.readouterr()
    assert "error" in captured.err.lower()


@pytest.mark.asyncio
async def test_capture_missing_request_key_exits_nonzero(tmp_path, capsys):
    """ac-missing-file: capture without 'request' key is treated as malformed."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"version": 1}))

    rc = await replay(bad_file, "alpha", config_path=config_path, stdout=StringIO())
    assert rc != 0


# ---------------------------------------------------------------------------
# AC: Unknown profile exits non-zero with descriptive error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unknown_profile_exits_nonzero(tmp_path, capsys):
    """ac-missing-profile: unknown profile name exits 1 with error message."""
    from services.replay import replay

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    rc = await replay(
        capture_file, "nonexistent",
        config_path=config_path,
        stdout=StringIO(),
    )
    assert rc != 0

    captured = capsys.readouterr()
    assert "nonexistent" in captured.err or "profile" in captured.err.lower()


# ---------------------------------------------------------------------------
# AC: CLI invocation (ccproxy replay exits 0)
# ---------------------------------------------------------------------------

def test_ccproxy_replay_cli_exits_zero(tmp_path):
    """ac-cli-invocation: 'ccproxy replay <file> --profile <name>' exits 0 on success."""
    from unittest.mock import patch as _patch

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    alpha_resp = json.dumps(ALPHA_RESPONSE).encode()

    async def _mock_post(url, *, content, headers, **kwargs):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.content = alpha_resp
        return mock_resp

    with _patch("services.replay.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.post = _mock_post
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        MockClient.return_value = mock_instance

        from ccproxy import cmd_replay
        rc = cmd_replay(
            str(capture_file),
            profile="alpha",
            stream_override=None,
            config_path=config_path,
        )
        assert rc == 0


def test_ccproxy_replay_missing_file_exits_nonzero(tmp_path, capsys):
    """ac-missing-file: CLI exits non-zero when capture file is missing."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    from ccproxy import cmd_replay
    rc = cmd_replay(
        str(tmp_path / "missing.json"),
        profile="alpha",
        stream_override=None,
        config_path=config_path,
    )
    assert rc != 0


def test_ccproxy_replay_unknown_profile_exits_nonzero(tmp_path, capsys):
    """ac-missing-profile: CLI exits non-zero when profile is not configured."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    capture_file = tmp_path / "request_001.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    from ccproxy import cmd_replay
    rc = cmd_replay(
        str(capture_file),
        profile="ghost",
        stream_override=None,
        config_path=config_path,
    )
    assert rc != 0
