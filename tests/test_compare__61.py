"""Tests for issue #61: ccproxy compare command for multi-profile replay.

AC coverage:
- success_two_profiles: table produced, manifest written, cheapest and fastest
  markers present, exit code 0
- partial_failure: FAILED profile shown in table, successful profile data present,
  manifest records both outcomes, exit code 0
"""
import asyncio
import json
from io import StringIO
from unittest.mock import MagicMock, patch

from ccproxy import cmd_compare, _replay_profile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CONFIG_TOML = """\
[profiles.profile-a]
kind = "passthrough"
upstream = "http://fake-a.test"

[profiles.profile-a.pricing]
input_per_mtok = 3.0
output_per_mtok = 15.0

[profiles.profile-b]
kind = "passthrough"
upstream = "http://fake-b.test"

[profiles.profile-b.pricing]
input_per_mtok = 1.0
output_per_mtok = 5.0
"""

SAMPLE_CAPTURE = {
    "request_id": "test-request-id",
    "profile": {"name": "profile-a", "settings": {}},
    "request": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hello"}],
    },
    "response": {},
    "timing": {"start": "2026-01-01T00:00:00Z", "duration_ms": 100},
}

RESPONSE_A = {
    "id": "msg_a",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from A"}],
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 200, "output_tokens": 100},
}

RESPONSE_B = {
    "id": "msg_b",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello from B"}],
    "model": "claude-haiku-4-5-20251001",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 50, "output_tokens": 20},
}


def _make_mock_response(status_code: int, body: dict):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body)
    resp.text = json.dumps(body)
    return resp


class _FakeAsyncClient:
    """Async context manager mock that dispatches post() calls by URL."""

    def __init__(self, url_responses: dict):
        self._url_responses = url_responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, *, content, headers, **kwargs):
        for key, value in self._url_responses.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                return value
        raise ValueError(f"No mock for URL: {url}")


# ---------------------------------------------------------------------------
# AC: successful two-profile comparison
# ---------------------------------------------------------------------------


def test_compare_two_profiles_success_exit_zero(tmp_path):
    """AC: exit code is 0 when both profiles succeed."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": _make_mock_response(200, RESPONSE_B),
    })

    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        rc = cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=StringIO(),
        )

    assert rc == 0


def test_compare_two_profiles_manifest_written(tmp_path):
    """AC: a JSON manifest file is written after the run."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": _make_mock_response(200, RESPONSE_B),
    })

    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=StringIO(),
        )

    manifests = list(tmp_path.glob("compare-*.json"))
    assert len(manifests) == 1, f"Expected 1 manifest file, got {len(manifests)}"

    manifest = json.loads(manifests[0].read_text())
    assert "results" in manifest
    assert len(manifest["results"]) == 2


def test_compare_two_profiles_manifest_contains_all_fields(tmp_path):
    """AC: manifest contains all per-profile result fields."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": _make_mock_response(200, RESPONSE_B),
    })

    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=StringIO(),
        )

    manifests = list(tmp_path.glob("compare-*.json"))
    manifest = json.loads(manifests[0].read_text())
    for result in manifest["results"]:
        assert "profile" in result
        assert "status" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "latency_ms" in result
        assert "finish_reason" in result


def test_compare_table_contains_required_columns(tmp_path):
    """AC: table shows est_cost_usd, input_tokens, output_tokens, latency_ms, finish_reason, preview."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": _make_mock_response(200, RESPONSE_B),
    })

    out = StringIO()
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=out,
        )

    table = out.getvalue()
    assert "est_cost_usd" in table
    assert "input_tokens" in table
    assert "output_tokens" in table
    assert "latency_ms" in table
    assert "finish_reason" in table
    assert "profile-a" in table
    assert "profile-b" in table


def test_compare_cheapest_marker_present(tmp_path):
    """AC: cheapest marker appears on the lowest-cost profile row."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    # profile-b has fewer tokens and cheaper per-token pricing → lowest cost
    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": _make_mock_response(200, RESPONSE_B),
    })

    out = StringIO()
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=out,
        )

    table = out.getvalue()
    assert "CHEAPEST" in table, "CHEAPEST marker must appear in table output"

    # profile-b is cheaper: (50/1e6)*1.0 + (20/1e6)*5.0 = 0.000050 + 0.000100 = $0.000150
    # profile-a: (200/1e6)*3.0 + (100/1e6)*15.0 = 0.000600 + 0.001500 = $0.002100
    lines = table.splitlines()
    profile_b_line = next(ln for ln in lines if "profile-b" in ln)
    assert "CHEAPEST" in profile_b_line, "CHEAPEST must appear on profile-b row (lower cost)"


def test_compare_fastest_marker_present(tmp_path):
    """AC: fastest marker appears in the table output."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": _make_mock_response(200, RESPONSE_B),
    })

    out = StringIO()
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=out,
        )

    table = out.getvalue()
    assert "FASTEST" in table, "FASTEST marker must appear in table output"


def test_compare_no_config_changes(tmp_path, monkeypatch):
    """AC: compare command makes no changes to any config file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)
    original_content = config_file.read_text()

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": _make_mock_response(200, RESPONSE_B),
    })

    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=StringIO(),
        )

    assert config_file.read_text() == original_content, "config.toml must not be modified"


# ---------------------------------------------------------------------------
# AC: partial failure (one profile fails, one succeeds)
# ---------------------------------------------------------------------------


def test_partial_failure_exit_zero(tmp_path):
    """AC: exit code is 0 when at least one profile succeeds."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": ConnectionError("connection refused"),
    })

    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        rc = cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=StringIO(),
        )

    assert rc == 0, "exit code must be 0 when at least one profile succeeds"


def test_partial_failure_failed_profile_in_table(tmp_path):
    """AC: failed profile appears as FAILED in the table."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": ConnectionError("connection refused"),
    })

    out = StringIO()
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=out,
        )

    table = out.getvalue()
    lines = table.splitlines()
    profile_b_line = next(ln for ln in lines if "profile-b" in ln)
    assert "FAILED" in profile_b_line, "profile-b must appear as FAILED in the table"


def test_partial_failure_successful_profile_data_present(tmp_path):
    """AC: successful profile data is still present when other profile fails."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": ConnectionError("connection refused"),
    })

    out = StringIO()
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=out,
        )

    table = out.getvalue()
    lines = table.splitlines()
    profile_a_line = next(ln for ln in lines if "profile-a" in ln)
    # profile-a should show token counts and cost, not FAILED
    assert "FAILED" not in profile_a_line, "profile-a must not be shown as FAILED"
    assert "200" in profile_a_line or "end_turn" in profile_a_line, (
        "profile-a row must contain its result data"
    )


def test_partial_failure_manifest_records_both(tmp_path):
    """AC: manifest records both the failed and successful profile outcomes."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
        "fake-b.test": ConnectionError("connection refused"),
    })

    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=StringIO(),
        )

    manifests = list(tmp_path.glob("compare-*.json"))
    manifest = json.loads(manifests[0].read_text())
    results_by_profile = {r["profile"]: r for r in manifest["results"]}

    assert "profile-a" in results_by_profile
    assert "profile-b" in results_by_profile
    assert results_by_profile["profile-a"]["status"] == "OK"
    assert results_by_profile["profile-b"]["status"] == "FAILED"
    assert "error" in results_by_profile["profile-b"]


# ---------------------------------------------------------------------------
# AC: all profiles fail → exit non-zero
# ---------------------------------------------------------------------------


def test_all_profiles_fail_exit_nonzero(tmp_path):
    """AC: exit code is non-zero only when ALL profiles fail."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": ConnectionError("refused"),
        "fake-b.test": ConnectionError("refused"),
    })

    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        rc = cmd_compare(
            capture_file,
            ["profile-a", "profile-b"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=StringIO(),
        )

    assert rc != 0, "exit code must be non-zero when all profiles fail"


# ---------------------------------------------------------------------------
# AC: missing capture file → error
# ---------------------------------------------------------------------------


def test_missing_capture_file_returns_nonzero(tmp_path):
    """AC: non-existent capture file causes non-zero exit."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    rc = cmd_compare(
        tmp_path / "does_not_exist.json",
        ["profile-a"],
        config_path=config_file,
        manifest_dir=tmp_path,
        output=StringIO(),
    )
    assert rc != 0


# ---------------------------------------------------------------------------
# AC: unknown profile → FAILED row, not crash
# ---------------------------------------------------------------------------


def test_unknown_profile_shows_failed_in_table(tmp_path):
    """AC: unknown profile name is shown as FAILED with error reason, does not crash."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    capture_file = tmp_path / "capture.json"
    capture_file.write_text(json.dumps(SAMPLE_CAPTURE))

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
    })

    out = StringIO()
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        rc = cmd_compare(
            capture_file,
            ["profile-a", "no-such-profile"],
            config_path=config_file,
            manifest_dir=tmp_path,
            output=out,
        )

    table = out.getvalue()
    assert "no-such-profile" in table
    lines = table.splitlines()
    bad_line = next(ln for ln in lines if "no-such-profile" in ln)
    assert "FAILED" in bad_line
    # Partial success — one profile is OK
    assert rc == 0


# ---------------------------------------------------------------------------
# Unit: _replay_profile async function
# ---------------------------------------------------------------------------


def test_replay_profile_success(tmp_path):
    """Unit: _replay_profile returns OK result with correct fields."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(200, RESPONSE_A),
    })

    request_body = SAMPLE_CAPTURE["request"]
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        result = asyncio.run(_replay_profile("profile-a", request_body, config_file))

    assert result["status"] == "OK"
    assert result["input_tokens"] == 200
    assert result["output_tokens"] == 100
    assert result["finish_reason"] == "end_turn"
    assert "Hello from A" in result["preview"]
    assert result["est_cost_usd"] is not None
    assert result["latency_ms"] >= 0


def test_replay_profile_http_error(tmp_path):
    """Unit: _replay_profile returns FAILED result on HTTP 4xx."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    error_body = {"error": {"type": "auth_error", "message": "Invalid API key"}}
    fake_client = _FakeAsyncClient({
        "fake-a.test": _make_mock_response(401, error_body),
    })

    request_body = SAMPLE_CAPTURE["request"]
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        result = asyncio.run(_replay_profile("profile-a", request_body, config_file))

    assert result["status"] == "FAILED"
    assert "401" in result["error"]


def test_replay_profile_network_error(tmp_path):
    """Unit: _replay_profile returns FAILED result on network error."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(SAMPLE_CONFIG_TOML)

    fake_client = _FakeAsyncClient({
        "fake-a.test": ConnectionError("connection refused"),
    })

    request_body = SAMPLE_CAPTURE["request"]
    with patch("ccproxy.httpx.AsyncClient", return_value=fake_client):
        result = asyncio.run(_replay_profile("profile-a", request_body, config_file))

    assert result["status"] == "FAILED"
    assert result["error"] != ""
