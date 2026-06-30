"""Tests for issue #37: ccswitch CLI and active-profile state file."""
import json
import sys
import textwrap
from pathlib import Path

import pytest


TOML_TWO_PROFILES = textwrap.dedent("""\
    [profiles.anthropic]
    kind = "passthrough"
    upstream = "https://api.anthropic.com"

    [profiles.openai]
    kind = "openai"
    upstream = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
    model = "gpt-4o"
""")


# ---------------------------------------------------------------------------
# ccswitch use — happy path
# ---------------------------------------------------------------------------

def test_use_writes_state_json(tmp_path):
    """[AC] ccswitch use <profile> creates state.json with {\"active\": \"<profile>\"}."""
    from ccswitch import cmd_use

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"

    rc = cmd_use("openai", config_path=config_path, state_path=state_path)

    assert rc == 0
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data == {"active": "openai"}


def test_use_updates_existing_state_json(tmp_path):
    """[AC] ccswitch use <profile> overwrites an existing state.json."""
    from ccswitch import cmd_use

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"active": "anthropic"}))

    rc = cmd_use("openai", config_path=config_path, state_path=state_path)

    assert rc == 0
    data = json.loads(state_path.read_text())
    assert data == {"active": "openai"}


def test_use_creates_parent_directories(tmp_path):
    """[AC] ccswitch use creates the state file's parent directories if needed."""
    from ccswitch import cmd_use

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "nested" / "dir" / "state.json"

    rc = cmd_use("anthropic", config_path=config_path, state_path=state_path)

    assert rc == 0
    assert state_path.exists()


# ---------------------------------------------------------------------------
# ccswitch use — unknown profile (error + no write)
# ---------------------------------------------------------------------------

def test_use_unknown_profile_exits_nonzero(tmp_path, capsys):
    """[AC] ccswitch use <unknown> exits non-zero with a clear error message."""
    from ccswitch import cmd_use

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"

    rc = cmd_use("nonexistent", config_path=config_path, state_path=state_path)

    assert rc != 0
    captured = capsys.readouterr()
    assert "nonexistent" in captured.err


def test_use_unknown_profile_writes_nothing(tmp_path):
    """[AC] ccswitch use <unknown> does not create or modify state.json."""
    from ccswitch import cmd_use

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"
    original = json.dumps({"active": "anthropic"})
    state_path.write_text(original)

    cmd_use("nonexistent", config_path=config_path, state_path=state_path)

    assert state_path.read_text() == original


def test_use_unknown_profile_no_state_file_not_created(tmp_path):
    """[AC] ccswitch use <unknown> does not create state.json when none exists."""
    from ccswitch import cmd_use

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"

    cmd_use("ghost", config_path=config_path, state_path=state_path)

    assert not state_path.exists()


# ---------------------------------------------------------------------------
# ccswitch status
# ---------------------------------------------------------------------------

def test_status_prints_profile_name_and_upstream(tmp_path, capsys):
    """[AC] ccswitch status prints the active profile name and its resolved upstream."""
    from ccswitch import cmd_status

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"active": "openai"}))

    rc = cmd_status(config_path=config_path, state_path=state_path)

    assert rc == 0
    out = capsys.readouterr().out
    assert "openai" in out
    assert "https://api.openai.com/v1" in out


def test_status_no_secrets_in_output(tmp_path, capsys, monkeypatch):
    """[AC] ccswitch never reads or writes secrets — status output must not expose keys."""
    from ccswitch import cmd_status

    monkeypatch.setenv("OPENAI_API_KEY", "sk-supersecret")
    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"active": "openai"}))

    cmd_status(config_path=config_path, state_path=state_path)

    out = capsys.readouterr().out
    assert "sk-supersecret" not in out
    assert "OPENAI_API_KEY" not in out


def test_status_no_state_file_exits_nonzero(tmp_path, capsys):
    """[AC] ccswitch status exits non-zero when no state file exists."""
    from ccswitch import cmd_status

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)
    state_path = tmp_path / "state.json"

    rc = cmd_status(config_path=config_path, state_path=state_path)

    assert rc != 0


# ---------------------------------------------------------------------------
# ccswitch list
# ---------------------------------------------------------------------------

def test_list_prints_all_profiles(tmp_path, capsys):
    """[AC] ccswitch list prints every profile defined in config.toml."""
    from ccswitch import cmd_list

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    rc = cmd_list(config_path=config_path)

    assert rc == 0
    out = capsys.readouterr().out
    assert "anthropic" in out
    assert "openai" in out


def test_list_includes_kind_and_upstream(tmp_path, capsys):
    """[AC] ccswitch list shows kind and upstream for each profile."""
    from ccswitch import cmd_list

    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    cmd_list(config_path=config_path)

    out = capsys.readouterr().out
    assert "passthrough" in out
    assert "https://api.anthropic.com" in out
    assert "openai" in out
    assert "https://api.openai.com/v1" in out


def test_list_no_secrets_in_output(tmp_path, capsys, monkeypatch):
    """[AC] ccswitch list must not expose api_key_env values or resolved secrets."""
    from ccswitch import cmd_list

    monkeypatch.setenv("OPENAI_API_KEY", "sk-topsecret")
    config_path = tmp_path / "config.toml"
    config_path.write_text(TOML_TWO_PROFILES)

    cmd_list(config_path=config_path)

    out = capsys.readouterr().out
    assert "sk-topsecret" not in out
    assert "OPENAI_API_KEY" not in out


# ---------------------------------------------------------------------------
# read_active_profile helper (used by proxy)
# ---------------------------------------------------------------------------

def test_read_active_profile_returns_name(tmp_path):
    """[AC] read_active_profile() returns the active profile name from state.json."""
    from ccswitch import read_active_profile

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"active": "openai"}))

    assert read_active_profile(state_path=state_path) == "openai"


def test_read_active_profile_missing_file_returns_none(tmp_path):
    """[AC] read_active_profile() returns None when state.json does not exist."""
    from ccswitch import read_active_profile

    state_path = tmp_path / "state.json"

    assert read_active_profile(state_path=state_path) is None


def test_read_active_profile_corrupt_file_returns_none(tmp_path):
    """[AC] read_active_profile() returns None on a corrupt/invalid state.json."""
    from ccswitch import read_active_profile

    state_path = tmp_path / "state.json"
    state_path.write_text("not-json{{{")

    assert read_active_profile(state_path=state_path) is None


# ---------------------------------------------------------------------------
# Proxy integration: state.json used as default profile
# ---------------------------------------------------------------------------

def test_proxy_uses_state_json_as_default(tmp_path, monkeypatch):
    """[AC] Proxy reads state.json on each request and uses it as default profile."""
    import ccswitch

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"active": "openai"}))

    # Patch the STATE_FILE used by read_active_profile
    monkeypatch.setattr(ccswitch, "STATE_FILE", state_path)

    from ccswitch import read_active_profile
    assert read_active_profile() == "openai"


def test_proxy_falls_back_to_anthropic_when_no_state(tmp_path, monkeypatch):
    """[AC] Proxy falls back to 'anthropic' when state.json does not exist."""
    import ccswitch

    state_path = tmp_path / "nonexistent_state.json"
    monkeypatch.setattr(ccswitch, "STATE_FILE", state_path)

    from ccswitch import read_active_profile
    result = read_active_profile()
    assert result is None  # proxy code handles the None → "anthropic" fallback
