"""Smoke tests for issue #48: installable console entrypoints.

AC coverage:
- ac-import-main: main module is importable without error
- ac-import-ccswitch: ccswitch module is importable without error
- ac-ccswitch-list: ccswitch list exits 0 against config.example.toml
- ac-config-example: config.example.toml exists at repo root
- ac-env-example: .env.example exists at repo root
- ac-no-secrets-toml: config.example.toml contains no literal secret values
- ac-no-secrets-env: .env.example contains no literal secret values
- ac-entrypoints-declared: pyproject.toml declares claude-proxy and ccswitch entrypoints
- ac-bounded-deps: pyproject.toml runtime deps have version specifiers
- ac-entrypoints-callable: main module exposes a callable main()
"""

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Grep pattern from UAT step 6 (extended for common secret formats)
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9]{20,}"
    r"|Bearer [A-Za-z0-9]{10,}"
    r"|password\s*=\s*[^\$\s\"\'#\n]+)",
    re.IGNORECASE,
)


def test_import_main():
    """ac-import-main: main module is importable without error."""
    import main  # noqa: F401


def test_import_ccswitch():
    """ac-import-ccswitch: ccswitch module is importable without error."""
    import ccswitch  # noqa: F401


def test_main_has_callable_main_function():
    """ac-entrypoints-callable: main.main is a callable for the console entrypoint."""
    import main
    assert callable(getattr(main, "main", None)), "main.main() function not defined"


def test_config_example_toml_exists():
    """ac-config-example: config.example.toml is present at repo root."""
    assert (REPO_ROOT / "config.example.toml").exists(), "config.example.toml missing"


def test_env_example_exists():
    """ac-env-example: .env.example is present at repo root."""
    assert (REPO_ROOT / ".env.example").exists(), ".env.example missing"


def test_config_example_toml_is_valid_toml():
    """config.example.toml must be parseable TOML."""
    with open(REPO_ROOT / "config.example.toml", "rb") as f:
        data = tomllib.load(f)
    assert "profiles" in data, "config.example.toml must define [profiles.*]"


def test_config_example_has_anthropic_passthrough():
    """config.example.toml must include an Anthropic passthrough profile."""
    with open(REPO_ROOT / "config.example.toml", "rb") as f:
        data = tomllib.load(f)
    profiles = data.get("profiles", {})
    passthrough = [n for n, p in profiles.items() if p.get("kind") == "passthrough"]
    assert passthrough, "No passthrough profile in config.example.toml"


def test_config_example_has_openai_profile_with_pricing():
    """config.example.toml must include an OpenAI-compatible profile with pricing."""
    with open(REPO_ROOT / "config.example.toml", "rb") as f:
        data = tomllib.load(f)
    profiles = data.get("profiles", {})
    openai_profiles = [n for n, p in profiles.items() if p.get("kind") == "openai"]
    assert openai_profiles, "No openai profile in config.example.toml"
    p = profiles[openai_profiles[0]]
    assert "pricing" in p, "OpenAI profile must include a [pricing] section"


def test_config_example_has_api_key_env():
    """config.example.toml profiles must reference api_key_env (no literal secrets)."""
    with open(REPO_ROOT / "config.example.toml", "rb") as f:
        data = tomllib.load(f)
    profiles = data.get("profiles", {})
    # At least one profile must declare api_key_env
    has_api_key_env = any(p.get("api_key_env") for p in profiles.values())
    assert has_api_key_env, "No profile in config.example.toml defines api_key_env"


def test_ccswitch_list_exits_zero_with_example_config(capsys):
    """ac-ccswitch-list: ccswitch list exits 0 and lists >= 2 profiles."""
    from ccswitch import cmd_list
    example_config = REPO_ROOT / "config.example.toml"
    rc = cmd_list(config_path=example_config)
    assert rc == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.strip().splitlines() if ln]
    assert len(lines) >= 2, f"Expected >= 2 profiles, got: {lines}"


def test_no_secrets_in_config_example_toml():
    """ac-no-secrets-toml: config.example.toml contains no literal secret values."""
    content = (REPO_ROOT / "config.example.toml").read_text()
    matches = _SECRET_RE.findall(content)
    assert not matches, f"Potential secrets found in config.example.toml: {matches}"


def test_no_secrets_in_env_example():
    """ac-no-secrets-env: .env.example contains no literal secret values."""
    content = (REPO_ROOT / ".env.example").read_text()
    matches = _SECRET_RE.findall(content)
    assert not matches, f"Potential secrets found in .env.example: {matches}"


def test_pyproject_declares_both_entrypoints():
    """ac-entrypoints-declared: pyproject.toml has claude-proxy and ccswitch in [project.scripts]."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    scripts = data.get("project", {}).get("scripts", {})
    assert "claude-proxy" in scripts, "claude-proxy missing from [project.scripts]"
    assert "ccswitch" in scripts, "ccswitch missing from [project.scripts]"


def test_pyproject_deps_have_version_specifiers():
    """ac-bounded-deps: runtime dependencies in pyproject.toml carry version bounds."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    # Deps that are just a bare package name with no version operator
    bare = [d for d in deps if not re.search(r"[><=!~]", d)]
    assert not bare, f"Deps without version specifiers: {bare}"
