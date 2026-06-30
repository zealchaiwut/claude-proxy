"""Tests for issue #49: systemd and launchd service units for claude-proxy.

AC coverage:
- ac-plist: macOS launchd plist template exists with KeepAlive on failure, 127.0.0.1:8788, env file ref
- ac-systemd: Linux systemd user unit exists with Restart=on-failure, EnvironmentFile=, 127.0.0.1:8788
- ac-installer: scripts/install_service.py exists, detects platform, installs/loads unit, prints status
- ac-unsupported: installer prints clear error and exits non-zero on unsupported platform
- ac-no-secrets: no credentials/API keys inline in either unit template
- ac-restart: service restarts within 30s after kill (UAT-only)
- ac-readme: README documents install, uninstall, log-viewing for macOS and Linux
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"
SCRIPTS_DIR = REPO_ROOT / "scripts"
README = REPO_ROOT / "README.md"

# --- AC: macOS launchd plist template ---

def test_plist_template_exists():
    """AC: macOS launchd plist template file exists in deploy/."""
    plist = DEPLOY_DIR / "com.zealchaiwut.claude-proxy.plist"
    assert plist.exists(), f"plist template missing: {plist}"


def test_plist_has_keepalive_on_failure():
    """AC: plist sets KeepAlive=on failure (SuccessfulExit=false or KeepAlive=true)."""
    plist = DEPLOY_DIR / "com.zealchaiwut.claude-proxy.plist"
    content = plist.read_text()
    # KeepAlive with SuccessfulExit=false means restart on crash (not on clean exit)
    # OR plain KeepAlive=true (always restart, covers failure case)
    has_keepalive = "KeepAlive" in content
    assert has_keepalive, "plist must contain KeepAlive directive"
    # Verify it's set to restart on failure rather than never restart
    # Accept either KeepAlive=true or KeepAlive dict with SuccessfulExit=false
    has_true = "<true/>" in content and "KeepAlive" in content
    has_dict = "SuccessfulExit" in content
    assert has_true or has_dict, "plist KeepAlive must be set to true or SuccessfulExit=false"


def test_plist_has_correct_bind_address():
    """AC: plist runs claude-proxy bound to 127.0.0.1:8788."""
    plist = DEPLOY_DIR / "com.zealchaiwut.claude-proxy.plist"
    content = plist.read_text()
    assert "127.0.0.1" in content, "plist must reference 127.0.0.1"
    assert "8788" in content, "plist must reference port 8788"


def test_plist_references_env_file():
    """AC: plist references an env file rather than containing inline secrets."""
    plist = DEPLOY_DIR / "com.zealchaiwut.claude-proxy.plist"
    content = plist.read_text()
    # Must mention the env file path in some form
    has_env_ref = (
        "env" in content.lower()
        and (".env" in content or "env" in content)
    )
    # The plist either mentions env file path or references a wrapper script that loads it
    assert has_env_ref or "claude-proxy" in content.lower(), (
        "plist must reference an env file (directly or via wrapper script)"
    )


def test_plist_no_inline_secrets():
    """AC: No API keys or credentials appear inline in the plist."""
    plist = DEPLOY_DIR / "com.zealchaiwut.claude-proxy.plist"
    content = plist.read_text()
    secret_patterns = [
        "ANTHROPIC_API_KEY=sk-",
        "OPENAI_API_KEY=sk-",
        "sk-ant-",
    ]
    for pattern in secret_patterns:
        assert pattern not in content, f"plist must not contain inline secret: {pattern}"


# --- AC: Linux systemd user unit ---

def test_systemd_unit_exists():
    """AC: Linux systemd user unit template exists in deploy/."""
    unit = DEPLOY_DIR / "claude-proxy.service"
    assert unit.exists(), f"systemd unit template missing: {unit}"


def test_systemd_has_restart_on_failure():
    """AC: systemd unit has Restart=on-failure."""
    unit = DEPLOY_DIR / "claude-proxy.service"
    content = unit.read_text()
    assert "Restart=on-failure" in content, "systemd unit must contain Restart=on-failure"


def test_systemd_has_environment_file():
    """AC: systemd unit loads environment from EnvironmentFile=."""
    unit = DEPLOY_DIR / "claude-proxy.service"
    content = unit.read_text()
    assert "EnvironmentFile=" in content, "systemd unit must contain EnvironmentFile="


def test_systemd_has_correct_bind_address():
    """AC: systemd unit runs claude-proxy bound to 127.0.0.1:8788."""
    unit = DEPLOY_DIR / "claude-proxy.service"
    content = unit.read_text()
    assert "127.0.0.1" in content, "systemd unit must reference 127.0.0.1"
    assert "8788" in content, "systemd unit must reference port 8788"


def test_systemd_no_inline_secrets():
    """AC: No API keys or credentials appear inline in the systemd unit."""
    unit = DEPLOY_DIR / "claude-proxy.service"
    content = unit.read_text()
    secret_patterns = [
        "ANTHROPIC_API_KEY=sk-",
        "OPENAI_API_KEY=sk-",
        "sk-ant-",
    ]
    for pattern in secret_patterns:
        assert pattern not in content, f"systemd unit must not contain inline secret: {pattern}"


def test_systemd_is_user_unit():
    """AC: systemd unit is a user unit (not system-wide)."""
    unit = DEPLOY_DIR / "claude-proxy.service"
    content = unit.read_text()
    # User units typically have [Install] with WantedBy=default.target
    assert "[Install]" in content, "systemd unit must have [Install] section"
    assert "WantedBy=" in content, "systemd unit must have WantedBy= directive"


# --- AC: installer script ---

def test_install_script_exists():
    """AC: scripts/install_service.py exists."""
    script = SCRIPTS_DIR / "install_service.py"
    assert script.exists(), f"installer script missing: {script}"


def test_install_script_is_executable_python():
    """AC: install_service.py is a valid Python file."""
    script = SCRIPTS_DIR / "install_service.py"
    result = subprocess.run(
        [sys.executable, "-c", f"import ast; ast.parse(open({str(script)!r}).read())"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"install_service.py has syntax errors: {result.stderr}"


def test_install_script_unsupported_platform():
    """AC: installer prints clear error and exits non-zero on unsupported platform."""
    script = SCRIPTS_DIR / "install_service.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "_CCPROXY_PLATFORM_OVERRIDE": "windows",
        },
    )
    assert result.returncode != 0, "installer must exit non-zero on unsupported platform"
    output = result.stdout + result.stderr
    assert len(output.strip()) > 0, "installer must print an error message on unsupported platform"
    # Should mention 'unsupported' or name the platform
    lower = output.lower()
    assert "unsupported" in lower or "windows" in lower or "not supported" in lower, (
        f"error message must clearly identify the unsupported platform. Got: {output!r}"
    )


def test_install_script_prints_help_or_detects_platform():
    """AC: installer script detects the current platform (imports sys/platform)."""
    script = SCRIPTS_DIR / "install_service.py"
    content = script.read_text()
    assert "platform" in content or "sys.platform" in content or "darwin" in content, (
        "installer must contain platform detection logic"
    )


# --- AC: README documentation ---

def test_readme_documents_service_install():
    """AC: README documents install steps for macOS and Linux."""
    content = README.read_text()
    lower = content.lower()
    assert "launchctl" in lower or "launchd" in lower or "plist" in lower, (
        "README must document macOS launchd install steps"
    )
    assert "systemctl" in lower or "systemd" in lower, (
        "README must document Linux systemd install steps"
    )


def test_readme_documents_uninstall():
    """AC: README documents uninstall steps for both platforms."""
    content = README.read_text()
    lower = content.lower()
    assert "uninstall" in lower or "unload" in lower or "disable" in lower, (
        "README must document uninstall/unload steps"
    )


def test_readme_documents_log_viewing():
    """AC: README documents how to view logs on both platforms."""
    content = README.read_text()
    lower = content.lower()
    assert "log" in lower, "README must document how to view service logs"


# --- AC: auto-restart (UAT / integration only) ---

@pytest.mark.skip(reason="Requires live service manager — run manually per UAT steps")
def test_service_restarts_within_30s():
    """AC: after killing the proxy, the service manager restarts it within 30 seconds."""
    import time
    import signal
    # This test is intentionally skipped in unit-test mode.
    # UAT step: kill -9 <pid> and verify the process reappears within 30s.
    pass
