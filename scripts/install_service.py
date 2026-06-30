#!/usr/bin/env python3
"""Install claude-proxy as a platform-native background service.

Usage:
    python scripts/install_service.py [--dry-run]

Platforms:
    macOS  — installs a launchd user agent (~/Library/LaunchAgents/)
    Linux  — installs a systemd user unit (~/.config/systemd/user/)

Secrets:
    All credentials are sourced from ~/.config/claude-proxy/env at startup.
    No API keys are written into the service unit.

Uninstall:
    macOS:  launchctl unload ~/Library/LaunchAgents/com.zealchaiwut.claude-proxy.plist
            rm ~/Library/LaunchAgents/com.zealchaiwut.claude-proxy.plist
    Linux:  systemctl --user disable --now claude-proxy
            rm ~/.config/systemd/user/claude-proxy.service
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Allow test override of platform detection via env var.
_PLATFORM = os.environ.get("_CCPROXY_PLATFORM_OVERRIDE") or sys.platform

DEPLOY_DIR = Path(__file__).parent.parent / "deploy"
INSTALL_DIR = Path(__file__).parent.parent


def _find_uvicorn() -> str:
    found = shutil.which("uvicorn")
    if found:
        return found
    return f"{sys.executable} -m uvicorn"


def _ensure_env_file(env_path: Path) -> None:
    if env_path.exists():
        return
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        "# claude-proxy environment file\n"
        "# Add your secrets here — this file is sourced before the service starts.\n"
        "# Do NOT commit this file to version control.\n"
        "#\n"
        "# ANTHROPIC_API_KEY=sk-ant-...\n"
        "# CCPROXY_PROFILE=anthropic\n"
        "# UPSTREAM_BASE_URL=https://api.anthropic.com\n"
    )
    print(f"Created env file template: {env_path}")
    print("  Edit it to add your credentials before starting the service.")


def _install_macos(dry_run: bool = False) -> None:
    home = Path.home()
    log_dir = home / "Library" / "Logs" / "claude-proxy"
    agents_dir = home / "Library" / "LaunchAgents"
    plist_dest = agents_dir / "com.zealchaiwut.claude-proxy.plist"
    wrapper_script = home / ".local" / "share" / "claude-proxy" / "start.sh"
    env_file = home / ".config" / "claude-proxy" / "env"
    uvicorn = _find_uvicorn()
    install_dir = INSTALL_DIR.resolve()

    plist_template = (DEPLOY_DIR / "com.zealchaiwut.claude-proxy.plist").read_text()
    plist_content = (
        plist_template
        .replace("{{WRAPPER_SCRIPT}}", str(wrapper_script))
        .replace("{{INSTALL_DIR}}", str(install_dir))
        .replace("{{LOG_DIR}}", str(log_dir))
    )

    wrapper_content = (
        "#!/bin/bash\n"
        'ENV_FILE="$HOME/.config/claude-proxy/env"\n'
        '[ -f "$ENV_FILE" ] && set -a && source "$ENV_FILE" && set +a\n'
        f"exec {uvicorn} main:app \"$@\"\n"
    )

    if dry_run:
        print("[dry-run] macOS install — would write:")
        print(f"  plist:   {plist_dest}")
        print(f"  wrapper: {wrapper_script}")
        print(f"  env:     {env_file}")
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    wrapper_script.parent.mkdir(parents=True, exist_ok=True)

    wrapper_script.write_text(wrapper_content)
    wrapper_script.chmod(0o755)
    print(f"Installed wrapper script: {wrapper_script}")

    plist_dest.write_text(plist_content)
    print(f"Installed plist:          {plist_dest}")

    _ensure_env_file(env_file)

    subprocess.run(["launchctl", "unload", str(plist_dest)], capture_output=True)

    result = subprocess.run(
        ["launchctl", "load", str(plist_dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Warning: launchctl load: {result.stderr.strip()}", file=sys.stderr)
    else:
        print("Service loaded via launchctl.")

    status = subprocess.run(
        ["launchctl", "list", "com.zealchaiwut.claude-proxy"],
        capture_output=True,
        text=True,
    )
    if status.returncode == 0:
        print(f"\nService status:\n{status.stdout}")
    else:
        print("\nTo check status:  launchctl list com.zealchaiwut.claude-proxy")
        print("To view logs:     tail -f ~/Library/Logs/claude-proxy/claude-proxy.log")


def _install_linux(dry_run: bool = False) -> None:
    home = Path.home()
    systemd_dir = home / ".config" / "systemd" / "user"
    unit_dest = systemd_dir / "claude-proxy.service"
    env_file = home / ".config" / "claude-proxy" / "env"
    uvicorn = _find_uvicorn()
    install_dir = INSTALL_DIR.resolve()

    unit_template = (DEPLOY_DIR / "claude-proxy.service").read_text()
    unit_content = (
        unit_template
        .replace("{{UVICORN}}", uvicorn)
        .replace("{{INSTALL_DIR}}", str(install_dir))
    )

    if dry_run:
        print("[dry-run] Linux install — would write:")
        print(f"  unit: {unit_dest}")
        print(f"  env:  {env_file}")
        return

    systemd_dir.mkdir(parents=True, exist_ok=True)
    unit_dest.write_text(unit_content)
    print(f"Installed systemd unit: {unit_dest}")

    _ensure_env_file(env_file)

    cmds = [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "claude-proxy"],
        ["systemctl", "--user", "start", "claude-proxy"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        label = " ".join(cmd[2:])
        if result.returncode != 0:
            print(f"Warning: {label}: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"OK: {label}")

    status = subprocess.run(
        ["systemctl", "--user", "status", "claude-proxy"],
        capture_output=True,
        text=True,
    )
    print(f"\nService status:\n{status.stdout}")
    print("To view logs: journalctl --user -u claude-proxy -f")


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    if _PLATFORM == "darwin":
        _install_macos(dry_run=dry_run)
    elif _PLATFORM.startswith("linux"):
        _install_linux(dry_run=dry_run)
    else:
        print(
            f"Error: unsupported platform '{_PLATFORM}'.\n"
            "claude-proxy service installation supports macOS (launchd) and Linux (systemd) only.\n"
            "To run the proxy manually:\n"
            "  uvicorn main:app --host 127.0.0.1 --port 8788",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
