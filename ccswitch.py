"""ccswitch — CLI for switching the proxy's active profile at runtime.

Usage:
  ccswitch use <profile>   Validate and activate a profile from config.toml
  ccswitch status          Show the currently active profile and its upstream
  ccswitch list            List all profiles defined in config.toml

State file: ~/.config/ccswitch/state.json  (user-scoped, no secrets stored)
"""

import argparse
import json
import sys
from pathlib import Path

from profiles import get_or_load_config

STATE_FILE = Path.home() / ".config" / "ccswitch" / "state.json"
CONFIG_FILE = Path("config.toml")


def read_active_profile(*, state_path: Path | None = None) -> str | None:
    """Return the active profile name from state.json, or None if unavailable."""
    path = state_path if state_path is not None else STATE_FILE
    try:
        if path.exists():
            data = json.loads(path.read_text())
            return data.get("active") or None
    except (json.JSONDecodeError, OSError):
        pass
    return None


def cmd_use(
    profile: str,
    *,
    config_path: Path | None = None,
    state_path: Path | None = None,
) -> int:
    """Validate profile against config.toml and write state.json. Returns exit code."""
    cfg_path = config_path if config_path is not None else CONFIG_FILE
    st_path = state_path if state_path is not None else STATE_FILE
    config, _ = get_or_load_config(cfg_path)
    if profile not in config.profiles:
        known = ", ".join(sorted(config.profiles)) or "(none)"
        print(
            f"error: unknown profile '{profile}' (known: {known})",
            file=sys.stderr,
        )
        return 1
    st_path.parent.mkdir(parents=True, exist_ok=True)
    st_path.write_text(json.dumps({"active": profile}))
    print(f"Active profile set to '{profile}'.")
    return 0


def cmd_status(
    *,
    config_path: Path | None = None,
    state_path: Path | None = None,
) -> int:
    """Print the active profile name and its resolved upstream URL. Returns exit code."""
    active = read_active_profile(state_path=state_path)
    if not active:
        print("error: no active profile set — run 'ccswitch use <profile>'", file=sys.stderr)
        return 1
    cfg_path = config_path if config_path is not None else CONFIG_FILE
    config, _ = get_or_load_config(cfg_path)
    if active not in config.profiles:
        print(
            f"error: active profile '{active}' not found in config.toml",
            file=sys.stderr,
        )
        return 1
    profile = config.profiles[active]
    print(f"{active}  {profile.kind}  {profile.upstream}")
    return 0


def cmd_list(*, config_path: Path | None = None) -> int:
    """List all profiles in config.toml with kind and upstream. Returns exit code."""
    cfg_path = config_path if config_path is not None else CONFIG_FILE
    config, _ = get_or_load_config(cfg_path)
    if not config.profiles:
        print("(no profiles defined in config.toml)")
        return 0
    for name, profile in config.profiles.items():
        print(f"{name}  {profile.kind}  {profile.upstream}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ccswitch",
        description="Switch the claude-proxy active profile at runtime.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    use_p = sub.add_parser("use", help="Activate a profile")
    use_p.add_argument("profile", help="Profile name from config.toml")

    sub.add_parser("status", help="Show the currently active profile")
    sub.add_parser("list", help="List all profiles defined in config.toml")

    args = parser.parse_args()

    if args.command == "use":
        rc = cmd_use(args.profile)
    elif args.command == "status":
        rc = cmd_status()
    elif args.command == "list":
        rc = cmd_list()
    else:
        parser.print_help()
        rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()
