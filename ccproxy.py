"""ccproxy — CLI for proxy utilities.

Usage:
  ccproxy replay <capture-file> --profile <name> [--stream | --no-stream]

Commands:
  replay   Replay a captured request through a configured profile.
"""

import argparse
import asyncio
import sys
from pathlib import Path


def cmd_replay(
    capture_file: str,
    *,
    profile: str,
    stream_override: bool | None,
    config_path: Path = Path("config.toml"),
) -> int:
    """Run the replay command synchronously. Returns exit code."""
    from services.replay import replay

    return asyncio.run(
        replay(
            Path(capture_file),
            profile,
            stream_override=stream_override,
            config_path=config_path,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ccproxy",
        description="claude-proxy CLI utilities",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    replay_p = sub.add_parser("replay", help="Replay a captured request")
    replay_p.add_argument("capture_file", metavar="<capture-file>")
    replay_p.add_argument("--profile", required=True, metavar="<name>")
    stream_grp = replay_p.add_mutually_exclusive_group()
    stream_grp.add_argument(
        "--stream",
        dest="stream_override",
        action="store_true",
        default=None,
        help="Force streaming mode",
    )
    stream_grp.add_argument(
        "--no-stream",
        dest="stream_override",
        action="store_false",
        help="Force non-streaming mode",
    )

    args = parser.parse_args()

    if args.command == "replay":
        stream_override: bool | None = args.stream_override
        rc = cmd_replay(
            args.capture_file,
            profile=args.profile,
            stream_override=stream_override,
        )
        sys.exit(rc)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
