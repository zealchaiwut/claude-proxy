"""ccproxy — CLI for multi-profile comparison and replay operations.

Usage:
  ccproxy replay <capture-file> --profile <name> [--stream | --no-stream]
  ccproxy compare <capture-file> --profiles a,b[,c,...]

Commands:
  replay   Replay a captured request through a configured profile.
  compare  Replay the captured request against each listed profile and print a
           summary table plus a persisted JSON manifest.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from profiles import get_or_load_config
from services.cost_accounting import compute_est_cost, extract_usage_from_response

CONFIG_FILE = Path("config.toml")
PREVIEW_LENGTH = 200


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


async def _replay_profile(
    profile_name: str,
    request_body: dict,
    config_path: Path,
) -> dict:
    """Replay request_body against the named profile. Returns a result dict."""
    config, from_file = get_or_load_config(config_path)

    if not from_file or profile_name not in config.profiles:
        return {
            "profile": profile_name,
            "status": "FAILED",
            "error": f"profile '{profile_name}' not found in config.toml",
            "latency_ms": 0.0,
            "est_cost_usd": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "",
            "preview": "",
        }

    profile = config.profiles[profile_name]
    api_key = os.environ.get(profile.api_key_env) if profile.api_key_env else None

    # Force non-streaming for consistent comparison output
    body = {**request_body, "stream": False}

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            if profile.kind == "passthrough":
                headers: dict[str, str] = {"Content-Type": "application/json"}
                if api_key:
                    headers["x-api-key"] = api_key
                    headers["anthropic-version"] = "2023-06-01"
                resp = await client.post(
                    f"{profile.upstream}/v1/messages",
                    content=json.dumps(body).encode(),
                    headers=headers,
                )
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code >= 400:
                    return {
                        "profile": profile_name,
                        "status": "FAILED",
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        "latency_ms": latency_ms,
                        "est_cost_usd": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "finish_reason": "",
                        "preview": "",
                    }

                resp_json = resp.json()
                usage = extract_usage_from_response(resp_json)
                input_tokens, output_tokens = usage if usage else (0, 0)
                finish_reason = resp_json.get("stop_reason", "")
                preview = _extract_text_preview(resp_json.get("content", []))

            else:  # openai kind
                from schemas.anthropic import MessagesRequest
                from schemas.openai import ChatResponse
                from services.translator import from_openai_response, to_openai_request

                anthropic_req = MessagesRequest(**body)
                client_model = body.get("model", "")
                upstream_model = (
                    profile.model_map.get(client_model)
                    or profile.model
                    or "gpt-4o"
                )
                openai_req = to_openai_request(
                    anthropic_req,
                    model=upstream_model,
                    prompt_cache=profile.prompt_cache,
                    cache_provider_hint=profile.cache_provider_hint,
                    thinking_mode=profile.openai_thinking_mode or "disabled",
                )
                oai_headers: dict[str, str] = {"Content-Type": "application/json"}
                if api_key:
                    oai_headers["Authorization"] = f"Bearer {api_key}"
                resp = await client.post(
                    f"{profile.upstream}/chat/completions",
                    content=openai_req.model_dump_json().encode(),
                    headers=oai_headers,
                )
                latency_ms = (time.monotonic() - start) * 1000

                if resp.status_code >= 400:
                    return {
                        "profile": profile_name,
                        "status": "FAILED",
                        "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        "latency_ms": latency_ms,
                        "est_cost_usd": None,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "finish_reason": "",
                        "preview": "",
                    }

                openai_resp = ChatResponse(**resp.json())
                anthropic_resp = from_openai_response(openai_resp)
                resp_dict = json.loads(anthropic_resp.model_dump_json())
                usage = extract_usage_from_response(resp_dict)
                input_tokens, output_tokens = usage if usage else (0, 0)
                finish_reason = resp_dict.get("stop_reason", "")
                preview = _extract_text_preview(resp_dict.get("content", []))

            est_cost = compute_est_cost(input_tokens, output_tokens, profile.pricing)
            return {
                "profile": profile_name,
                "status": "OK",
                "est_cost_usd": est_cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "finish_reason": finish_reason,
                "preview": preview[:PREVIEW_LENGTH],
            }

    except Exception as exc:
        latency_ms = (time.monotonic() - start) * 1000
        return {
            "profile": profile_name,
            "status": "FAILED",
            "error": str(exc),
            "latency_ms": latency_ms,
            "est_cost_usd": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "finish_reason": "",
            "preview": "",
        }


def _extract_text_preview(content_blocks: list) -> str:
    """Extract concatenated text from Anthropic content blocks."""
    return "".join(
        b.get("text", "")
        for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )


async def _run_all_profiles(
    profile_names: list[str],
    request_body: dict,
    config_path: Path,
) -> list[dict]:
    """Run all profile replays concurrently and return results in input order."""
    return list(
        await asyncio.gather(
            *[_replay_profile(name, request_body, config_path) for name in profile_names]
        )
    )


def _print_table(results: list[dict], output=None) -> None:
    """Print comparison table with cheapest/fastest markers."""
    if output is None:
        output = sys.stdout

    costs_with_idx = [
        (r["est_cost_usd"], i)
        for i, r in enumerate(results)
        if r["status"] == "OK" and r.get("est_cost_usd") is not None
    ]
    latency_with_idx = [
        (r["latency_ms"], i)
        for i, r in enumerate(results)
        if r["status"] == "OK"
    ]

    cheapest_idx = min(costs_with_idx, key=lambda x: x[0])[1] if costs_with_idx else None
    fastest_idx = min(latency_with_idx, key=lambda x: x[0])[1] if latency_with_idx else None

    col_profile = max(len("Profile"), max(len(r["profile"]) for r in results))

    header = (
        f"{'Profile':<{col_profile}} | {'est_cost_usd':>12} | {'input_tokens':>12} | "
        f"{'output_tokens':>13} | {'latency_ms':>10} | {'finish_reason':<14} | preview"
    )
    sep = "-" * len(header)

    print(header, file=output)
    print(sep, file=output)

    for i, r in enumerate(results):
        markers = []
        if i == cheapest_idx:
            markers.append("CHEAPEST")
        if i == fastest_idx:
            markers.append("FASTEST")
        marker_str = "  " + " ".join(markers) if markers else ""

        if r["status"] == "FAILED":
            error = r.get("error", "unknown error")
            latency = r.get("latency_ms", 0.0)
            row = (
                f"{r['profile']:<{col_profile}} | {'FAILED':>12} | {'':>12} | "
                f"{'':>13} | {latency:>10.1f} | {'':14} | {error[:80]}"
            )
        else:
            cost_str = (
                f"${r['est_cost_usd']:.6f}" if r["est_cost_usd"] is not None else "N/A"
            )
            preview = r.get("preview", "")[:80]
            row = (
                f"{r['profile']:<{col_profile}} | {cost_str:>12} | {r['input_tokens']:>12} | "
                f"{r['output_tokens']:>13} | {r['latency_ms']:>10.1f} | "
                f"{r['finish_reason']:<14} | {preview}"
            )
        print(row + marker_str, file=output)


def _write_manifest(results: list[dict], capture_file: Path, manifest_dir: Path) -> Path:
    """Write JSON manifest and return its path."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = manifest_dir / f"compare-{capture_file.stem}-{ts}.json"
    path.write_text(
        json.dumps(
            {
                "capture_file": str(capture_file),
                "run_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            },
            indent=2,
        )
    )
    return path


def cmd_compare(
    capture_file: Path,
    profile_names: list[str],
    *,
    config_path: Path | None = None,
    manifest_dir: Path | None = None,
    output=None,
) -> int:
    """Run compare and return exit code (1 when all profiles fail, else 0)."""
    cfg = config_path or CONFIG_FILE
    mdir = manifest_dir or Path(".")
    out = output or sys.stdout

    if not capture_file.exists():
        print(f"error: capture file not found: {capture_file}", file=sys.stderr)
        return 1

    try:
        capture = json.loads(capture_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"error: could not read capture file: {exc}", file=sys.stderr)
        return 1

    request_body = capture.get("request", {})
    if not request_body:
        print("error: capture file has no 'request' field", file=sys.stderr)
        return 1

    results = asyncio.run(_run_all_profiles(profile_names, request_body, cfg))

    _print_table(results, output=out)

    manifest_path = _write_manifest(results, capture_file, mdir)
    print(f"\nManifest written to: {manifest_path}", file=out)

    return 1 if all(r["status"] == "FAILED" for r in results) else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ccproxy",
        description="claude-proxy CLI tools.",
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

    cmp_p = sub.add_parser(
        "compare",
        help="Replay a captured request across multiple profiles and compare results",
    )
    cmp_p.add_argument("capture_file", type=Path, help="Path to capture JSON file")
    cmp_p.add_argument(
        "--profiles",
        required=True,
        help="Comma-separated profile names (e.g. profile-a,profile-b)",
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
    elif args.command == "compare":
        profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
        sys.exit(cmd_compare(args.capture_file, profiles))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
