"""Replay engine for the ccproxy replay command (issue #60).

Routes captured requests through the same profile resolution and translation
code path as the live proxy — no separate HTTP client, no bypass.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

import httpx

from profiles import ProfileRegistry, get_or_load_config
from services.cost_accounting import (
    compute_est_cost,
    count_input_tokens,
    count_output_tokens,
    extract_usage_from_response,
    parse_anthropic_sse_usage,
)


def load_capture(capture_path: Path) -> dict[str, Any]:
    """Load and validate a capture file. Raises ValueError on any problem."""
    try:
        raw = capture_path.read_text()
    except OSError as e:
        raise ValueError(f"Cannot open capture file '{capture_path}': {e}") from e

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Capture file '{capture_path}' is not valid JSON: {e}") from e

    if not isinstance(data, dict) or "request" not in data:
        raise ValueError(
            f"Capture file '{capture_path}' is malformed: missing 'request' key"
        )

    return data


def artifact_path(capture_path: Path, profile_name: str) -> Path:
    """Return the artifact path written adjacent to the capture file."""
    return capture_path.parent / f"{capture_path.stem}.replay-{profile_name}.json"


async def replay(
    capture_path: Path,
    profile_name: str,
    *,
    stream_override: bool | None = None,
    config_path: Path = Path("config.toml"),
    stdout: TextIO | None = None,
) -> int:
    """Execute a replay of a captured request. Returns exit code (0 = success)."""
    if stdout is None:
        stdout = sys.stdout

    try:
        capture = load_capture(capture_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    proxy_config, _ = get_or_load_config(config_path)
    if profile_name not in proxy_config.profiles:
        known = ", ".join(sorted(proxy_config.profiles)) or "(none)"
        print(
            f"error: profile '{profile_name}' not found in config (known: {known})",
            file=sys.stderr,
        )
        return 1

    registry = ProfileRegistry(proxy_config)
    kind, upstream_url, api_key, model, model_map = registry.resolve(profile_name)
    pricing = registry.get_pricing(profile_name)

    use_stream = (
        stream_override if stream_override is not None else capture.get("stream", False)
    )

    request_body: dict[str, Any] = {**capture["request"], "stream": use_stream}

    t0 = time.monotonic()

    if kind == "passthrough":
        response_data, prompt_tokens, completion_tokens = await _replay_passthrough(
            upstream_url, api_key, request_body, use_stream, stdout
        )
    elif kind == "openai":
        response_data, prompt_tokens, completion_tokens = await _replay_openai(
            upstream_url, api_key, model, model_map, request_body, use_stream, stdout
        )
    else:
        print(f"error: unsupported profile kind '{kind}'", file=sys.stderr)
        return 1

    latency_ms = int((time.monotonic() - t0) * 1000)
    est_cost = compute_est_cost(prompt_tokens, completion_tokens, pricing)

    out_path = artifact_path(capture_path, profile_name)
    out_path.write_text(
        json.dumps(
            {
                "response": response_data,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "est_cost": est_cost,
                "latency_ms": latency_ms,
            },
            indent=2,
        )
    )

    print(f"\n[artifact → {out_path}]", file=sys.stderr)
    return 0


def _passthrough_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
    return headers


async def _replay_passthrough(
    upstream_url: str,
    api_key: str | None,
    request_body: dict[str, Any],
    use_stream: bool,
    stdout: TextIO,
) -> tuple[Any, int, int]:
    """Forward the request to the passthrough upstream and return (response_data, input_tok, output_tok)."""
    headers = _passthrough_headers(api_key)
    body_bytes = json.dumps(request_body).encode()

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        if use_stream:
            chunks: list[bytes] = []
            async with client.stream(
                "POST",
                f"{upstream_url}/v1/messages",
                content=body_bytes,
                headers=headers,
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    stdout.write(chunk.decode("utf-8", errors="replace"))
                    stdout.flush()
                    chunks.append(chunk)
            raw = b"".join(chunks)
            prompt_tokens, completion_tokens = parse_anthropic_sse_usage(
                raw, request_body
            )
            return raw.decode("utf-8", errors="replace"), prompt_tokens, completion_tokens

        resp = await client.post(
            f"{upstream_url}/v1/messages",
            content=body_bytes,
            headers=headers,
        )
        text = resp.content.decode("utf-8", errors="replace")
        stdout.write(text)
        stdout.flush()

        try:
            resp_json: dict[str, Any] = json.loads(resp.content)
        except json.JSONDecodeError:
            resp_json = {}

        usage = extract_usage_from_response(resp_json)
        if usage:
            prompt_tokens, completion_tokens = usage
        else:
            prompt_tokens = count_input_tokens(request_body)
            output_text = "".join(
                b.get("text", "")
                for b in resp_json.get("content", [])
                if isinstance(b, dict) and b.get("type") == "text"
            )
            completion_tokens = count_output_tokens(output_text)

        return resp_json, prompt_tokens, completion_tokens


async def _replay_openai(
    upstream_url: str,
    api_key: str | None,
    model: str | None,
    model_map: dict[str, str],
    request_body: dict[str, Any],
    use_stream: bool,
    stdout: TextIO,
) -> tuple[Any, int, int]:
    """Translate to OpenAI format, forward, translate back; same path as the live proxy."""
    import os

    from schemas.anthropic import MessagesRequest
    from services.translator import (
        from_openai_response,
        live_stream_to_anthropic_sse,
        to_openai_request,
    )

    client_model = request_body.get("model", "")
    upstream_model = (
        model_map.get(client_model) or model or os.getenv("OPENAI_MODEL", "gpt-4o")
    )

    anthropic_req = MessagesRequest(**request_body)
    openai_req = to_openai_request(anthropic_req, model=upstream_model)

    openai_api_key = api_key or ""
    auth_headers = {
        "authorization": f"Bearer {openai_api_key}",
        "content-type": "application/json",
    }

    from schemas.openai import ChatRequest

    if use_stream:
        stream_req = ChatRequest(
            model=upstream_model,
            messages=openai_req.messages,
            max_tokens=openai_req.max_tokens,
            stream=True,
        )
        body_bytes = stream_req.model_dump_json().encode()

        chunks: list[bytes] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream(
                "POST",
                f"{upstream_url}/chat/completions",
                content=body_bytes,
                headers=auth_headers,
            ) as resp:
                async for frame in live_stream_to_anthropic_sse(resp.aiter_bytes(), model=upstream_model):
                    encoded = frame.encode() if isinstance(frame, str) else frame
                    stdout.write(encoded.decode("utf-8", errors="replace"))
                    stdout.flush()
                    chunks.append(encoded)

        raw = b"".join(chunks)
        prompt_tokens, completion_tokens = parse_anthropic_sse_usage(raw, request_body)
        return raw.decode("utf-8", errors="replace"), prompt_tokens, completion_tokens

    # Non-streaming
    body_bytes = openai_req.model_dump_json().encode()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.post(
            f"{upstream_url}/chat/completions",
            content=body_bytes,
            headers=auth_headers,
        )

    from schemas.openai import ChatResponse

    openai_resp = ChatResponse(**json.loads(resp.content))
    anthropic_resp = from_openai_response(openai_resp)
    resp_json = json.loads(anthropic_resp.model_dump_json())

    stdout.write(json.dumps(resp_json))
    stdout.flush()

    usage = extract_usage_from_response(resp_json)
    if usage:
        prompt_tokens, completion_tokens = usage
    else:
        prompt_tokens = count_input_tokens(request_body)
        output_text = "".join(
            b.get("text", "")
            for b in resp_json.get("content", [])
            if isinstance(b, dict) and b.get("type") == "text"
        )
        completion_tokens = count_output_tokens(output_text)

    return resp_json, prompt_tokens, completion_tokens
