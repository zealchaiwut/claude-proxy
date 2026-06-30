"""Token and cost accounting for request log records (issue #41)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PricingConfig:
    input_per_mtok: float
    output_per_mtok: float


def compute_est_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: PricingConfig | None,
) -> float | None:
    """Return estimated cost in USD, or None when no pricing is configured."""
    if pricing is None:
        return None
    return (
        (input_tokens / 1_000_000) * pricing.input_per_mtok
        + (output_tokens / 1_000_000) * pricing.output_per_mtok
    )


def extract_usage_from_response(response_json: dict[str, Any]) -> tuple[int, int] | None:
    """Return (input_tokens, output_tokens) from upstream response JSON, or None."""
    usage = response_json.get("usage")
    if not usage or not isinstance(usage, dict):
        return None
    it = usage.get("input_tokens")
    ot = usage.get("output_tokens")
    if it is None or ot is None:
        return None
    return int(it), int(ot)


def count_input_tokens(body_json: dict[str, Any]) -> int:
    """Estimate input token count from request body (total chars / 4 heuristic)."""
    messages = body_json.get("messages", [])
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
    return max(1, total_chars // 4)


def count_output_tokens(text: str) -> int:
    """Count output tokens using the real tokenizer (issue #54)."""
    from services.tokenizer import count_text_tokens
    count = count_text_tokens(text)
    return count if count > 0 else 1


def extract_upstream_usage_from_sse(data: bytes) -> tuple[int, int] | None:
    """Return (input_tokens, output_tokens) only if upstream SSE reported them.

    Unlike parse_anthropic_sse_usage, this never falls back to heuristics —
    returns None when no upstream usage events appear in the stream.
    Used for drift computation: drift = proxy_estimate - upstream_reported.
    """
    import json as _json

    input_tokens = 0
    output_tokens = 0
    has_input = False
    has_output = False

    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        try:
            payload = _json.loads(raw)
        except (_json.JSONDecodeError, ValueError):
            continue

        ptype = payload.get("type")
        if ptype == "message_start":
            msg = payload.get("message", {})
            usage = msg.get("usage", {})
            it = usage.get("input_tokens")
            if it:
                input_tokens = int(it)
                has_input = True
        elif ptype == "message_delta":
            usage = payload.get("usage", {})
            ot = usage.get("output_tokens")
            if ot:
                output_tokens = int(ot)
                has_output = True

    if has_input or has_output:
        return input_tokens, output_tokens
    return None


def extract_text_from_sse(data: bytes) -> str:
    """Extract accumulated text content from buffered Anthropic SSE text_delta frames."""
    import json as _json

    text = ""
    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        try:
            payload = _json.loads(raw)
        except (_json.JSONDecodeError, ValueError):
            continue
        if payload.get("type") == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
    return text


def parse_anthropic_sse_usage(
    data: bytes,
    body_json: dict[str, Any],
) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from buffered Anthropic SSE frames.

    Falls back to heuristic counts when upstream usage events are absent.
    """
    import json as _json

    input_tokens = 0
    output_tokens = 0
    accumulated_text = ""
    has_output_usage = False

    for line in data.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        try:
            payload = _json.loads(raw)
        except (_json.JSONDecodeError, ValueError):
            continue

        ptype = payload.get("type")
        if ptype == "message_start":
            msg = payload.get("message", {})
            usage = msg.get("usage", {})
            it = usage.get("input_tokens", 0)
            if it:
                input_tokens = int(it)
        elif ptype == "message_delta":
            usage = payload.get("usage", {})
            ot = usage.get("output_tokens", 0)
            if ot:
                output_tokens = int(ot)
                has_output_usage = True
        elif ptype == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                accumulated_text += delta.get("text", "")

    if not input_tokens:
        input_tokens = count_input_tokens(body_json)
    if not has_output_usage:
        output_tokens = count_output_tokens(accumulated_text)

    return input_tokens, output_tokens
