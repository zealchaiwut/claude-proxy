"""Anthropic Messages API SSE streaming event emitter (issue #16).

Pure async generator — no HTTP framework dependency.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterable, AsyncIterator


def _frame(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _input_tokens(usage: Any) -> int:
    if isinstance(usage, dict):
        return usage.get("input_tokens", 0)
    return getattr(usage, "input_tokens", 0)


def _output_tokens(usage: Any) -> int:
    if isinstance(usage, dict):
        return usage.get("output_tokens", 0)
    return getattr(usage, "output_tokens", 0)


_DONE = object()


async def anthropic_sse_stream(
    deltas: AsyncIterable[str],
    *,
    stop_reason: str,
    usage: Any,
    model: str,
    message_id: str,
    ping_interval: float | None = None,
) -> AsyncIterator[str]:
    """Yield Anthropic Messages streaming SSE frames for the given inputs.

    Event sequence:
      message_start → content_block_start → N×content_block_delta
      → content_block_stop → message_delta → message_stop

    If ping_interval (seconds) is set, ping frames are interspersed while
    waiting for the next delta.
    """
    yield _frame("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "usage": {"input_tokens": _input_tokens(usage)},
        },
    })
    yield _frame("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    delta_iter: AsyncIterator[str] = deltas.__aiter__()

    async def _next() -> Any:
        try:
            return await delta_iter.__anext__()
        except StopAsyncIteration:
            return _DONE

    if ping_interval is not None:
        task: asyncio.Future = asyncio.ensure_future(_next())
        while True:
            done, _ = await asyncio.wait({task}, timeout=ping_interval)
            if not done:
                yield _frame("ping", {})
                continue
            result = task.result()
            if result is _DONE:
                break
            yield _frame("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": result},
            })
            task = asyncio.ensure_future(_next())
    else:
        async for delta in deltas:
            yield _frame("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta},
            })

    yield _frame("content_block_stop", {
        "type": "content_block_stop",
        "index": 0,
    })
    yield _frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": _output_tokens(usage)},
    })
    yield _frame("message_stop", {"type": "message_stop"})
