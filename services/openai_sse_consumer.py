"""OpenAI SSE streaming consumer for /chat/completions byte streams."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass
class ContentEvent:
    text: str


@dataclass
class FinishEvent:
    reason: str


@dataclass
class UsageEvent:
    usage: dict[str, Any]


@dataclass
class ToolCallStartEvent:
    """First fragment for a streamed tool call — carries the OAI index, id, and function name."""
    index: int
    id: str
    name: str


@dataclass
class ToolCallDeltaEvent:
    """Subsequent fragment — carries incremental function.arguments JSON."""
    index: int
    partial_json: str


async def consume_openai_sse_stream(
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[ContentEvent | FinishEvent | UsageEvent | ToolCallStartEvent | ToolCallDeltaEvent]:
    """Parse an OpenAI SSE byte stream and yield typed events.

    Handles partial lines split across read boundaries by accumulating a buffer
    until a complete `\\n`-terminated line is available.
    """
    buf = b""

    async for chunk in byte_stream:
        buf += chunk
        while b"\n" in buf:
            line_bytes, buf = buf.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")

            if not line or line.startswith(":"):
                continue

            if not line.startswith("data:"):
                continue

            data = line[len("data:"):].strip()

            if data == "[DONE]":
                return

            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue

            if isinstance(payload.get("usage"), dict):
                yield UsageEvent(usage=payload["usage"])

            choices = payload.get("choices")
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta", {})

            content = delta.get("content")
            if content:
                yield ContentEvent(text=content)

            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    idx = tc.get("index", 0)
                    tc_id = tc.get("id")
                    fn = tc.get("function", {})
                    name = fn.get("name")
                    args = fn.get("arguments", "")
                    if tc_id is not None and name is not None:
                        yield ToolCallStartEvent(index=idx, id=tc_id, name=name)
                    if args:
                        yield ToolCallDeltaEvent(index=idx, partial_json=args)

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                yield FinishEvent(reason=finish_reason)
