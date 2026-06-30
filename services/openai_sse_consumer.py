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


async def consume_openai_sse_stream(
    byte_stream: AsyncIterator[bytes],
) -> AsyncIterator[ContentEvent | FinishEvent | UsageEvent]:
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

            finish_reason = choice.get("finish_reason")
            if finish_reason:
                yield FinishEvent(reason=finish_reason)
