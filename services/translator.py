# M1 limitation: image and tool blocks (image, tool_use, tool_result) are silently skipped; full support deferred to M3.
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator

from schemas.anthropic import MessagesRequest, MessagesResponse, TextBlock
from schemas.anthropic import Usage as AnthropicUsage
from schemas.openai import ChatRequest, ChatResponse
from services.openai_sse_consumer import (
    ContentEvent,
    FinishEvent,
    UsageEvent,
    consume_openai_sse_stream,
)


def _extract_text(content: Any) -> str:
    """Return text from a string, a single text block, or a list of content blocks (non-text skipped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, TextBlock):
                parts.append(block.text)
        return "".join(parts)
    if isinstance(content, TextBlock):
        return content.text
    return ""


def to_openai_request(anthropic_req: MessagesRequest, model: str) -> ChatRequest:
    messages: list[dict[str, str]] = []

    if anthropic_req.system is not None:
        if isinstance(anthropic_req.system, str):
            system_text = anthropic_req.system
        else:
            system_text = "".join(
                block.text for block in anthropic_req.system if isinstance(block, TextBlock)
            )
        messages.append({"role": "system", "content": system_text})

    for turn in anthropic_req.messages:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            content = _extract_text(turn.get("content", ""))
        else:
            role = getattr(turn, "role", "user")
            content = _extract_text(getattr(turn, "content", ""))
        messages.append({"role": role, "content": content})

    return ChatRequest(
        model=model,
        messages=messages,
        max_tokens=anthropic_req.max_tokens,
    )


_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _sse_frame(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def live_stream_to_anthropic_sse(
    byte_stream: AsyncIterator[bytes],
    *,
    model: str,
    message_id: str | None = None,
    ping_interval: float = 15.0,
) -> AsyncIterator[str]:
    """Translate an OpenAI SSE byte stream to Anthropic SSE frames with no buffering.

    Each content delta is forwarded immediately. Periodic `: ping` comments are
    emitted to keep the connection alive. Collects stop_reason and usage from
    downstream events and emits them in the final Anthropic frames.
    """
    mid = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0

    yield _sse_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": mid,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "usage": {"input_tokens": input_tokens},
        },
    })
    yield _sse_frame("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    _DONE = object()
    event_iter = consume_openai_sse_stream(byte_stream).__aiter__()

    async def _next():
        try:
            return await event_iter.__anext__()
        except StopAsyncIteration:
            return _DONE

    task: asyncio.Future = asyncio.ensure_future(_next())
    while True:
        done, _ = await asyncio.wait({task}, timeout=ping_interval)
        if not done:
            yield ": ping\n\n"
            continue
        result = task.result()
        if result is _DONE:
            break
        if isinstance(result, ContentEvent):
            yield _sse_frame("content_block_delta", {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": result.text},
            })
        elif isinstance(result, FinishEvent):
            stop_reason = _FINISH_REASON_MAP.get(result.reason, "end_turn")
        elif isinstance(result, UsageEvent):
            input_tokens = result.usage.get("prompt_tokens", 0) or 0
            output_tokens = result.usage.get("completion_tokens", 0) or 0
        task = asyncio.ensure_future(_next())

    yield _sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield _sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse_frame("message_stop", {"type": "message_stop"})


def from_openai_response(openai_resp: ChatResponse) -> MessagesResponse:
    """Convert a non-streaming OpenAI ChatCompletion response to an Anthropic MessagesResponse."""
    choice = openai_resp.choices[0]
    text = choice.message.content or ""
    stop_reason = _FINISH_REASON_MAP.get(choice.finish_reason or "", "end_turn")

    usage = AnthropicUsage(
        input_tokens=openai_resp.usage.prompt_tokens,
        output_tokens=openai_resp.usage.completion_tokens,
    )

    resp_id = getattr(openai_resp, "id", None) or "msg_translated"
    model = getattr(openai_resp, "model", None) or "unknown"

    return MessagesResponse(
        id=resp_id,
        type="message",
        role="assistant",
        model=model,
        content=[TextBlock(type="text", text=text)],
        stop_reason=stop_reason,
        usage=usage,
    )
