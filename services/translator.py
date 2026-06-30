from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

from schemas.anthropic import MessagesRequest, MessagesResponse, TextBlock, ToolUseBlock
from schemas.anthropic import Usage as AnthropicUsage

_log = logging.getLogger(__name__)
from schemas.openai import ChatRequest, ChatResponse
from services.openai_sse_consumer import (
    ContentEvent,
    FinishEvent,
    UsageEvent,
    consume_openai_sse_stream,
)
from services.sse import anthropic_sse_stream


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


def _map_tools(anthropic_tools: list) -> list[dict]:
    result = []
    for tool in anthropic_tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("input_schema", {})
        else:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            parameters = getattr(tool, "input_schema", {})
        result.append({"type": "function", "function": {"name": name, "description": description, "parameters": parameters}})
    return result


def _map_tool_choice(anthropic_choice: Any) -> Any:
    if anthropic_choice == "auto":
        return "auto"
    if anthropic_choice == "any":
        return "required"
    if isinstance(anthropic_choice, dict) and anthropic_choice.get("type") == "tool":
        return {"type": "function", "function": {"name": anthropic_choice["name"]}}
    return anthropic_choice


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "")


def _block_get(block: Any, attr: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(attr, default)
    return getattr(block, attr, default)


def _tool_result_content_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _extract_text(content)
    return json.dumps(content)


def to_openai_request(anthropic_req: MessagesRequest, model: str) -> ChatRequest:
    messages: list[dict] = []

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
            raw_content = turn.get("content", "")
        else:
            role = getattr(turn, "role", "user")
            raw_content = getattr(turn, "content", "")

        if isinstance(raw_content, list):
            if role == "assistant":
                tool_use_blocks = [b for b in raw_content if _block_type(b) == "tool_use"]
                if tool_use_blocks:
                    tool_calls = [
                        {
                            "id": _block_get(b, "id", ""),
                            "type": "function",
                            "function": {
                                "name": _block_get(b, "name", ""),
                                "arguments": json.dumps(_block_get(b, "input", {}) or {}),
                            },
                        }
                        for b in tool_use_blocks
                    ]
                    msg: dict = {"role": "assistant", "tool_calls": tool_calls}
                    text_str = _extract_text(raw_content)
                    if text_str:
                        msg["content"] = text_str
                    messages.append(msg)
                    continue
            elif role == "user":
                tool_result_blocks = [b for b in raw_content if _block_type(b) == "tool_result"]
                if tool_result_blocks:
                    for b in tool_result_blocks:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": _block_get(b, "tool_use_id", ""),
                            "content": _tool_result_content_str(_block_get(b, "content", "")),
                        })
                    other_text = _extract_text(
                        [b for b in raw_content if _block_type(b) != "tool_result"]
                    )
                    if other_text:
                        messages.append({"role": "user", "content": other_text})
                    continue

        messages.append({"role": role, "content": _extract_text(raw_content)})

    oai_tools = _map_tools(anthropic_req.tools) if anthropic_req.tools else None
    oai_tool_choice = _map_tool_choice(anthropic_req.tool_choice) if anthropic_req.tool_choice is not None else None

    return ChatRequest(
        model=model,
        messages=messages,
        max_tokens=anthropic_req.max_tokens,
        tools=oai_tools,
        tool_choice=oai_tool_choice,
    )


_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


def _sse_frame(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def stream_to_anthropic_sse(
    event_stream: AsyncIterator,
    *,
    model: str,
    message_id: str,
) -> AsyncIterator[str]:
    """Bridge normalized OpenAI SSE events to Anthropic SSE frames.

    Consumes an async iterator that yields duck-typed event objects:
      - .text (str)   → ContentEvent: a text delta
      - .reason (str) → FinishEvent: the finish reason
      - .usage (dict) → UsageEvent: token usage from upstream

    Buffers all events, then replays text deltas through anthropic_sse_stream
    so that stop_reason and usage — which arrive at stream end — are available
    when the emitter emits message_start and message_delta.
    """
    text_parts: list[str] = []
    stop_reason = "end_turn"
    upstream_usage: dict[str, Any] | None = None

    async for event in event_stream:
        if hasattr(event, "text"):
            text_parts.append(event.text)
        elif hasattr(event, "reason"):
            stop_reason = _FINISH_REASON_MAP.get(event.reason, "end_turn")
        elif hasattr(event, "usage"):
            upstream_usage = event.usage

    if upstream_usage is not None:
        output_tokens = upstream_usage.get("completion_tokens", 0) or 0
    else:
        # Fallback: count words in accumulated text as a rough token estimate.
        accumulated = "".join(text_parts)
        output_tokens = max(1, len(accumulated.split())) if accumulated else 0

    usage = {"input_tokens": 0, "output_tokens": output_tokens}

    async def _replay():
        for text in text_parts:
            yield text

    async for frame in anthropic_sse_stream(
        _replay(),
        stop_reason=stop_reason,
        usage=usage,
        model=model,
        message_id=message_id,
    ):
        yield frame


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

    tool_calls = getattr(choice.message, "tool_calls", None)
    if tool_calls:
        stop_reason = "tool_use"
        content: list[Any] = []
        if text:
            content.append(TextBlock(type="text", text=text))
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            fn = tc.get("function", {})
            name = fn.get("name", "")
            arguments_str = fn.get("arguments", "{}")
            try:
                input_data = json.loads(arguments_str)
            except (json.JSONDecodeError, TypeError, ValueError):
                _log.warning(
                    "tool_call %r has malformed arguments JSON; defaulting input to {}",
                    tc_id,
                )
                input_data = {}
            content.append(ToolUseBlock(type="tool_use", id=tc_id, name=name, input=input_data))
    else:
        content = [TextBlock(type="text", text=text)]

    return MessagesResponse(
        id=resp_id,
        type="message",
        role="assistant",
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=usage,
    )
