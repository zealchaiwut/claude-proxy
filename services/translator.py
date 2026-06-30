from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

from schemas.anthropic import MessagesRequest, MessagesResponse, TextBlock, ToolUseBlock
from schemas.anthropic import Usage as AnthropicUsage
from schemas.openai import ChatRequest, ChatResponse
from services.openai_sse_consumer import (
    ContentEvent,
    FinishEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    UsageEvent,
    consume_openai_sse_stream,
)
from services.sse import anthropic_sse_stream

_log = logging.getLogger(__name__)

# Provider hints that support upstream prompt caching mechanisms.
_CACHE_CAPABLE_PROVIDERS: frozenset[str] = frozenset({"openai", "deepseek", "together", "fireworks"})


def _get_blocks(content: Any) -> list[Any]:
    """Return content as a list of block dicts/objects (empty if content is a plain string)."""
    if isinstance(content, list):
        return content
    return []


def _block_type(block: Any) -> str:
    if isinstance(block, dict):
        return block.get("type", "")
    return getattr(block, "type", "")


def _has_tool_use(content: Any) -> bool:
    return any(_block_type(b) == "tool_use" for b in _get_blocks(content))


def _has_tool_result(content: Any) -> bool:
    return any(_block_type(b) == "tool_result" for b in _get_blocks(content))


def _tool_result_content_to_str(value: Any) -> str:
    """Convert tool_result content to a string for the OpenAI tool message."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "type") and item.type == "text":
                parts.append(item.text)
            else:
                parts.append(json.dumps(item) if isinstance(item, dict) else str(item))
        return "".join(parts)
    return json.dumps(value)


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


def _has_cache_control(block: Any) -> bool:
    """Return True if a block (dict or TextBlock) carries a cache_control marker."""
    if isinstance(block, dict):
        return "cache_control" in block
    extras = getattr(block, "model_extra", {}) or {}
    return "cache_control" in extras


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


def to_openai_request(
    anthropic_req: MessagesRequest,
    model: str,
    *,
    prompt_cache: str = "none",
    cache_provider_hint: str | None = None,
    thinking_mode: str = "disabled",
) -> ChatRequest:
    messages: list[dict[str, str]] = []
    has_cache_marker = False

    if anthropic_req.system is not None:
        if isinstance(anthropic_req.system, str):
            system_text = anthropic_req.system
        else:
            system_text = "".join(
                block.text for block in anthropic_req.system if isinstance(block, TextBlock)
            )
            if any(_has_cache_control(b) for b in anthropic_req.system):
                has_cache_marker = True
        messages.append({"role": "system", "content": system_text})

    for turn in anthropic_req.messages:
        if isinstance(turn, dict):
            role = turn.get("role", "user")
            raw_content = turn.get("content", "")
        else:
            role = getattr(turn, "role", "user")
            raw_content = getattr(turn, "content", "")

        if role == "assistant" and _has_tool_use(raw_content):
            blocks = _get_blocks(raw_content)
            tool_calls = []
            text_parts = []
            for block in blocks:
                btype = _block_type(block)
                if btype == "tool_use":
                    if isinstance(block, dict):
                        bid, bname, binput = block["id"], block["name"], block.get("input", {})
                    else:
                        bid, bname, binput = block.id, block.name, block.input
                    tool_calls.append({
                        "id": bid,
                        "type": "function",
                        "function": {
                            "name": bname,
                            "arguments": json.dumps(binput),
                        },
                    })
                elif btype == "text":
                    text_parts.append(block["text"] if isinstance(block, dict) else block.text)
            msg: dict[str, Any] = {"role": "assistant", "tool_calls": tool_calls}
            if text_parts:
                msg["content"] = "".join(text_parts)
            messages.append(msg)

        elif role == "user" and _has_tool_result(raw_content):
            blocks = _get_blocks(raw_content)
            text_parts = []
            for block in blocks:
                btype = _block_type(block)
                if btype == "tool_result":
                    if isinstance(block, dict):
                        tid = block["tool_use_id"]
                        tcontent = _tool_result_content_to_str(block.get("content"))
                    else:
                        tid = block.tool_use_id
                        tcontent = _tool_result_content_to_str(getattr(block, "content", None))
                    messages.append({"role": "tool", "tool_call_id": tid, "content": tcontent})
                elif btype == "text":
                    text_parts.append(block["text"] if isinstance(block, dict) else block.text)
            if text_parts:
                messages.append({"role": "user", "content": "".join(text_parts)})

        else:
            messages.append({"role": role, "content": _extract_text(raw_content)})

    if anthropic_req.tools:
        if any(
            isinstance(t, dict) and "cache_control" in t
            for t in anthropic_req.tools
        ):
            has_cache_marker = True
        oai_tools = _map_tools(anthropic_req.tools)
    else:
        oai_tools = None
    oai_tool_choice = _map_tool_choice(anthropic_req.tool_choice) if anthropic_req.tool_choice is not None else None

    extra_kwargs: dict[str, Any] = {}
    if (
        prompt_cache == "auto"
        and cache_provider_hint in _CACHE_CAPABLE_PROVIDERS
        and has_cache_marker
    ):
        extra_kwargs["cache_control"] = {"type": "ephemeral"}

    req = ChatRequest(
        model=model,
        messages=messages,
        max_tokens=anthropic_req.max_tokens,
        tools=oai_tools,
        tool_choice=oai_tool_choice,
        **extra_kwargs,
    )

    # Extended-thinking handling. ChatRequest has extra="allow", so assigning
    # req.thinking serializes into the upstream JSON.
    #   "disabled" (default): force thinking off so the upstream never enables it
    #                         with a budget that can exceed max_tokens.
    #   "forward": pass the client's thinking block through unchanged when present.
    #   "strip": send no thinking field at all (for upstreams that reject it).
    if thinking_mode == "disabled":
        req.thinking = {"type": "disabled"}
    elif thinking_mode == "forward":
        if anthropic_req.thinking is not None:
            req.thinking = anthropic_req.thinking
    # "strip" (or anything else): leave thinking unset.

    return req


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

    _DONE = object()
    event_iter = consume_openai_sse_stream(byte_stream).__aiter__()

    async def _next():
        try:
            return await event_iter.__anext__()
        except StopAsyncIteration:
            return _DONE

    # Track content blocks: next index to assign, and the index of the currently open block.
    next_block_idx = 0
    open_block_idx: int | None = None

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
            if open_block_idx is None:
                yield _sse_frame("content_block_start", {
                    "type": "content_block_start",
                    "index": next_block_idx,
                    "content_block": {"type": "text", "text": ""},
                })
                open_block_idx = next_block_idx
                next_block_idx += 1
            yield _sse_frame("content_block_delta", {
                "type": "content_block_delta",
                "index": open_block_idx,
                "delta": {"type": "text_delta", "text": result.text},
            })
        elif isinstance(result, ToolCallStartEvent):
            if open_block_idx is not None:
                yield _sse_frame("content_block_stop", {
                    "type": "content_block_stop",
                    "index": open_block_idx,
                })
                open_block_idx = None
            yield _sse_frame("content_block_start", {
                "type": "content_block_start",
                "index": next_block_idx,
                "content_block": {
                    "type": "tool_use",
                    "id": result.id,
                    "name": result.name,
                    "input": {},
                },
            })
            open_block_idx = next_block_idx
            next_block_idx += 1
        elif isinstance(result, ToolCallDeltaEvent):
            if open_block_idx is not None:
                yield _sse_frame("content_block_delta", {
                    "type": "content_block_delta",
                    "index": open_block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": result.partial_json},
                })
        elif isinstance(result, FinishEvent):
            stop_reason = _FINISH_REASON_MAP.get(result.reason, "end_turn")
        elif isinstance(result, UsageEvent):
            input_tokens = result.usage.get("prompt_tokens", 0) or 0
            output_tokens = result.usage.get("completion_tokens", 0) or 0
        task = asyncio.ensure_future(_next())

    if open_block_idx is not None:
        yield _sse_frame("content_block_stop", {"type": "content_block_stop", "index": open_block_idx})

    yield _sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse_frame("message_stop", {"type": "message_stop"})


def from_openai_response(openai_resp: ChatResponse) -> MessagesResponse:
    """Convert a non-streaming OpenAI ChatCompletion response to an Anthropic MessagesResponse."""
    choice = openai_resp.choices[0]
    # Only real content becomes Anthropic text. Any upstream reasoning /
    # reasoning_content / thinking field lands in ChatMessage's extra (extra=
    # "allow") and is deliberately NOT read here, so it never leaks as a text block.
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
