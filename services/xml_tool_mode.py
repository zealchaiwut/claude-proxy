"""XML tool-call mode for non-function-calling upstreams (issue #26).

When CCPROXY_TOOL_MODE=xml, tool definitions are injected into the system
prompt as an XML spec and XML tool-call blocks in the upstream response are
parsed back into Anthropic tool_use blocks.  Native mode is unaffected.
"""
from __future__ import annotations

import json
import re
import uuid
import xml.etree.ElementTree as ET
from typing import Any, AsyncIterator

from schemas.anthropic import ToolUseBlock

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_XML_PREAMBLE = (
    "You have access to the following tools. "
    "To call a tool, respond with a <tool_call> block as shown below.\n\n"
)

_XML_EXAMPLE = (
    "\n\nTo call a tool:\n"
    "<tool_call>\n"
    "<name>TOOL_NAME</name>\n"
    "<id>UNIQUE_ID</id>\n"
    "<input>{\"param\": \"value\"}</input>\n"
    "</tool_call>"
)


def _sse_frame(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def build_xml_system_prompt(system: str | None, tools: list[Any]) -> str:
    """Return a system prompt string with XML tool definitions prepended."""
    parts: list[str] = []
    if system:
        parts.append(system)

    tool_lines = [_XML_PREAMBLE + "<tools>"]
    for tool in tools:
        if isinstance(tool, dict):
            name = tool.get("name", "")
            description = tool.get("description", "")
            input_schema = json.dumps(tool.get("input_schema", {}))
        else:
            name = getattr(tool, "name", "")
            description = getattr(tool, "description", "")
            raw = getattr(tool, "input_schema", {})
            input_schema = json.dumps(raw if isinstance(raw, dict) else {})
        tool_lines.append(
            f"<tool>\n<name>{name}</name>\n<description>{description}</description>\n"
            f"<input_schema>{input_schema}</input_schema>\n</tool>"
        )
    tool_lines.append("</tools>" + _XML_EXAMPLE)
    parts.append("\n".join(tool_lines))

    return "\n\n".join(parts)


def parse_xml_tool_calls(text: str) -> tuple[str, list[ToolUseBlock]]:
    """Extract <tool_call> blocks from text; return (cleaned_text, tool_use_blocks).

    On any XML parse error, returns (original_text, []) for graceful fallback.
    """
    matches = list(_TOOL_CALL_RE.finditer(text))
    if not matches:
        return text, []

    tool_blocks: list[ToolUseBlock] = []
    cleaned = text

    for match in reversed(matches):
        inner = match.group(1)
        try:
            root = ET.fromstring(f"<tool_call>{inner}</tool_call>")
        except ET.ParseError:
            return text, []

        name_el = root.find("name")
        id_el = root.find("id")
        input_el = root.find("input")

        name = (name_el.text or "").strip() if name_el is not None else ""
        tool_id = (id_el.text or "").strip() if id_el is not None else f"call_{uuid.uuid4().hex[:8]}"
        input_text = (input_el.text or "{}").strip() if input_el is not None else "{}"

        try:
            input_data = json.loads(input_text)
        except (json.JSONDecodeError, ValueError):
            input_data = {}

        tool_blocks.insert(0, ToolUseBlock(type="tool_use", id=tool_id, name=name, input=input_data))
        cleaned = cleaned[: match.start()] + cleaned[match.end() :]

    return cleaned.strip(), tool_blocks


async def xml_buffered_sse(
    byte_stream: AsyncIterator[bytes],
    *,
    model: str,
    message_id: str | None = None,
) -> AsyncIterator[str]:
    """Buffer a full OpenAI SSE byte stream, parse XML tool calls, emit Anthropic SSE frames."""
    from services.openai_sse_consumer import ContentEvent, FinishEvent, UsageEvent, consume_openai_sse_stream
    from services.translator import _FINISH_REASON_MAP

    mid = message_id or f"msg_{uuid.uuid4().hex[:24]}"
    text_parts: list[str] = []
    stop_reason = "end_turn"
    input_tokens = 0
    output_tokens = 0

    async for event in consume_openai_sse_stream(byte_stream):
        if isinstance(event, ContentEvent):
            text_parts.append(event.text)
        elif isinstance(event, FinishEvent):
            stop_reason = _FINISH_REASON_MAP.get(event.reason, "end_turn")
        elif isinstance(event, UsageEvent):
            input_tokens = event.usage.get("prompt_tokens", 0) or 0
            output_tokens = event.usage.get("completion_tokens", 0) or 0

    full_text = "".join(text_parts)
    cleaned_text, tool_blocks = parse_xml_tool_calls(full_text)

    if tool_blocks:
        stop_reason = "tool_use"

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

    block_index = 0

    if cleaned_text:
        yield _sse_frame("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse_frame("content_block_delta", {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "text_delta", "text": cleaned_text},
        })
        yield _sse_frame("content_block_stop", {"type": "content_block_stop", "index": block_index})
        block_index += 1
    elif not tool_blocks:
        yield _sse_frame("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse_frame("content_block_stop", {"type": "content_block_stop", "index": block_index})
        block_index += 1

    for tool_block in tool_blocks:
        yield _sse_frame("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": "tool_use",
                "id": tool_block.id,
                "name": tool_block.name,
                "input": {},
            },
        })
        yield _sse_frame("content_block_delta", {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(tool_block.input)},
        })
        yield _sse_frame("content_block_stop", {"type": "content_block_stop", "index": block_index})
        block_index += 1

    yield _sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield _sse_frame("message_stop", {"type": "message_stop"})
