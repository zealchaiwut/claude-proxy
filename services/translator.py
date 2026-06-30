# M1 limitation: image and tool blocks (image, tool_use, tool_result) are silently skipped; full support deferred to M3.
from __future__ import annotations

from typing import Any

from schemas.anthropic import MessagesRequest, MessagesResponse, TextBlock
from schemas.anthropic import Usage as AnthropicUsage
from schemas.openai import ChatRequest, ChatResponse


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
