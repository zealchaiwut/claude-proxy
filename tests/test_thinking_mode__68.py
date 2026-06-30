"""Tests for extended-thinking handling on the OpenAI translation path.

Covers:
  - to_openai_request thinking_mode: "disabled" / "forward" / "strip"
  - small-max_tokens 400-trigger avoidance (thinking disabled regardless)
  - response-side guard: upstream reasoning/reasoning_content never becomes
    Anthropic text content (non-streaming) or text_delta (streaming).
"""

import asyncio

from schemas.anthropic import MessagesRequest, TextBlock
from schemas.openai import ChatResponse
from services.openai_sse_consumer import (
    ContentEvent,
    ToolCallStartEvent,
    consume_openai_sse_stream,
)
from services.translator import from_openai_response, to_openai_request


def _req(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


# --- to_openai_request thinking_mode ---------------------------------------


def test_disabled_mode_emits_thinking_disabled():
    req = _req()
    result = to_openai_request(req, model="gpt-4o", thinking_mode="disabled")
    assert result.thinking == {"type": "disabled"}


def test_disabled_is_default():
    req = _req()
    result = to_openai_request(req, model="gpt-4o")
    assert result.thinking == {"type": "disabled"}


def test_forward_mode_passes_client_thinking_through():
    client_thinking = {"type": "enabled", "budget_tokens": 1024}
    req = _req(thinking=client_thinking)
    result = to_openai_request(req, model="gpt-4o", thinking_mode="forward")
    assert result.thinking == client_thinking


def test_forward_mode_omits_when_client_thinking_none():
    req = _req(thinking=None)
    result = to_openai_request(req, model="gpt-4o", thinking_mode="forward")
    assert "thinking" not in result.model_dump()


def test_strip_mode_emits_no_thinking_field():
    req = _req(thinking={"type": "enabled", "budget_tokens": 1024})
    result = to_openai_request(req, model="gpt-4o", thinking_mode="strip")
    assert "thinking" not in result.model_dump()


def test_small_max_tokens_disabled_avoids_400():
    # A small max_tokens that would trip "max_tokens must be greater than
    # thinking.budget_tokens" if the upstream defaulted thinking ON.
    req = _req(max_tokens=16)
    result = to_openai_request(req, model="gpt-4o", thinking_mode="disabled")
    assert result.max_tokens == 16
    assert result.thinking == {"type": "disabled"}
    # Serializes into the outgoing upstream JSON.
    assert result.model_dump()["thinking"] == {"type": "disabled"}


# --- response-side guard (non-streaming) -----------------------------------


def test_from_openai_response_drops_reasoning_fields():
    resp = ChatResponse(
        **{
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "real answer",
                        "reasoning": "secret chain of thought",
                        "reasoning_content": "more secret reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    )
    out = from_openai_response(resp)
    texts = [b.text for b in out.content if isinstance(b, TextBlock)]
    assert texts == ["real answer"]
    assert "secret" not in "".join(texts)
    assert "reasoning" not in "".join(texts)


def test_from_openai_response_keeps_tool_calls_drops_reasoning():
    resp = ChatResponse(
        **{
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "thinking...",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    )
    out = from_openai_response(resp)
    assert out.stop_reason == "tool_use"
    types = [getattr(b, "type", None) for b in out.content]
    assert "tool_use" in types
    text = "".join(b.text for b in out.content if isinstance(b, TextBlock))
    assert "thinking" not in text


# --- response-side guard (streaming SSE) -----------------------------------


def _collect(byte_chunks):
    async def _gen():
        for c in byte_chunks:
            yield c

    async def _run():
        return [ev async for ev in consume_openai_sse_stream(_gen())]

    return asyncio.run(_run())


def test_sse_consumer_drops_reasoning_delta_keeps_content():
    frames = [
        b'data: {"choices":[{"delta":{"reasoning":"secret"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"more"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    events = _collect(frames)
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].text == "hello"
    # No reasoning text ever surfaced as a ContentEvent.
    assert all("secret" not in e.text and "more" not in e.text for e in content_events)


def test_sse_consumer_drops_reasoning_keeps_tool_calls():
    frames = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        b'"function":{"name":"get_weather","arguments":""}}]}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    events = _collect(frames)
    assert any(isinstance(e, ToolCallStartEvent) for e in events)
    assert not any(isinstance(e, ContentEvent) for e in events)
