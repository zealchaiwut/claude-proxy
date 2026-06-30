"""Tests for issue #10: from_openai_response translator in services/translator.py."""
from __future__ import annotations

import pytest

from schemas.openai import ChatResponse, Choice, ChatMessage, Usage as OpenAIUsage
from services.translator import from_openai_response


def _make_openai_response(
    content: str = "Hello!",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    model: str = "gpt-4o",
) -> ChatResponse:
    return ChatResponse(
        id="chatcmpl-test123",
        object="chat.completion",
        model=model,
        choices=[Choice(message=ChatMessage(role="assistant", content=content), finish_reason=finish_reason)],
        usage=OpenAIUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens),
    )


# --- AC: finish_reason mapping ---

def test_stop_maps_to_end_turn():
    """AC: finish_reason='stop' → stop_reason='end_turn'."""
    resp = from_openai_response(_make_openai_response(finish_reason="stop"))
    assert resp.stop_reason == "end_turn"


def test_length_maps_to_max_tokens():
    """AC: finish_reason='length' → stop_reason='max_tokens'."""
    resp = from_openai_response(_make_openai_response(finish_reason="length"))
    assert resp.stop_reason == "max_tokens"


def test_tool_calls_maps_to_tool_use():
    """AC: finish_reason='tool_calls' → stop_reason='tool_use'."""
    resp = from_openai_response(_make_openai_response(finish_reason="tool_calls"))
    assert resp.stop_reason == "tool_use"


def test_unknown_finish_reason_maps_to_end_turn():
    """AC: unknown finish_reason (e.g. 'content_filter') → stop_reason='end_turn', no exception."""
    resp = from_openai_response(_make_openai_response(finish_reason="content_filter"))
    assert resp.stop_reason == "end_turn"


# --- AC: content block ---

def test_assistant_text_in_text_block():
    """AC: assistant turn text placed in a single Anthropic text content block."""
    resp = from_openai_response(_make_openai_response(content="World"))
    assert len(resp.content) == 1
    block = resp.content[0]
    assert block.type == "text"
    assert block.text == "World"


# --- AC: usage mapping ---

def test_prompt_tokens_maps_to_input_tokens():
    """AC: usage.prompt_tokens → input_tokens."""
    resp = from_openai_response(_make_openai_response(prompt_tokens=42, completion_tokens=7))
    assert resp.usage.input_tokens == 42


def test_completion_tokens_maps_to_output_tokens():
    """AC: usage.completion_tokens → output_tokens."""
    resp = from_openai_response(_make_openai_response(prompt_tokens=5, completion_tokens=99))
    assert resp.usage.output_tokens == 99


# --- AC: synthesised metadata fields ---

def test_role_is_assistant():
    """AC: returned object has role='assistant'."""
    resp = from_openai_response(_make_openai_response())
    assert resp.role == "assistant"


def test_type_is_message():
    """AC: returned object has type='message'."""
    resp = from_openai_response(_make_openai_response())
    assert resp.type == "message"


def test_id_is_non_empty_string():
    """AC: returned object has a non-empty id string."""
    resp = from_openai_response(_make_openai_response())
    assert isinstance(resp.id, str)
    assert len(resp.id) > 0


def test_model_is_non_empty_string():
    """AC: returned object has a non-empty model string."""
    resp = from_openai_response(_make_openai_response(model="gpt-4o"))
    assert isinstance(resp.model, str)
    assert len(resp.model) > 0
