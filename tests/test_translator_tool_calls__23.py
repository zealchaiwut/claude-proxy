"""Tests for issue #23: translate OpenAI tool_calls to Anthropic tool_use in from_openai_response."""
import logging

import pytest

from schemas.anthropic import MessagesResponse, TextBlock, ToolUseBlock
from schemas.openai import ChatMessage, ChatResponse, Choice, Usage
from services.translator import from_openai_response


def _resp(
    *,
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
) -> ChatResponse:
    msg_kwargs = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg_kwargs["tool_calls"] = tool_calls
    return ChatResponse(
        choices=[
            Choice(
                message=ChatMessage(**msg_kwargs),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


# AC: single tool call → one tool_use block, stop_reason=tool_use
def test_single_tool_call_produces_tool_use_block():
    resp = _resp(
        tool_calls=[
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location":"NYC"}'},
            }
        ],
        finish_reason="tool_calls",
    )
    result = from_openai_response(resp)
    assert isinstance(result, MessagesResponse)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.id == "call_abc"
    assert block.name == "get_weather"
    assert block.input == {"location": "NYC"}


# AC: multiple tool calls → multiple tool_use blocks in same order
def test_multiple_tool_calls_preserves_order():
    resp = _resp(
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "func_a", "arguments": '{"x":1}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "func_b", "arguments": '{"y":2}'},
            },
        ],
        finish_reason="tool_calls",
    )
    result = from_openai_response(resp)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 2
    assert isinstance(result.content[0], ToolUseBlock)
    assert result.content[0].id == "call_1"
    assert result.content[0].name == "func_a"
    assert result.content[0].input == {"x": 1}
    assert isinstance(result.content[1], ToolUseBlock)
    assert result.content[1].id == "call_2"
    assert result.content[1].name == "func_b"
    assert result.content[1].input == {"y": 2}


# AC: non-empty content alongside tool calls → text block first, then tool_use
def test_text_content_with_tool_call_text_block_comes_first():
    resp = _resp(
        content="Sure, I'll look that up.",
        tool_calls=[
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"weather"}'},
            }
        ],
        finish_reason="tool_calls",
    )
    result = from_openai_response(resp)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 2
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == "Sure, I'll look that up."
    assert isinstance(result.content[1], ToolUseBlock)
    assert result.content[1].id == "call_xyz"


# AC: malformed JSON in function.arguments → input={}, no exception, warning logged
def test_malformed_arguments_json_sets_empty_input(caplog):
    resp = _resp(
        tool_calls=[
            {
                "id": "call_bad",
                "type": "function",
                "function": {"name": "broken_func", "arguments": "{invalid json"},
            }
        ],
        finish_reason="tool_calls",
    )
    with caplog.at_level(logging.WARNING, logger="services.translator"):
        result = from_openai_response(resp)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, ToolUseBlock)
    assert block.input == {}
    assert any("malformed" in r.message.lower() or "invalid" in r.message.lower() for r in caplog.records)


# AC: no tool_calls → behavior unchanged from before this feature
def test_no_tool_calls_unchanged_behavior():
    resp = _resp(content="Hello there.", finish_reason="stop")
    result = from_openai_response(resp)
    assert result.stop_reason == "end_turn"
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == "Hello there."


# AC: no tool_calls, empty content → unchanged (text block with empty string)
def test_no_tool_calls_empty_content_unchanged():
    resp = _resp(content=None, finish_reason="stop")
    result = from_openai_response(resp)
    assert result.stop_reason == "end_turn"
    assert len(result.content) == 1
    assert isinstance(result.content[0], TextBlock)
    assert result.content[0].text == ""
