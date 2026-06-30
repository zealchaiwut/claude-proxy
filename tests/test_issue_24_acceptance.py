"""Acceptance criterion tests for issue #24: Anthropic tool_result → OpenAI tool messages.

Tests verify:
- AC1: assistant turn with tool_use → OpenAI assistant message with tool_calls
- AC2: tool_calls entry has required fields: id, type, function.name, function.arguments (JSON)
- AC3: tool_result blocks → discrete role:tool messages with matching tool_call_id and content
- AC4: tool message immediately follows assistant message with matching tool_calls
- AC5: tool_call_id exactly matches tool_use id (no drift/remapping)
- AC6: parallel tool_use/tool_result blocks handled correctly
- AC7: plain text turns unchanged
- AC8: pytest suite covers single and parallel round-trips with id matching
"""

import json
import pytest
from schemas.anthropic import MessagesRequest
from services.translator import to_openai_request


def _req(**kwargs) -> MessagesRequest:
    """Helper to build MessagesRequest with defaults."""
    defaults = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 256,
        "messages": [],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


# === AC1: Convert assistant tool_use to OpenAI tool_calls ===

def test_ac1_assistant_tool_use_becomes_tool_calls():
    """AC1: assistant turn with tool_use → OpenAI assistant message with tool_calls array."""
    req = _req(messages=[
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_001", "name": "get_weather", "input": {"location": "Paris"}},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    # Two messages: user + assistant
    assert len(result.messages) == 2
    asst = result.messages[1]
    assert asst["role"] == "assistant"
    assert "tool_calls" in asst
    assert isinstance(asst["tool_calls"], list)
    assert len(asst["tool_calls"]) == 1


# === AC2: tool_calls entry has required fields ===

def test_ac2_tool_calls_has_all_required_fields():
    """AC2: each tool_calls entry has id, type:function, function.name, function.arguments (JSON)."""
    req = _req(messages=[
        {"role": "user", "content": "Call get_weather."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_abc123", "name": "get_weather", "input": {"location": "Tokyo"}},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    tc = result.messages[1]["tool_calls"][0]

    # Required fields
    assert "id" in tc
    assert tc["id"] == "tu_abc123"
    assert "type" in tc
    assert tc["type"] == "function"
    assert "function" in tc
    assert "name" in tc["function"]
    assert tc["function"]["name"] == "get_weather"
    assert "arguments" in tc["function"]

    # arguments must be JSON string
    assert isinstance(tc["function"]["arguments"], str)
    args = json.loads(tc["function"]["arguments"])
    assert args == {"location": "Tokyo"}


# === AC3 & AC4: tool_result blocks become discrete tool messages ===

def test_ac3_ac4_tool_result_becomes_tool_message():
    """AC3+AC4: tool_result → role:tool message with matching tool_call_id immediately following assistant."""
    req = _req(messages=[
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_xyz789", "name": "get_weather", "input": {"location": "Paris"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_xyz789", "content": "Sunny, 22°C"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    # Three messages: user, assistant(tool_calls), tool
    assert len(result.messages) == 3

    tool_msg = result.messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tu_xyz789"
    assert tool_msg["content"] == "Sunny, 22°C"

    # tool message immediately follows assistant with matching id
    asst_msg = result.messages[1]
    assert asst_msg["tool_calls"][0]["id"] == "tu_xyz789"


# === AC5: tool_call_id exactly matches tool_use id ===

def test_ac5_tool_call_id_no_drift():
    """AC5: tool_call_id on tool message exactly matches id from tool_use (no id drift/remapping)."""
    tool_id = "toolu_complex_123_ABC"

    req = _req(messages=[
        {"role": "user", "content": "Run tool."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": "do_thing", "input": {"x": 1}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tool_id, "content": "done"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    asst = result.messages[1]
    tc_id = asst["tool_calls"][0]["id"]

    tool_msg = result.messages[2]
    tool_call_id = tool_msg["tool_call_id"]

    # All three must be identical
    assert tc_id == tool_id
    assert tool_call_id == tool_id
    assert tc_id == tool_call_id


# === AC6a: Multiple parallel tool_use blocks → multiple tool_calls ===

def test_ac6a_parallel_tool_use_blocks():
    """AC6: Multiple parallel tool_use blocks in a single assistant turn → multiple tool_calls entries."""
    req = _req(messages=[
        {"role": "user", "content": "Weather and email please."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"location": "Paris"}},
            {"type": "tool_use", "id": "tu_2", "name": "send_email", "input": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    asst = result.messages[1]
    assert len(asst["tool_calls"]) == 2

    ids = {tc["id"] for tc in asst["tool_calls"]}
    assert ids == {"tu_1", "tu_2"}

    names = {tc["function"]["name"] for tc in asst["tool_calls"]}
    assert names == {"get_weather", "send_email"}


# === AC6b: Multiple parallel tool_result blocks → multiple tool messages with correct id pairing ===

def test_ac6b_parallel_tool_results_correct_pairing():
    """AC6: Multiple parallel tool_result blocks → tool messages, each with correct id (no cross-pairing)."""
    req = _req(messages=[
        {"role": "user", "content": "Weather and email please."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"location": "Paris"}},
            {"type": "tool_use", "id": "tu_2", "name": "send_email", "input": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "Sunny, 22°C"},
            {"type": "tool_result", "tool_use_id": "tu_2", "content": "Email sent."},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    # 4 messages: user, assistant, tool(tu_1), tool(tu_2)
    assert len(result.messages) == 4

    tool_msgs = result.messages[2:]
    assert all(m["role"] == "tool" for m in tool_msgs)

    # Build map of tool_call_id → content
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert by_id["tu_1"] == "Sunny, 22°C"
    assert by_id["tu_2"] == "Email sent."

    # No cross-pairing: each tool_call_id matches exactly one content
    assert len(by_id) == 2


# === AC7: Plain text turns unchanged ===

def test_ac7_plain_text_turns_unchanged():
    """AC7: Plain text user/assistant turns with no tool blocks pass through unchanged."""
    req = _req(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Bye"},
    ])
    result = to_openai_request(req, model="gpt-4o")

    assert len(result.messages) == 3
    assert result.messages[0] == {"role": "user", "content": "Hello"}
    assert result.messages[1] == {"role": "assistant", "content": "Hi there"}
    assert result.messages[2] == {"role": "user", "content": "Bye"}

    # No tool_calls or role:tool messages
    assert not any("tool_calls" in m for m in result.messages)
    assert not any(m.get("role") == "tool" for m in result.messages)


# === AC8: pytest suite covers single and parallel round-trips ===

def test_ac8_single_tool_round_trip_id_matching():
    """AC8a: Single tool round-trip with explicit id-matching assertions."""
    req = _req(messages=[
        {"role": "user", "content": "Use the calculator."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "calc_call_42", "name": "calculate", "input": {"expr": "2+2"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "calc_call_42", "content": "4"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    # Extract IDs for explicit matching assertion
    asst_tool_call_id = result.messages[1]["tool_calls"][0]["id"]
    tool_msg_id = result.messages[2]["tool_call_id"]

    # AC8: explicit id matching
    assert asst_tool_call_id == "calc_call_42"
    assert tool_msg_id == "calc_call_42"
    assert asst_tool_call_id == tool_msg_id


def test_ac8_parallel_tool_round_trip_id_matching():
    """AC8b: Parallel tool round-trip with explicit per-result id pairing assertions."""
    req = _req(messages=[
        {"role": "user", "content": "Multiple tools."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "search_1", "name": "search", "input": {"q": "weather"}},
            {"type": "tool_use", "id": "news_1", "name": "get_news", "input": {"topic": "tech"}},
            {"type": "tool_use", "id": "translate_1", "name": "translate", "input": {"text": "hello", "lang": "fr"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "search_1", "content": "sunny"},
            {"type": "tool_result", "tool_use_id": "news_1", "content": "ai news"},
            {"type": "tool_result", "tool_use_id": "translate_1", "content": "bonjour"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    # Extract tool_calls ids
    tool_calls_ids = {tc["id"] for tc in result.messages[1]["tool_calls"]}
    assert tool_calls_ids == {"search_1", "news_1", "translate_1"}

    # Extract tool message ids and content
    tool_msgs = [m for m in result.messages[3:] if m.get("role") == "tool"]
    assert len(tool_msgs) == 3

    # AC8b: explicit per-result id pairing
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert by_id["search_1"] == "sunny"
    assert by_id["news_1"] == "ai news"
    assert by_id["translate_1"] == "bonjour"

    # Each tool_call_id from tool_calls appears exactly once
    tool_msg_ids = {m["tool_call_id"] for m in tool_msgs}
    assert tool_msg_ids == tool_calls_ids


# === Edge cases and additional validation ===

def test_arguments_json_round_trip():
    """Verify function.arguments is valid JSON that round-trips to original input."""
    original_input = {"location": "Tokyo", "units": "celsius", "details": {"temp": True}}

    req = _req(messages=[
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_rt", "name": "get_weather", "input": original_input},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    tc = result.messages[1]["tool_calls"][0]
    assert isinstance(tc["function"]["arguments"], str)

    # Round-trip
    parsed = json.loads(tc["function"]["arguments"])
    assert parsed == original_input


def test_tool_message_order_with_text_blocks():
    """Verify tool messages appear in correct order when mixed with text."""
    req = _req(messages=[
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
            {"type": "text", "text": "Searching..."},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "found"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    # user, assistant (with text and tool_calls), tool
    assert len(result.messages) == 3
    assert result.messages[1]["role"] == "assistant"
    assert result.messages[1]["content"] == "Searching..."
    assert len(result.messages[1]["tool_calls"]) == 1
    assert result.messages[2]["role"] == "tool"


def test_tool_result_with_complex_content():
    """Verify tool_result content serialization for various types."""
    req = _req(messages=[
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "text", "text": "result1"},
                {"type": "text", "text": "result2"},
            ]},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")

    tool_msg = result.messages[2]
    assert tool_msg["role"] == "tool"
    # Content should be concatenated text
    assert tool_msg["content"] is not None
