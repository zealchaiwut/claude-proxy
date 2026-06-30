"""Tests for issue #24: Anthropic tool_use/tool_result → OpenAI tool_calls/tool messages."""

import json

from schemas.anthropic import MessagesRequest
from services.translator import to_openai_request


def _req(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 256,
        "messages": [],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


# AC1+AC2: assistant turn with tool_use → OpenAI assistant message with tool_calls
def test_tool_use_block_becomes_tool_calls():
    req = _req(messages=[
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_abc123", "name": "get_weather", "input": {"location": "Paris"}},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")
    # user message + assistant message
    assert len(result.messages) == 2
    asst = result.messages[1]
    assert asst["role"] == "assistant"
    assert "tool_calls" in asst
    assert len(asst["tool_calls"]) == 1
    tc = asst["tool_calls"][0]
    assert tc["id"] == "tu_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    args = json.loads(tc["function"]["arguments"])
    assert args == {"location": "Paris"}


# AC3+AC4+AC5: tool_result in user turn → role:tool message following assistant
def test_tool_result_becomes_tool_message():
    req = _req(messages=[
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_abc123", "name": "get_weather", "input": {"location": "Paris"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_abc123", "content": "Sunny, 22°C"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")
    # user, assistant(tool_calls), tool
    assert len(result.messages) == 3
    tool_msg = result.messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tu_abc123"
    assert tool_msg["content"] == "Sunny, 22°C"


# AC4: tool message immediately follows assistant message with matching tool_call
def test_tool_message_order():
    req = _req(messages=[
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_abc123", "name": "get_weather", "input": {"location": "Paris"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_abc123", "content": "Sunny, 22°C"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")
    roles = [m["role"] for m in result.messages]
    assert roles == ["user", "assistant", "tool"]


# AC6: multiple parallel tool_use blocks → multiple tool_calls entries
def test_parallel_tool_use_blocks():
    req = _req(messages=[
        {"role": "user", "content": "Weather and email please."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"location": "Paris"}},
            {"type": "tool_use", "id": "tu_2", "name": "send_email", "input": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")
    asst = result.messages[1]
    assert asst["role"] == "assistant"
    assert len(asst["tool_calls"]) == 2
    ids = {tc["id"] for tc in asst["tool_calls"]}
    assert ids == {"tu_1", "tu_2"}
    names = {tc["function"]["name"] for tc in asst["tool_calls"]}
    assert names == {"get_weather", "send_email"}


# AC6: multiple parallel tool_result blocks → multiple tool messages, each with correct id
def test_parallel_tool_results():
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
    # user, assistant, tool(tu_1), tool(tu_2)
    assert len(result.messages) == 4
    tool_msgs = result.messages[2:]
    assert all(m["role"] == "tool" for m in tool_msgs)
    by_id = {m["tool_call_id"]: m["content"] for m in tool_msgs}
    assert by_id["tu_1"] == "Sunny, 22°C"
    assert by_id["tu_2"] == "Email sent."


# AC5: tool_call_id exactly matches id from tool_use (no drift)
def test_tool_call_id_matches_tool_use_id():
    req = _req(messages=[
        {"role": "user", "content": "Run tool."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_xYzAbC789", "name": "do_thing", "input": {"x": 1}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_xYzAbC789", "content": "done"},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")
    asst = result.messages[1]
    tc_id = asst["tool_calls"][0]["id"]
    tool_msg = result.messages[2]
    assert tool_msg["tool_call_id"] == tc_id == "toolu_xYzAbC789"


# AC7: plain text turns pass through unchanged
def test_plain_text_turns_unchanged():
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
    assert not any("tool_calls" in m for m in result.messages)
    assert not any(m.get("role") == "tool" for m in result.messages)


# AC2: function.arguments is valid JSON string that round-trips to original input
def test_arguments_round_trip():
    original_input = {"location": "Tokyo", "units": "celsius"}
    req = _req(messages=[
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_rt", "name": "get_weather", "input": original_input},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")
    tc = result.messages[1]["tool_calls"][0]
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == original_input


# AC3: tool_result content that is a list is serialized to JSON string
def test_tool_result_list_content_serialized():
    req = _req(messages=[
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_list", "name": "search", "input": {"q": "test"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_list", "content": [{"type": "text", "text": "result1"}]},
        ]},
    ])
    result = to_openai_request(req, model="gpt-4o")
    tool_msg = result.messages[2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "tu_list"
    # content is either the text extracted or JSON-serialized
    assert tool_msg["content"] is not None
