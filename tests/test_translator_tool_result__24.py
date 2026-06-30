"""Tests for issue #24: translate Anthropic tool_use/tool_result into OpenAI tool_calls/tool messages."""
import json

from schemas.anthropic import MessagesRequest
from services.translator import to_openai_request


def _req(messages: list, **kwargs) -> MessagesRequest:
    defaults = {"model": "claude-3-haiku-20240307", "max_tokens": 256}
    defaults.update(kwargs)
    return MessagesRequest(messages=messages, **defaults)


# AC1 & AC2: assistant turn with tool_use block → OpenAI assistant message with tool_calls
def test_tool_use_produces_tool_calls_entry():
    req = _req(
        messages=[
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_001", "name": "get_weather", "input": {"location": "NYC"}},
                ],
            },
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    asst = result.messages[1]
    assert asst["role"] == "assistant"
    assert "tool_calls" in asst
    assert len(asst["tool_calls"]) == 1
    tc = asst["tool_calls"][0]
    assert tc["id"] == "call_001"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"location": "NYC"}


# AC2: function.arguments round-trips back to the original input object
def test_tool_calls_arguments_roundtrips():
    input_data = {"city": "Paris", "units": "metric"}
    req = _req(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_x", "name": "weather", "input": input_data},
                ],
            }
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    args_str = result.messages[0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args_str, str)
    assert json.loads(args_str) == input_data


# AC3 & AC5: tool_result in user turn → discrete {role: tool} message with matching tool_call_id
def test_tool_result_produces_tool_message():
    req = _req(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_001", "content": "Sunny, 72°F"},
                ],
            }
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert len(result.messages) == 1
    tool_msg = result.messages[0]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_001"
    assert tool_msg["content"] == "Sunny, 72°F"


# AC4 & AC5: single tool round-trip — tool message immediately follows assistant, ids match exactly
def test_single_tool_roundtrip_id_matching():
    req = _req(
        messages=[
            {"role": "user", "content": "Search for something"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_abc", "name": "search", "input": {"q": "cats"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_abc", "content": "Found 100 results"},
                ],
            },
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert len(result.messages) == 3
    user_msg, asst_msg, tool_msg = result.messages

    assert user_msg["role"] == "user"
    assert asst_msg["role"] == "assistant"
    assert tool_msg["role"] == "tool"

    tc_id = asst_msg["tool_calls"][0]["id"]
    assert tc_id == "call_abc"
    assert tool_msg["tool_call_id"] == tc_id  # no id drift


# AC6: parallel tool_use → multiple tool_calls; parallel tool_result → multiple tool messages, correct pairing
def test_parallel_tools_roundtrip():
    req = _req(
        messages=[
            {"role": "user", "content": "Do two things"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_1", "name": "func_a", "input": {"x": 1}},
                    {"type": "tool_use", "id": "call_2", "name": "func_b", "input": {"y": 2}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": "result_a"},
                    {"type": "tool_result", "tool_use_id": "call_2", "content": "result_b"},
                ],
            },
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert len(result.messages) == 4  # user, assistant, tool1, tool2

    asst_msg = result.messages[1]
    assert asst_msg["role"] == "assistant"
    assert len(asst_msg["tool_calls"]) == 2
    assert asst_msg["tool_calls"][0]["id"] == "call_1"
    assert asst_msg["tool_calls"][0]["function"]["name"] == "func_a"
    assert asst_msg["tool_calls"][1]["id"] == "call_2"
    assert asst_msg["tool_calls"][1]["function"]["name"] == "func_b"

    tool1 = result.messages[2]
    tool2 = result.messages[3]
    assert tool1["role"] == "tool"
    assert tool1["tool_call_id"] == "call_1"  # no cross-pairing
    assert tool1["content"] == "result_a"
    assert tool2["role"] == "tool"
    assert tool2["tool_call_id"] == "call_2"
    assert tool2["content"] == "result_b"


# AC7: plain text turns with no tool blocks pass through unchanged
def test_plain_text_turns_unchanged():
    req = _req(
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Goodbye"},
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert len(result.messages) == 3
    assert result.messages[0] == {"role": "user", "content": "Hello"}
    assert result.messages[1] == {"role": "assistant", "content": "Hi there"}
    assert result.messages[2] == {"role": "user", "content": "Goodbye"}
    for m in result.messages:
        assert "tool_calls" not in m
        assert m["role"] != "tool"
