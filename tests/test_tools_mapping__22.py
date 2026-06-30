"""Tests for issue #22: Anthropic tools and tool_choice → OpenAI format mapping."""

from schemas.anthropic import MessagesRequest
from services.translator import to_openai_request


def _req(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


_TOOL_A = {
    "name": "get_weather",
    "description": "Get the current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}

_TOOL_B = {
    "name": "send_email",
    "description": "Send an email to a recipient.",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
}


# AC: Each Anthropic tool is converted to an OpenAI function tool
def test_two_tools_passthrough():
    req = _req(tools=[_TOOL_A, _TOOL_B], tool_choice="auto")
    result = to_openai_request(req, model="gpt-4o")
    assert result.tools is not None
    assert len(result.tools) == 2
    oai_a = result.tools[0]
    assert oai_a["type"] == "function"
    assert oai_a["function"]["name"] == _TOOL_A["name"]
    assert oai_a["function"]["description"] == _TOOL_A["description"]
    assert oai_a["function"]["parameters"] == _TOOL_A["input_schema"]
    oai_b = result.tools[1]
    assert oai_b["type"] == "function"
    assert oai_b["function"]["name"] == _TOOL_B["name"]
    assert oai_b["function"]["description"] == _TOOL_B["description"]
    assert oai_b["function"]["parameters"] == _TOOL_B["input_schema"]


# AC: tool_choice "auto" → "auto"
def test_tool_choice_auto():
    req = _req(tools=[_TOOL_A], tool_choice="auto")
    result = to_openai_request(req, model="gpt-4o")
    assert result.tool_choice == "auto"


# AC: tool_choice "any" → "required"
def test_tool_choice_any():
    req = _req(tools=[_TOOL_A], tool_choice="any")
    result = to_openai_request(req, model="gpt-4o")
    assert result.tool_choice == "required"


# AC: tool_choice {type: "tool", name: X} → {type: "function", function: {name: X}}
def test_tool_choice_named():
    req = _req(tools=[_TOOL_A], tool_choice={"type": "tool", "name": "get_weather"})
    result = to_openai_request(req, model="gpt-4o")
    assert result.tool_choice == {"type": "function", "function": {"name": "get_weather"}}


# AC: tools present but no tool_choice → tools array present, tool_choice omitted
def test_tools_present_no_tool_choice():
    req = _req(tools=[_TOOL_A])
    result = to_openai_request(req, model="gpt-4o")
    assert result.tools is not None
    assert len(result.tools) == 1
    assert result.tool_choice is None


# AC: no tools and no tool_choice → both omitted from output
def test_no_tools_no_tool_choice():
    req = _req()
    result = to_openai_request(req, model="gpt-4o")
    assert result.tools is None
    assert result.tool_choice is None
