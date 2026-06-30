"""Tests for issue #8: Pydantic schemas for OpenAI API shapes."""

from schemas.openai import ChatRequest, ChatResponse


# --- AC4: ChatRequest fields ---

def test_chat_request_required_fields():
    """AC4: ChatRequest has model, messages, max_tokens, stream."""
    req = ChatRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        stream=False,
    )
    assert req.model == "gpt-4"
    assert req.max_tokens == 100
    assert req.stream is False
    assert req.messages == [{"role": "user", "content": "hi"}]


def test_chat_request_optional_fields():
    """AC4: optional fields tools and tool_choice are accessible."""
    req = ChatRequest(
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=512,
        stream=False,
        tools=[{"type": "function", "function": {"name": "calculator"}}],
        tool_choice="auto",
    )
    assert req.tools is not None
    assert req.tool_choice == "auto"


# --- AC5: ChatResponse fields ---

def test_chat_response_fields():
    """AC5: ChatResponse has choices (with message and finish_reason) and usage."""
    data = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    resp = ChatResponse(**data)
    assert len(resp.choices) == 1
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.role == "assistant"
    assert resp.usage.total_tokens == 8


# --- AC9: Representative ChatRequest and ChatResponse parse without error ---

def test_chat_request_representative():
    """AC9: Full representative ChatRequest parses without error."""
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's 2+2?"},
        ],
        "max_tokens": 512,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform calculations",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    req = ChatRequest(**data)
    assert req.model == "gpt-4o"
    assert req.tool_choice == "auto"
    assert len(req.messages) == 2


def test_chat_response_representative():
    """AC9: Full representative ChatResponse parses without error."""
    data = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "4"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    resp = ChatResponse(**data)
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].message.content == "4"


# --- AC10/UAT5: Extra/unknown fields do not raise ValidationError ---

def test_chat_request_extra_fields_ignored():
    """AC10: Unknown top-level fields on ChatRequest do not raise."""
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "stream": False,
        "x_custom_param": "value",
    }
    req = ChatRequest(**data)
    assert req.model == "gpt-4"


def test_chat_response_extra_fields_in_message():
    """AC10/UAT5: Unknown fields inside choices[].message do not raise."""
    data = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hi!",
                    "unknown_field": "some_value",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    resp = ChatResponse(**data)
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Hi!"


def test_chat_response_extra_fields_on_top_level():
    """AC10: Unknown top-level fields on ChatResponse do not raise."""
    data = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "gpt-4",
        "choices": [
            {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "x_internal_flag": True,
    }
    resp = ChatResponse(**data)
    assert len(resp.choices) == 1


# --- AC11: Round-trip tests for extra/unknown fields ---

def test_round_trip_anthropic_extra_fields():
    """AC11: MessagesRequest round-trip with extra fields."""
    from schemas.anthropic import MessagesRequest

    data = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "stream": False,
        "x_internal": "abc",
    }
    req = MessagesRequest(**data)
    dumped = req.model_dump(mode="json")
    assert dumped["model"] == "claude-3-haiku-20240307"


def test_round_trip_openai_extra_fields():
    """AC11: ChatRequest round-trip with extra fields."""
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "stream": False,
        "x_internal": "abc",
    }
    req = ChatRequest(**data)
    dumped = req.model_dump(mode="json")
    assert dumped["model"] == "gpt-4"
