"""Tests for issue #8: Pydantic schemas for Anthropic API shapes."""

from schemas.anthropic import (
    ImageBlock,
    MessagesRequest,
    MessagesResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


# --- AC1: MessagesRequest fields ---

def test_messages_request_required_fields():
    """AC1: MessagesRequest has model, messages, max_tokens, stream."""
    req = MessagesRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1024,
        stream=False,
    )
    assert req.model == "claude-3-5-sonnet-20241022"
    assert req.max_tokens == 1024
    assert req.stream is False


def test_messages_request_system_as_string():
    """AC1/UAT1: system field accepts a plain string."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        stream=False,
    )
    assert req.system == "You are a helpful assistant."
    assert isinstance(req.system, str)


def test_messages_request_system_as_list_of_text_blocks():
    """AC1/UAT2: system field accepts a list of text-block objects."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        system=[{"type": "text", "text": "You are helpful."}],
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        stream=False,
    )
    assert isinstance(req.system, list)
    assert isinstance(req.system[0], TextBlock)
    assert req.system[0].text == "You are helpful."


def test_messages_request_optional_fields_present():
    """AC1: optional fields tools, tool_choice, thinking are accessible."""
    req = MessagesRequest(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1024,
        stream=False,
        tools=[{"name": "search", "description": "search", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto"},
        thinking={"type": "enabled", "budget_tokens": 1000},
    )
    assert req.tools is not None
    assert req.tool_choice == {"type": "auto"}
    assert req.thinking == {"type": "enabled", "budget_tokens": 1000}


# --- AC2: MessagesResponse fields ---

def test_messages_response_fields():
    """AC2: MessagesResponse has id, role, content, stop_reason, usage."""
    resp = MessagesResponse(
        id="msg_01",
        role="assistant",
        content=[{"type": "text", "text": "Hi!"}],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    assert resp.id == "msg_01"
    assert resp.role == "assistant"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 10


# --- AC3: Discriminated content-block types ---

def test_text_block():
    """AC3: TextBlock is a discriminated content block."""
    block = TextBlock(type="text", text="Hello world")
    assert block.type == "text"
    assert block.text == "Hello world"


def test_image_block():
    """AC3: ImageBlock is a discriminated content block."""
    block = ImageBlock(
        type="image",
        source={"type": "base64", "media_type": "image/jpeg", "data": "abc123"},
    )
    assert block.type == "image"
    assert block.source["type"] == "base64"


def test_tool_use_block():
    """AC3: ToolUseBlock is a discriminated content block."""
    block = ToolUseBlock(
        type="tool_use",
        id="tu_01",
        name="search",
        input={"q": "cats"},
    )
    assert block.type == "tool_use"
    assert block.id == "tu_01"
    assert block.name == "search"


def test_tool_result_block():
    """AC3: ToolResultBlock is a discriminated content block."""
    block = ToolResultBlock(
        type="tool_result",
        tool_use_id="tu_01",
        content="Search results here",
    )
    assert block.type == "tool_result"
    assert block.tool_use_id == "tu_01"


# --- AC7: Representative MessagesRequest with tool_use and thinking ---

def test_messages_request_representative_with_tool_use_and_thinking():
    """AC7: Full representative MessagesRequest parses without error."""
    data = {
        "model": "claude-3-5-sonnet-20241022",
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Search for cats"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu_01", "name": "search", "input": {"q": "cats"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu_01", "content": "Cat results..."}
                ],
            },
        ],
        "max_tokens": 1024,
        "stream": False,
        "tools": [{"name": "search", "description": "search", "input_schema": {"type": "object"}}],
        "tool_choice": {"type": "auto"},
        "thinking": {"type": "enabled", "budget_tokens": 1000},
    }
    req = MessagesRequest(**data)
    assert req.model == "claude-3-5-sonnet-20241022"
    assert req.system == "You are helpful."
    assert req.thinking == {"type": "enabled", "budget_tokens": 1000}
    assert len(req.messages) == 3


# --- AC8: Representative MessagesResponse with all block types ---

def test_messages_response_all_block_types():
    """AC8/UAT3: MessagesResponse with text, tool_use, tool_result, image blocks."""
    data = {
        "id": "msg_01",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Here is the answer"},
            {"type": "tool_use", "id": "tu_01", "name": "search", "input": {"q": "cats"}},
            {"type": "tool_result", "tool_use_id": "tu_01", "content": "Cat results"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc123"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 50},
    }
    resp = MessagesResponse(**data)
    assert isinstance(resp.content[0], TextBlock)
    assert isinstance(resp.content[1], ToolUseBlock)
    assert isinstance(resp.content[2], ToolResultBlock)
    assert isinstance(resp.content[3], ImageBlock)


# --- AC10/UAT4: Extra/unknown fields do not raise ValidationError ---

def test_messages_request_extra_fields_ignored():
    """AC10/UAT4: Unknown top-level fields on MessagesRequest do not raise."""
    data = {
        "model": "claude-3-haiku-20240307",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100,
        "stream": False,
        "x_internal_trace_id": "abc123",
    }
    req = MessagesRequest(**data)
    assert req.model == "claude-3-haiku-20240307"


def test_messages_response_extra_fields_ignored():
    """AC10: Unknown top-level fields on MessagesResponse do not raise."""
    data = {
        "id": "msg_01",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
        "type": "message",
        "model": "claude-3-haiku-20240307",
        "x_unknown": "value",
    }
    resp = MessagesResponse(**data)
    assert resp.id == "msg_01"


# --- AC11: Round-trip tests ---

def test_round_trip_tool_use_block():
    """AC11: Round-trip serialization of a tool_use block."""
    data = {
        "type": "tool_use",
        "id": "tu_01",
        "name": "calculator",
        "input": {"x": 1, "y": 2},
    }
    block = ToolUseBlock(**data)
    dumped = block.model_dump(mode="json")
    assert dumped["type"] == "tool_use"
    assert dumped["id"] == "tu_01"
    assert dumped["input"] == {"x": 1, "y": 2}


def test_round_trip_tool_result_block():
    """AC11: Round-trip serialization of a tool_result block."""
    data = {"type": "tool_result", "tool_use_id": "tu_01", "content": "42"}
    block = ToolResultBlock(**data)
    dumped = block.model_dump(mode="json")
    assert dumped["type"] == "tool_result"
    assert dumped["tool_use_id"] == "tu_01"
    assert dumped["content"] == "42"


def test_round_trip_image_block():
    """AC11: Round-trip serialization of an image block."""
    data = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/img.png"},
    }
    block = ImageBlock(**data)
    dumped = block.model_dump(mode="json")
    assert dumped["type"] == "image"
    assert dumped["source"]["url"] == "https://example.com/img.png"
