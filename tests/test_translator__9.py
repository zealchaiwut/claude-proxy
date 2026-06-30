"""Tests for issue #9: Anthropic → OpenAI request translation (services/translator.py)."""

from schemas.anthropic import MessagesRequest, TextBlock
from schemas.openai import ChatRequest
from services.translator import to_openai_request


def _req(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


# AC: system-as-string → single leading OpenAI system message
def test_system_string_maps_to_system_message():
    req = _req(system="You are helpful.")
    result = to_openai_request(req, model="gpt-4o")
    assert isinstance(result, ChatRequest)
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "You are helpful."


# AC: system-as-list-of-text-blocks → concatenated single system message
def test_system_list_of_text_blocks_concatenated():
    req = _req(
        system=[
            TextBlock(type="text", text="Block one. "),
            TextBlock(type="text", text="Block two."),
        ],
        messages=[{"role": "user", "content": "hi"}],
    )
    result = to_openai_request(req, model="gpt-4o")
    assert result.messages[0]["role"] == "system"
    assert result.messages[0]["content"] == "Block one. Block two."


# AC: multi-turn conversation without system prompt
def test_multiturn_no_system():
    req = _req(
        system=None,
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ],
    )
    result = to_openai_request(req, model="gpt-4o")
    assert len(result.messages) == 3
    assert [m["role"] for m in result.messages] == ["user", "assistant", "user"]
    assert result.messages[0]["content"] == "Hello"
    assert result.messages[1]["content"] == "Hi"
    assert result.messages[2]["content"] == "Bye"


# AC: max_tokens carried through unchanged
def test_max_tokens_preserved():
    req = _req(max_tokens=512)
    result = to_openai_request(req, model="gpt-4o")
    assert result.max_tokens == 512


# AC: model set from argument, not from client request
def test_model_from_argument_not_client():
    req = _req(model="claude-3-haiku-20240307")
    result = to_openai_request(req, model="gpt-4o-mini")
    assert result.model == "gpt-4o-mini"


# AC: stream not set on produced OpenAI request
def test_stream_not_set():
    req = _req(stream=True)
    result = to_openai_request(req, model="gpt-4o")
    assert result.stream is False


# AC: text content blocks in a turn flattened to content string
def test_text_content_blocks_flattened():
    req = _req(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part A "},
                    {"type": "text", "text": "Part B"},
                ],
            }
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert result.messages[0]["content"] == "Part A Part B"


# AC: image blocks silently skipped (no exception)
def test_image_blocks_silently_skipped():
    req = _req(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}},
                ],
            }
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert result.messages[0]["content"] == "describe this"


# AC: tool_use blocks silently skipped (no exception)
def test_tool_use_blocks_silently_skipped():
    req = _req(
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Using tool"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
                ],
            }
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert result.messages[0]["content"] == "Using tool"


# M3: tool_result blocks are now translated to discrete tool messages (supersedes M1 skip behavior)
def test_tool_result_blocks_translated_to_tool_messages():
    req = _req(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "result"},
                    {"type": "text", "text": "Done"},
                ],
            }
        ]
    )
    result = to_openai_request(req, model="gpt-4o")
    assert result.messages[0]["role"] == "tool"
    assert result.messages[0]["tool_call_id"] == "t1"
    assert result.messages[0]["content"] == "result"
    assert result.messages[1]["role"] == "user"
    assert result.messages[1]["content"] == "Done"


# AC: system message is leading (before conversation turns)
def test_system_message_is_first():
    req = _req(
        system="Be concise.",
        messages=[{"role": "user", "content": "Hi"}],
    )
    result = to_openai_request(req, model="gpt-4o")
    assert result.messages[0]["role"] == "system"
    assert result.messages[1]["role"] == "user"
    assert len(result.messages) == 2
