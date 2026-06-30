"""Tests for issue #26: XML tool-call fallback for non-function-calling upstreams."""
from __future__ import annotations

import contextlib
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from config import Settings
from main import app

OPENAI_BASE = "http://openai-stub.test"
OPENAI_KEY = "sk-test"
OPENAI_MODEL = "gpt-4o"
UPSTREAM_ANTHROPIC = "http://anthropic.test"

# Anthropic request body with tools defined
TOOLS_REQUEST_BODY = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ],
}).encode()

TOOLS_STREAM_REQUEST_BODY = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
    "stream": True,
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ],
}).encode()

PLAIN_REQUEST_BODY = json.dumps({
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}],
}).encode()

# Well-formed XML tool call the stub upstream echoes back as content text
XML_TOOL_CALL_TEXT = (
    "<tool_call>\n"
    "<name>get_weather</name>\n"
    "<id>call_abc123</id>\n"
    "<input>{\"location\": \"NYC\"}</input>\n"
    "</tool_call>"
)

MALFORMED_XML_TEXT = "<tool_call><name>get_weather<input>{bad}</tool_call>"

# Stub OpenAI response that contains the XML tool call as plain text
OPENAI_XML_RESPONSE = json.dumps({
    "id": "chatcmpl-xml01",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "message": {"role": "assistant", "content": XML_TOOL_CALL_TEXT},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
}).encode()

OPENAI_MALFORMED_XML_RESPONSE = json.dumps({
    "id": "chatcmpl-xml02",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "message": {"role": "assistant", "content": MALFORMED_XML_TEXT},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}).encode()

OPENAI_PLAIN_RESPONSE = json.dumps({
    "id": "chatcmpl-plain01",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}).encode()


def _openai_sse_chunks(tokens: list[str], finish_reason: str = "stop") -> list[bytes]:
    chunks = []
    for token in tokens:
        payload = {"choices": [{"delta": {"content": token}, "finish_reason": None}]}
        chunks.append(f"data: {json.dumps(payload)}\n\n".encode())
    finish = {"choices": [{"delta": {}, "finish_reason": finish_reason}]}
    chunks.append(f"data: {json.dumps(finish)}\n\n".encode())
    usage_payload = {
        "choices": [{"delta": {}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": len(tokens), "total_tokens": 5 + len(tokens)},
    }
    chunks.append(f"data: {json.dumps(usage_payload)}\n\n".encode())
    chunks.append(b"data: [DONE]\n\n")
    return chunks


class _MockStreamResponse:
    def __init__(self, status_code: int, chunks: list[bytes], headers: dict | None = None):
        self.status_code = status_code
        self._chunks = chunks
        self.headers = httpx.Headers(headers or {"content-type": "text/event-stream"})

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class MockOpenAIClient:
    def __init__(
        self,
        post_body: bytes = OPENAI_PLAIN_RESPONSE,
        stream_chunks: list[bytes] | None = None,
    ):
        self.post_body = post_body
        self.stream_chunks = stream_chunks if stream_chunks is not None else []
        self.post_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    @contextlib.asynccontextmanager
    async def stream(self, method, url, *, content, headers, **kwargs):
        self.stream_calls.append({
            "method": method,
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        yield _MockStreamResponse(200, self.stream_chunks)

    async def post(self, url, *, content, headers, **kwargs):
        self.post_calls.append({
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        return httpx.Response(200, content=self.post_body, headers={"content-type": "application/json"})

    async def aclose(self):
        pass


def _setup(mock_client, monkeypatch, *, tool_mode: str | None = None):
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)
    if tool_mode is not None:
        monkeypatch.setenv("CCPROXY_TOOL_MODE", tool_mode)
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=UPSTREAM_ANTHROPIC)


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        lines = block.split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[6:])
                except Exception:
                    data = line[6:]
        if event_type is not None:
            events.append({"event": event_type, "data": data})
    return events


# ---------------------------------------------------------------------------
# Unit tests: xml_tool_mode service functions
# ---------------------------------------------------------------------------

def test_build_xml_system_prompt_injects_tool_definitions():
    """build_xml_system_prompt includes tool name, description, and input_schema."""
    from services.xml_tool_mode import build_xml_system_prompt

    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
        }
    ]
    result = build_xml_system_prompt(None, tools)
    assert "get_weather" in result
    assert "Get weather" in result
    assert "<tools>" in result
    assert "<tool_call>" in result  # example is included


def test_build_xml_system_prompt_preserves_existing_system():
    """build_xml_system_prompt keeps existing system text and appends tool spec."""
    from services.xml_tool_mode import build_xml_system_prompt

    tools = [{"name": "my_tool", "description": "d", "input_schema": {}}]
    result = build_xml_system_prompt("You are helpful.", tools)
    assert result.startswith("You are helpful.")
    assert "my_tool" in result


def test_parse_xml_tool_calls_well_formed():
    """parse_xml_tool_calls extracts a valid tool_use block from well-formed XML."""
    from services.xml_tool_mode import parse_xml_tool_calls
    from schemas.anthropic import ToolUseBlock

    text = XML_TOOL_CALL_TEXT
    cleaned, blocks = parse_xml_tool_calls(text)
    assert len(blocks) == 1
    b = blocks[0]
    assert isinstance(b, ToolUseBlock)
    assert b.name == "get_weather"
    assert b.id == "call_abc123"
    assert b.input == {"location": "NYC"}
    assert cleaned == ""  # All text was tool_call, cleaned is empty


def test_parse_xml_tool_calls_text_before_and_after():
    """parse_xml_tool_calls strips tool_call block and leaves surrounding text."""
    from services.xml_tool_mode import parse_xml_tool_calls

    text = "Sure!\n" + XML_TOOL_CALL_TEXT + "\nDone."
    cleaned, blocks = parse_xml_tool_calls(text)
    assert len(blocks) == 1
    assert "Sure!" in cleaned
    assert "Done." in cleaned
    assert "<tool_call>" not in cleaned


def test_parse_xml_tool_calls_malformed_returns_original_text():
    """parse_xml_tool_calls returns (original_text, []) on malformed XML — graceful fallback."""
    from services.xml_tool_mode import parse_xml_tool_calls

    text = MALFORMED_XML_TEXT
    cleaned, blocks = parse_xml_tool_calls(text)
    assert blocks == []
    assert cleaned == text  # original text unchanged


def test_parse_xml_tool_calls_no_tool_call_block():
    """parse_xml_tool_calls returns (text, []) when no <tool_call> present."""
    from services.xml_tool_mode import parse_xml_tool_calls

    text = "Hello, world! No tool calls here."
    cleaned, blocks = parse_xml_tool_calls(text)
    assert blocks == []
    assert cleaned == text


# ---------------------------------------------------------------------------
# AC (e.a): xml-mode round-trip test against stub upstream — non-streaming
# ---------------------------------------------------------------------------

def test_xml_mode_non_streaming_round_trip(monkeypatch):
    """AC(a): xml mode non-streaming — stub upstream echoes XML tool call → valid tool_use block."""
    mock = MockOpenAIClient(post_body=OPENAI_XML_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        resp = tc.post(
            "/v1/messages",
            content=TOOLS_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use"
    content = body["content"]
    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1
    block = tool_use_blocks[0]
    assert block["name"] == "get_weather"
    assert block["id"] == "call_abc123"
    assert block["input"] == {"location": "NYC"}
    # Raw XML must NOT leak into text content blocks
    text_blocks = [b for b in content if b.get("type") == "text"]
    for tb in text_blocks:
        assert "<tool_call>" not in tb.get("text", "")


def test_xml_mode_injects_tool_spec_into_system_prompt(monkeypatch):
    """AC(a): xml mode injects XML tool spec into the system message sent to upstream."""
    mock = MockOpenAIClient(post_body=OPENAI_XML_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        tc.post("/v1/messages", content=TOOLS_REQUEST_BODY, headers={"content-type": "application/json"})

    assert len(mock.post_calls) == 1
    sent_body = json.loads(mock.post_calls[0]["content"])
    messages = sent_body.get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert len(system_messages) == 1
    system_text = system_messages[0]["content"]
    assert "get_weather" in system_text
    assert "<tools>" in system_text


def test_xml_mode_does_not_send_native_tools_to_upstream(monkeypatch):
    """AC(a): xml mode does not forward native tools/functions to upstream."""
    mock = MockOpenAIClient(post_body=OPENAI_XML_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        tc.post("/v1/messages", content=TOOLS_REQUEST_BODY, headers={"content-type": "application/json"})

    sent_body = json.loads(mock.post_calls[0]["content"])
    # No "tools" key or tools=null means no native function calling forwarded
    assert not sent_body.get("tools")


# ---------------------------------------------------------------------------
# AC (e.a): xml-mode streaming round-trip
# ---------------------------------------------------------------------------

def test_xml_mode_streaming_round_trip(monkeypatch):
    """AC(b): xml mode streaming — SSE stream with XML tool call → tool_use block in final response."""
    xml_tokens = [
        "<tool_call>\n",
        "<name>get_weather</name>\n",
        "<id>call_abc123</id>\n",
        "<input>{\"location\": \"NYC\"}</input>\n",
        "</tool_call>",
    ]
    stream_chunks = _openai_sse_chunks(xml_tokens, finish_reason="stop")
    mock = MockOpenAIClient(stream_chunks=stream_chunks)

    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        with tc.stream(
            "POST",
            "/v1/messages",
            content=TOOLS_STREAM_REQUEST_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            raw = resp.read().decode()

    events = _parse_sse(raw)
    tool_use_starts = [
        e for e in events
        if e["event"] == "content_block_start"
        and isinstance(e["data"], dict)
        and e["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_use_starts) == 1
    cb = tool_use_starts[0]["data"]["content_block"]
    assert cb["name"] == "get_weather"
    assert cb["id"] == "call_abc123"

    # stop_reason in message_delta must be tool_use
    msg_deltas = [e for e in events if e["event"] == "message_delta"]
    assert len(msg_deltas) == 1
    assert msg_deltas[0]["data"]["delta"]["stop_reason"] == "tool_use"

    # Response content-type is SSE
    assert "text/event-stream" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# AC (e.b): native mode is unaffected
# ---------------------------------------------------------------------------

def test_native_mode_unaffected_no_env_var(monkeypatch):
    """AC(c): CCPROXY_TOOL_MODE unset → native behavior, no XML injection, plain text returned."""
    mock = MockOpenAIClient(post_body=OPENAI_PLAIN_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch)  # tool_mode not set
        resp = tc.post(
            "/v1/messages",
            content=PLAIN_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Hello there!"

    # No XML injection happened in system prompt
    sent_body = json.loads(mock.post_calls[0]["content"])
    messages = sent_body.get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert len(system_messages) == 0  # no system prompt was added


def test_native_mode_explicit_native_value(monkeypatch):
    """AC(c): CCPROXY_TOOL_MODE=native → same as unset, no XML injection."""
    mock = MockOpenAIClient(post_body=OPENAI_PLAIN_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="native")
        resp = tc.post(
            "/v1/messages",
            content=PLAIN_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "end_turn"
    assert body["content"][0]["text"] == "Hello there!"


# ---------------------------------------------------------------------------
# AC (e.c): malformed XML → graceful text fallback
# ---------------------------------------------------------------------------

def test_malformed_xml_graceful_fallback_non_streaming(monkeypatch):
    """AC(d): malformed XML in upstream response → 200, raw text as plain text, no crash."""
    mock = MockOpenAIClient(post_body=OPENAI_MALFORMED_XML_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        resp = tc.post(
            "/v1/messages",
            content=TOOLS_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    # Must not crash or return 5xx
    assert resp.status_code == 200
    body = resp.json()
    # Content must include the raw malformed text (not parsed as tool_use)
    content = body["content"]
    text_blocks = [b for b in content if b.get("type") == "text"]
    assert len(text_blocks) >= 1
    all_text = "".join(b["text"] for b in text_blocks)
    assert MALFORMED_XML_TEXT in all_text
    # No tool_use blocks generated from malformed XML
    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 0


def test_malformed_xml_graceful_fallback_streaming(monkeypatch):
    """AC(d): malformed XML in streaming response → 200 SSE, raw text, no crash."""
    malformed_tokens = [MALFORMED_XML_TEXT]
    stream_chunks = _openai_sse_chunks(malformed_tokens, finish_reason="stop")
    mock = MockOpenAIClient(stream_chunks=stream_chunks)

    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        with tc.stream(
            "POST",
            "/v1/messages",
            content=TOOLS_STREAM_REQUEST_BODY,
            headers={"content-type": "application/json"},
        ) as resp:
            raw = resp.read().decode()

    # No crash, status 200
    assert resp.status_code == 200
    events = _parse_sse(raw)
    # No tool_use content_block_start events
    tool_use_starts = [
        e for e in events
        if e["event"] == "content_block_start"
        and isinstance(e["data"], dict)
        and e["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_use_starts) == 0
    # Text content with the raw malformed XML is present
    text_deltas = [
        e for e in events
        if e["event"] == "content_block_delta"
        and isinstance(e["data"], dict)
        and e["data"].get("delta", {}).get("type") == "text_delta"
    ]
    assert len(text_deltas) > 0
    all_text = "".join(e["data"]["delta"]["text"] for e in text_deltas)
    assert MALFORMED_XML_TEXT in all_text
