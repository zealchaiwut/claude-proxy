"""Acceptance criterion tests for issue #26: XML tool-call fallback for non-function-calling upstreams.

Tests verify:
- AC1: CCPROXY_TOOL_MODE=xml injects tool definitions into system prompt as XML spec
- AC2: XML tool-call blocks in upstream response are parsed and returned as valid Anthropic tool_use blocks
- AC3: CCPROXY_TOOL_MODE=native (or unset) leaves all existing behavior completely unchanged
- AC4: Malformed/partial XML degrades gracefully to plain text; no crash or 5xx
- AC5: pytest suite covers round-trip test, native mode test, and malformed XML test
- AC6: No changes required to any downstream (post-translation) code paths
"""

import json

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
    """Generate OpenAI SSE chunks for streaming response."""
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
        import httpx
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

    def stream(self, method, url, *, content, headers, **kwargs):
        import contextlib
        self.stream_calls.append({
            "method": method,
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        @contextlib.asynccontextmanager
        async def _stream_ctx():
            yield _MockStreamResponse(200, self.stream_chunks)
        return _stream_ctx()

    async def post(self, url, *, content, headers, **kwargs):
        import httpx
        self.post_calls.append({
            "url": url,
            "content": content,
            "headers": {k.lower(): v for k, v in dict(headers).items()},
        })
        return httpx.Response(200, content=self.post_body, headers={"content-type": "application/json"})

    async def aclose(self):
        pass


def _setup(mock_client, monkeypatch, *, tool_mode: str | None = None):
    """Setup test environment with mock OpenAI client."""
    monkeypatch.setenv("CCPROXY_PROFILE", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", OPENAI_BASE)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    monkeypatch.setenv("OPENAI_MODEL", OPENAI_MODEL)
    if tool_mode is not None:
        monkeypatch.setenv("CCPROXY_TOOL_MODE", tool_mode)
    app.state.http_client = mock_client
    app.state.settings = Settings(upstream_base_url=UPSTREAM_ANTHROPIC)


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE format into list of {event, data} dicts."""
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


# === AC1: CCPROXY_TOOL_MODE=xml injects tool definitions into system prompt ===

def test_ac1_xml_mode_injects_tool_spec_into_system_prompt(monkeypatch):
    """AC1: xml mode injects XML tool spec into the system message sent to upstream."""
    mock = MockOpenAIClient(post_body=OPENAI_XML_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        resp = tc.post("/v1/messages", content=TOOLS_REQUEST_BODY, headers={"content-type": "application/json"})

    assert resp.status_code == 200
    sent_body = json.loads(mock.post_calls[0]["content"])
    messages = sent_body.get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert len(system_messages) == 1, "System message not injected"
    system_text = system_messages[0]["content"]
    assert "get_weather" in system_text, "Tool name not in system prompt"
    assert "<tools>" in system_text, "XML tool spec not in system prompt"
    assert "<tool_call>" in system_text, "Example not in system prompt"


# === AC2: XML tool-call blocks parsed and returned as valid Anthropic tool_use blocks ===

def test_ac2_xml_tool_calls_parsed_to_tool_use_non_streaming(monkeypatch):
    """AC2(a): xml mode non-streaming — stub upstream echoes XML tool call → valid tool_use block."""
    mock = MockOpenAIClient(post_body=OPENAI_XML_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        resp = tc.post("/v1/messages", content=TOOLS_REQUEST_BODY, headers={"content-type": "application/json"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["stop_reason"] == "tool_use", "stop_reason not set to tool_use"
    content = body["content"]
    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1, "tool_use block not parsed"
    block = tool_use_blocks[0]
    assert block["name"] == "get_weather", "tool_use name mismatch"
    assert block["id"] == "call_abc123", "tool_use id mismatch"
    assert block["input"] == {"location": "NYC"}, "tool_use input mismatch"
    # Raw XML must NOT leak into text content blocks
    text_blocks = [b for b in content if b.get("type") == "text"]
    for tb in text_blocks:
        assert "<tool_call>" not in tb.get("text", ""), "XML leaked into text block"


def test_ac2_xml_tool_calls_parsed_streaming(monkeypatch):
    """AC2(b): xml mode streaming — SSE stream with XML tool call → tool_use block in final response."""
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
    assert len(tool_use_starts) == 1, "tool_use block not found in streaming response"
    cb = tool_use_starts[0]["data"]["content_block"]
    assert cb["name"] == "get_weather", "streaming tool_use name mismatch"
    assert cb["id"] == "call_abc123", "streaming tool_use id mismatch"
    # stop_reason in message_delta must be tool_use
    msg_deltas = [e for e in events if e["event"] == "message_delta"]
    assert len(msg_deltas) == 1
    assert msg_deltas[0]["data"]["delta"]["stop_reason"] == "tool_use", "stop_reason not tool_use in streaming"


# === AC3: CCPROXY_TOOL_MODE=native (or unset) leaves all existing behavior unchanged ===

def test_ac3_native_mode_unset_no_xml_injection(monkeypatch):
    """AC3(a): CCPROXY_TOOL_MODE unset → native behavior, no XML injection."""
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
    assert body["stop_reason"] == "end_turn", "native mode stop_reason changed"
    assert body["content"][0]["type"] == "text"
    assert body["content"][0]["text"] == "Hello there!"
    # No XML injection happened in system prompt
    sent_body = json.loads(mock.post_calls[0]["content"])
    messages = sent_body.get("messages", [])
    system_messages = [m for m in messages if m.get("role") == "system"]
    assert len(system_messages) == 0, "system prompt injected in native mode"


def test_ac3_native_mode_explicit_native_value(monkeypatch):
    """AC3(b): CCPROXY_TOOL_MODE=native → same as unset, no XML injection."""
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


# === AC4: Malformed/partial XML degrades gracefully to plain text; no crash or 5xx ===

def test_ac4_malformed_xml_graceful_fallback_non_streaming(monkeypatch):
    """AC4(a): malformed XML in upstream response → 200, raw text as plain text, no crash."""
    mock = MockOpenAIClient(post_body=OPENAI_MALFORMED_XML_RESPONSE)
    with TestClient(app) as tc:
        _setup(mock, monkeypatch, tool_mode="xml")
        resp = tc.post(
            "/v1/messages",
            content=TOOLS_REQUEST_BODY,
            headers={"content-type": "application/json"},
        )

    # Must not crash or return 5xx
    assert resp.status_code == 200, "malformed XML caused 5xx error"
    body = resp.json()
    # Content must include the raw malformed text (not parsed as tool_use)
    content = body["content"]
    text_blocks = [b for b in content if b.get("type") == "text"]
    assert len(text_blocks) >= 1, "no text block for malformed XML"
    all_text = "".join(b["text"] for b in text_blocks)
    assert MALFORMED_XML_TEXT in all_text, "malformed XML lost"
    # No tool_use blocks generated from malformed XML
    tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 0, "malformed XML incorrectly parsed as tool_use"


def test_ac4_malformed_xml_graceful_fallback_streaming(monkeypatch):
    """AC4(b): malformed XML in streaming response → 200 SSE, raw text, no crash."""
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
    assert resp.status_code == 200, "malformed XML streaming caused error"
    events = _parse_sse(raw)
    # No tool_use content_block_start events
    tool_use_starts = [
        e for e in events
        if e["event"] == "content_block_start"
        and isinstance(e["data"], dict)
        and e["data"].get("content_block", {}).get("type") == "tool_use"
    ]
    assert len(tool_use_starts) == 0, "malformed XML incorrectly parsed in streaming"
    # Text content with the raw malformed XML is present
    text_deltas = [
        e for e in events
        if e["event"] == "content_block_delta"
        and isinstance(e["data"], dict)
        and e["data"].get("delta", {}).get("type") == "text_delta"
    ]
    assert len(text_deltas) > 0, "no text delta for malformed XML"
    all_text = "".join(e["data"]["delta"]["text"] for e in text_deltas)
    assert MALFORMED_XML_TEXT in all_text, "malformed XML lost in streaming"


# === AC5: pytest suite covers round-trip, native mode, and malformed XML ===
# (Already covered by tests above: test_ac2_xml_tool_calls_parsed_to_tool_use_non_streaming,
#  test_ac3_native_mode_unset_no_xml_injection, test_ac4_malformed_xml_graceful_fallback_non_streaming)


# === AC6: No changes required to any downstream (post-translation) code paths ===

def test_ac6_native_mode_no_tools_downstream_unchanged(monkeypatch):
    """AC6: native mode requests without tools flow through unchanged."""
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
    # Verify standard response structure is intact
    assert "id" in body
    assert "type" in body
    assert "role" in body
    assert "model" in body
    assert "content" in body
    assert "stop_reason" in body
    assert "usage" in body
