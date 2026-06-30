"""Tests for issue #25: Streamed tool call support in M2 streaming bridge."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator


# ---------------------------------------------------------------------------
# Helpers — SSE chunk builders
# ---------------------------------------------------------------------------

def _tool_call_start_chunk(index: int, call_id: str, name: str) -> bytes:
    """First fragment for a tool call: carries id and name."""
    payload = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }]
            },
            "finish_reason": None,
        }]
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _tool_call_args_chunk(index: int, arguments: str) -> bytes:
    """Subsequent fragment for a tool call: carries only function.arguments."""
    payload = {
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": index,
                    "function": {"arguments": arguments},
                }]
            },
            "finish_reason": None,
        }]
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _text_chunk(content: str) -> bytes:
    payload = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _finish_chunk(reason: str) -> bytes:
    payload = {"choices": [{"delta": {}, "finish_reason": reason}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _usage_chunk(prompt_tokens: int = 10, completion_tokens: int = 5) -> bytes:
    payload = {
        "choices": [{"delta": {}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _byte_stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _parse_sse(frames: list[str]) -> list[dict]:
    """Parse SSE frame strings into list of {event, data} dicts."""
    events = []
    for frame in frames:
        frame = frame.strip()
        if not frame or frame.startswith(":"):
            continue
        lines = frame.split("\n")
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


async def _collect_live(chunks: list[bytes]) -> list[str]:
    from services.translator import live_stream_to_anthropic_sse
    frames = []
    async for frame in live_stream_to_anthropic_sse(
        _byte_stream(*chunks),
        model="gpt-4o",
        message_id="msg_test",
        ping_interval=9999.0,
    ):
        frames.append(frame)
    return frames


def collect_live(chunks: list[bytes]) -> list[dict]:
    frames = asyncio.run(_collect_live(chunks))
    return _parse_sse(frames)


# ---------------------------------------------------------------------------
# Consumer-level tests: openai_sse_consumer parses tool_calls correctly
# ---------------------------------------------------------------------------

async def _collect_consumer(chunks: list[bytes]):
    from services.openai_sse_consumer import consume_openai_sse_stream
    events = []
    async for ev in consume_openai_sse_stream(_byte_stream(*chunks)):
        events.append(ev)
    return events


def test_consumer_yields_tool_call_start_event():
    """AC6: first fragment (carries id and name) → ToolCallStartEvent."""
    from services.openai_sse_consumer import ToolCallStartEvent
    chunks = [
        _tool_call_start_chunk(0, "call_abc", "get_weather"),
        _finish_chunk("tool_calls"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(_collect_consumer(chunks))
    start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
    assert len(start_events) == 1
    assert start_events[0].id == "call_abc"
    assert start_events[0].name == "get_weather"
    assert start_events[0].index == 0


def test_consumer_yields_tool_call_delta_event():
    """AC6: subsequent fragments (only arguments) → ToolCallDeltaEvent."""
    from services.openai_sse_consumer import ToolCallDeltaEvent
    chunks = [
        _tool_call_start_chunk(0, "call_abc", "get_weather"),
        _tool_call_args_chunk(0, '{"city":'),
        _tool_call_args_chunk(0, '"London"}'),
        _finish_chunk("tool_calls"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(_collect_consumer(chunks))
    delta_events = [e for e in events if isinstance(e, ToolCallDeltaEvent)]
    assert len(delta_events) == 2
    assert delta_events[0].partial_json == '{"city":'
    assert delta_events[1].partial_json == '"London"}'


def test_consumer_empty_start_arguments_not_yielded_as_delta():
    """AC6: the empty-string arguments in the first fragment are not yielded as ToolCallDeltaEvent."""
    from services.openai_sse_consumer import ToolCallStartEvent, ToolCallDeltaEvent
    chunks = [
        _tool_call_start_chunk(0, "call_abc", "fn"),  # has arguments: ""
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(_collect_consumer(chunks))
    start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
    delta_events = [e for e in events if isinstance(e, ToolCallDeltaEvent)]
    assert len(start_events) == 1
    assert len(delta_events) == 0  # empty "" args should not produce a delta


def test_consumer_existing_content_events_unaffected():
    """Regression: ContentEvent, FinishEvent, UsageEvent still work after adding tool call support."""
    from services.openai_sse_consumer import ContentEvent, FinishEvent, UsageEvent
    chunks = [
        _text_chunk("hello"),
        _finish_chunk("stop"),
        _usage_chunk(5, 1),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(_collect_consumer(chunks))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    finish_events = [e for e in events if isinstance(e, FinishEvent)]
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert len(content_events) == 1
    assert content_events[0].text == "hello"
    assert len(finish_events) == 1
    assert finish_events[0].reason == "stop"
    assert len(usage_events) == 1


# ---------------------------------------------------------------------------
# AC1: Pure tool call → correct Anthropic event sequence
# ---------------------------------------------------------------------------

def test_pure_tool_call_event_sequence():
    """AC1: pure tool call stream → content_block_start(tool_use) → content_block_delta(input_json_delta) → content_block_stop."""
    chunks = [
        _tool_call_start_chunk(0, "call_xyz", "search"),
        _tool_call_args_chunk(0, '{"q":'),
        _tool_call_args_chunk(0, '"cats"}'),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    types = [e["event"] for e in events]

    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types

    start = next(e for e in events if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use")
    assert start["data"]["content_block"]["type"] == "tool_use"
    assert start["data"]["content_block"]["id"] == "call_xyz"
    assert start["data"]["content_block"]["name"] == "search"

    deltas = [e for e in events if e["event"] == "content_block_delta"]
    assert all(e["data"]["delta"]["type"] == "input_json_delta" for e in deltas)

    # No text content block should appear
    text_starts = [
        e for e in events
        if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "text"
    ]
    assert len(text_starts) == 0


# ---------------------------------------------------------------------------
# AC2: message_delta stop_reason is "tool_use"
# ---------------------------------------------------------------------------

def test_tool_call_stream_stop_reason_is_tool_use():
    """AC2: message_delta event has stop_reason='tool_use' for a tool call stream."""
    chunks = [
        _tool_call_start_chunk(0, "call_1", "fn"),
        _tool_call_args_chunk(0, "{}"),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    md = next(e for e in events if e["event"] == "message_delta")
    assert md["data"]["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# AC3: partial_json fragments concatenate to valid JSON
# ---------------------------------------------------------------------------

def test_tool_call_arguments_concatenate_to_valid_json():
    """AC3: all partial_json values from input_json_delta events concatenate to valid JSON."""
    full_args = '{"city": "Paris", "units": "metric"}'
    # Split into fragments
    frag1 = full_args[:10]
    frag2 = full_args[10:20]
    frag3 = full_args[20:]
    chunks = [
        _tool_call_start_chunk(0, "call_1", "get_weather"),
        _tool_call_args_chunk(0, frag1),
        _tool_call_args_chunk(0, frag2),
        _tool_call_args_chunk(0, frag3),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    deltas = [e for e in events if e["event"] == "content_block_delta" and e["data"]["delta"]["type"] == "input_json_delta"]
    concatenated = "".join(e["data"]["delta"]["partial_json"] for e in deltas)
    assert concatenated == full_args
    # Must be valid JSON
    parsed = json.loads(concatenated)
    assert parsed == {"city": "Paris", "units": "metric"}


# ---------------------------------------------------------------------------
# AC4: Text then tool call → text block (index 0) closed before tool_use (index 1)
# ---------------------------------------------------------------------------

def test_mixed_text_then_tool_call_event_sequence():
    """AC4: leading text block (index 0) is closed before tool_use block (index 1) is opened."""
    chunks = [
        _text_chunk("Let me check "),
        _text_chunk("the weather."),
        _tool_call_start_chunk(0, "call_w", "get_weather"),
        _tool_call_args_chunk(0, '{"city":"NYC"}'),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)

    # Both text and tool_use starts must be present
    starts = [e for e in events if e["event"] == "content_block_start"]
    assert len(starts) == 2
    assert starts[0]["data"]["content_block"]["type"] == "text"
    assert starts[0]["data"]["index"] == 0
    assert starts[1]["data"]["content_block"]["type"] == "tool_use"
    assert starts[1]["data"]["index"] == 1

    # content_block_stop for index 0 must come BEFORE content_block_start for index 1
    stops = [e for e in events if e["event"] == "content_block_stop"]
    stop_for_0 = next(e for e in stops if e["data"]["index"] == 0)
    start_for_1 = next(e for e in events if e["event"] == "content_block_start" and e["data"]["index"] == 1)

    idx_stop_0 = events.index(stop_for_0)
    idx_start_1 = events.index(start_for_1)
    assert idx_stop_0 < idx_start_1, "text block must be closed before tool_use block opens"


def test_mixed_text_indices_correct():
    """AC4/AC5: text content block gets index 0, tool_use gets index 1."""
    chunks = [
        _text_chunk("hi"),
        _tool_call_start_chunk(0, "c1", "fn"),
        _tool_call_args_chunk(0, "{}"),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    text_start = next(e for e in events if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "text")
    tool_start = next(e for e in events if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use")
    assert text_start["data"]["index"] == 0
    assert tool_start["data"]["index"] == 1


# ---------------------------------------------------------------------------
# AC5: Indices monotonically increasing, no gaps or collisions
# ---------------------------------------------------------------------------

def test_content_block_indices_monotonically_increasing():
    """AC5: all content_block_start indices are monotonically increasing with no gaps."""
    chunks = [
        _text_chunk("prefix"),
        _tool_call_start_chunk(0, "c1", "fn"),
        _tool_call_args_chunk(0, '{"x":1}'),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    starts = [e for e in events if e["event"] == "content_block_start"]
    indices = [e["data"]["index"] for e in starts]
    assert indices == list(range(len(indices))), f"Indices not monotonically increasing: {indices}"


def test_pure_tool_call_block_index_is_zero():
    """AC5: pure tool call (no text) → tool_use block has index 0."""
    chunks = [
        _tool_call_start_chunk(0, "c1", "fn"),
        _tool_call_args_chunk(0, "{}"),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    tool_start = next(e for e in events if e["event"] == "content_block_start")
    assert tool_start["data"]["index"] == 0
    assert tool_start["data"]["content_block"]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# AC6: First fragment triggers content_block_start; subsequent trigger content_block_delta
# ---------------------------------------------------------------------------

def test_first_fragment_triggers_content_block_start():
    """AC6: the chunk carrying id and name triggers content_block_start for the tool_use block."""
    chunks = [
        _tool_call_start_chunk(0, "call_abc", "my_fn"),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    starts = [e for e in events if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use"]
    assert len(starts) == 1
    assert starts[0]["data"]["content_block"]["id"] == "call_abc"
    assert starts[0]["data"]["content_block"]["name"] == "my_fn"


def test_subsequent_fragments_trigger_content_block_delta():
    """AC6: subsequent chunks (only arguments) trigger content_block_delta with input_json_delta."""
    chunks = [
        _tool_call_start_chunk(0, "call_abc", "my_fn"),
        _tool_call_args_chunk(0, '{"a":'),
        _tool_call_args_chunk(0, '1}'),
        _finish_chunk("tool_calls"),
        _usage_chunk(),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    deltas = [e for e in events if e["event"] == "content_block_delta"]
    assert len(deltas) == 2
    assert all(e["data"]["delta"]["type"] == "input_json_delta" for e in deltas)
    assert deltas[0]["data"]["delta"]["partial_json"] == '{"a":'
    assert deltas[1]["data"]["delta"]["partial_json"] == "1}"


# ---------------------------------------------------------------------------
# AC7: End-to-end through stub SSE byte chunks
# ---------------------------------------------------------------------------

def test_e2e_stub_sse_chunks_tool_call():
    """AC7: stub SSE chunks → exact Anthropic event sequence and correct reconstructed args JSON."""
    full_args = '{"location": "London", "unit": "celsius"}'
    parts = [full_args[i:i+10] for i in range(0, len(full_args), 10)]

    chunks = [_tool_call_start_chunk(0, "call_weather_01", "get_current_weather")]
    for part in parts:
        chunks.append(_tool_call_args_chunk(0, part))
    chunks += [_finish_chunk("tool_calls"), _usage_chunk(20, 8), b"data: [DONE]\n\n"]

    events = collect_live(chunks)
    types = [e["event"] for e in events]

    # Required event sequence
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert types[-1] == "message_stop"

    # content_block_start carries correct id and name
    tool_start = next(e for e in events if e["event"] == "content_block_start")
    assert tool_start["data"]["content_block"]["type"] == "tool_use"
    assert tool_start["data"]["content_block"]["id"] == "call_weather_01"
    assert tool_start["data"]["content_block"]["name"] == "get_current_weather"

    # All input_json_delta fragments concatenate to valid JSON
    deltas = [e for e in events if e["event"] == "content_block_delta"]
    reconstructed = "".join(e["data"]["delta"]["partial_json"] for e in deltas)
    assert reconstructed == full_args
    assert json.loads(reconstructed) == {"location": "London", "unit": "celsius"}

    # stop_reason is tool_use
    md = next(e for e in events if e["event"] == "message_delta")
    assert md["data"]["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# Regression: text-only stream unchanged (AC5 UAT step 5)
# ---------------------------------------------------------------------------

def test_text_only_stream_unchanged():
    """Regression: a normal text-only stream produces no tool_use events and stop_reason=end_turn."""
    chunks = [
        _text_chunk("Hello"),
        _text_chunk(" world"),
        _finish_chunk("stop"),
        _usage_chunk(10, 2),
        b"data: [DONE]\n\n",
    ]
    events = collect_live(chunks)
    types = [e["event"] for e in events]

    # No tool_use content blocks
    tool_starts = [e for e in events if e["event"] == "content_block_start" and e["data"]["content_block"]["type"] == "tool_use"]
    assert len(tool_starts) == 0

    # stop_reason is end_turn
    md = next(e for e in events if e["event"] == "message_delta")
    assert md["data"]["delta"]["stop_reason"] == "end_turn"

    # Standard text events present
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_stop" in types

    # Text content correct
    deltas = [e for e in events if e["event"] == "content_block_delta"]
    text = "".join(e["data"]["delta"]["text"] for e in deltas)
    assert text == "Hello world"
