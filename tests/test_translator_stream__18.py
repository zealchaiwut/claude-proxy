"""Tests for issue #18: Bridge OpenAI stream to Anthropic emitter in translator."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, AsyncIterator

import pytest


# ---------------------------------------------------------------------------
# Stub event types (mirrors services/openai_sse_consumer event shapes)
# ---------------------------------------------------------------------------

@dataclass
class _ContentEvent:
    text: str


@dataclass
class _FinishEvent:
    reason: str


@dataclass
class _UsageEvent:
    usage: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into list of {'event': str, 'data': dict}."""
    events = []
    for block in raw.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_type is not None:
            events.append({"event": event_type, "data": data})
    return events


async def _stream(*events) -> AsyncIterator:
    for e in events:
        yield e


async def collect(gen) -> str:
    parts = []
    async for frame in gen:
        parts.append(frame)
    return "".join(parts)


# ---------------------------------------------------------------------------
# AC1: text deltas forwarded as content_block_delta events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_deltas_forwarded_as_content_block_delta():
    """AC1: each text delta → content_block_delta event with delta text."""
    from services.translator import stream_to_anthropic_sse

    texts = ["Hello", ", ", "world", "!"]

    async def _gen():
        for t in texts:
            yield _ContentEvent(t)
        yield _FinishEvent("stop")
        yield _UsageEvent({"completion_tokens": 4, "prompt_tokens": 10})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="gpt-4o", message_id="msg_1"))
    parsed = parse_sse(raw)
    deltas = [e for e in parsed if e["event"] == "content_block_delta"]

    assert len(deltas) == len(texts), f"Expected {len(texts)} deltas, got {len(deltas)}"
    for i, (evt, expected) in enumerate(zip(deltas, texts)):
        assert evt["data"]["delta"]["text"] == expected, f"Delta {i}: text mismatch"
        assert evt["data"]["delta"]["type"] == "text_delta"
        assert evt["data"]["index"] == 0


# ---------------------------------------------------------------------------
# AC2: finish_reason stop → stop_reason end_turn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finish_reason_stop_maps_to_end_turn():
    """AC2: finish_reason='stop' → stop_reason='end_turn' in message_delta."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("hi")
        yield _FinishEvent("stop")
        yield _UsageEvent({"completion_tokens": 1})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["delta"]["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# AC3: finish_reason length → stop_reason max_tokens
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finish_reason_length_maps_to_max_tokens():
    """AC3: finish_reason='length' → stop_reason='max_tokens' in message_delta."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("truncated")
        yield _FinishEvent("length")
        yield _UsageEvent({"completion_tokens": 1})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["delta"]["stop_reason"] == "max_tokens"


# ---------------------------------------------------------------------------
# AC4: finish_reason tool_calls → stop_reason tool_use (no error)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_finish_reason_tool_calls_maps_to_tool_use():
    """AC4: finish_reason='tool_calls' → stop_reason='tool_use', no exception."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _FinishEvent("tool_calls")
        yield _UsageEvent({"completion_tokens": 0})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["delta"]["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# AC5: output_tokens from upstream usage when present
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_tokens_from_upstream_usage():
    """AC5: output_tokens populated from upstream usage.completion_tokens."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("test")
        yield _FinishEvent("stop")
        yield _UsageEvent({"completion_tokens": 42, "prompt_tokens": 10})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["usage"]["output_tokens"] == 42


# ---------------------------------------------------------------------------
# AC6: output_tokens fallback to counted value when usage omitted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_tokens_fallback_when_usage_omitted():
    """AC6: when upstream omits usage, output_tokens falls back to non-zero counted value."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("Hello world this is a test")
        yield _FinishEvent("stop")
        # No UsageEvent

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["usage"]["output_tokens"] > 0, "output_tokens must be non-zero when usage omitted"


# ---------------------------------------------------------------------------
# AC7: model and message_id propagate from context, not hardcoded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_and_message_id_propagate():
    """AC7: model and message_id appear in message_start from context, not hardcoded."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("ok")
        yield _FinishEvent("stop")
        yield _UsageEvent({"completion_tokens": 1})

    raw = await collect(stream_to_anthropic_sse(
        _gen(),
        model="gpt-4o-custom",
        message_id="msg_abc123",
    ))
    parsed = parse_sse(raw)
    start = next(e for e in parsed if e["event"] == "message_start")
    assert start["data"]["message"]["model"] == "gpt-4o-custom"
    assert start["data"]["message"]["id"] == "msg_abc123"


# ---------------------------------------------------------------------------
# AC8a: concatenated text reconstructs original
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concatenated_text_reconstructs_original():
    """AC8a: concatenating all content_block_delta texts exactly reproduces the original."""
    from services.translator import stream_to_anthropic_sse

    source_texts = ["The", " quick", " brown", " fox"]

    async def _gen():
        for t in source_texts:
            yield _ContentEvent(t)
        yield _FinishEvent("stop")
        yield _UsageEvent({"completion_tokens": 4})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    deltas = [e for e in parsed if e["event"] == "content_block_delta"]
    concatenated = "".join(e["data"]["delta"]["text"] for e in deltas)
    assert concatenated == "".join(source_texts)


# ---------------------------------------------------------------------------
# AC8b: stop_reason in message_delta matches expected mapped value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason,expected_stop_reason", [
    ("stop", "end_turn"),
    ("length", "max_tokens"),
    ("tool_calls", "tool_use"),
])
async def test_stop_reason_mapping(finish_reason, expected_stop_reason):
    """AC8b: stop_reason in message_delta matches expected mapped value."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("x")
        yield _FinishEvent(finish_reason)
        yield _UsageEvent({"completion_tokens": 1})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["delta"]["stop_reason"] == expected_stop_reason


# ---------------------------------------------------------------------------
# AC8c: output_tokens is non-zero
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_tokens_non_zero_with_usage():
    """AC8c: output_tokens is non-zero when text is present and usage is provided."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("some text here")
        yield _FinishEvent("stop")
        yield _UsageEvent({"completion_tokens": 3})

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["usage"]["output_tokens"] > 0


@pytest.mark.asyncio
async def test_output_tokens_non_zero_with_fallback():
    """AC8c: output_tokens is non-zero when text is present and usage is omitted (fallback)."""
    from services.translator import stream_to_anthropic_sse

    async def _gen():
        yield _ContentEvent("some text here")
        yield _FinishEvent("stop")

    raw = await collect(stream_to_anthropic_sse(_gen(), model="m", message_id="id"))
    parsed = parse_sse(raw)
    md = next(e for e in parsed if e["event"] == "message_delta")
    assert md["data"]["usage"]["output_tokens"] > 0
