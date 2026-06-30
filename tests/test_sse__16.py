"""Tests for issue #16: Anthropic SSE streaming event emitter service."""
from __future__ import annotations

import asyncio
import json

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_sse_text(text: str) -> list[dict]:
    """Parse raw SSE text into list of {'event': str, 'data': dict} dicts."""
    events = []
    for block in text.strip().split("\n\n"):
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


async def collect_stream(gen) -> str:
    parts = []
    async for frame in gen:
        parts.append(frame)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Fixtures / shared inputs
# ---------------------------------------------------------------------------

DELTAS = ["Hello", " world", "!"]
STOP_REASON = "end_turn"
USAGE = {"input_tokens": 10, "output_tokens": 3}
MODEL = "claude-3-haiku-20240307"
MESSAGE_ID = "msg_test123"


async def _simple_deltas():
    for d in DELTAS:
        yield d


# ---------------------------------------------------------------------------
# AC1 / UAT1: exactly 7 SSE events in correct order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_count_and_order():
    """AC2: exactly message_start → content_block_start → 3×content_block_delta
    → content_block_stop → message_delta → message_stop (7 events total)."""
    from services.sse import anthropic_sse_stream

    text = await collect_stream(
        anthropic_sse_stream(
            _simple_deltas(),
            stop_reason=STOP_REASON,
            usage=USAGE,
            model=MODEL,
            message_id=MESSAGE_ID,
        )
    )
    events = parse_sse_text(text)
    expected_types = [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    actual_types = [e["event"] for e in events]
    assert actual_types == expected_types, f"Event order mismatch: {actual_types}"


# ---------------------------------------------------------------------------
# AC3: SSE framing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_frame_format():
    """AC3: every event is framed as 'event: <type>\\ndata: <json>\\n\\n'."""
    from services.sse import anthropic_sse_stream

    frames: list[str] = []
    async for frame in anthropic_sse_stream(
        _simple_deltas(),
        stop_reason=STOP_REASON,
        usage=USAGE,
        model=MODEL,
        message_id=MESSAGE_ID,
    ):
        frames.append(frame)

    for frame in frames:
        parts = frame.split("\n")
        assert parts[0].startswith("event: "), f"Frame missing 'event:' line: {frame!r}"
        assert parts[1].startswith("data: "), f"Frame missing 'data:' line: {frame!r}"
        assert frame.endswith("\n\n"), f"Frame must end with blank line: {frame!r}"
        json.loads(parts[1][6:])  # must be valid JSON


# ---------------------------------------------------------------------------
# AC4 / UAT2: message_start data shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_start_shape():
    """AC4: message_start contains id, type=message, role=assistant, model, content=[], usage.input_tokens."""
    from services.sse import anthropic_sse_stream

    text = await collect_stream(
        anthropic_sse_stream(
            _simple_deltas(),
            stop_reason=STOP_REASON,
            usage=USAGE,
            model=MODEL,
            message_id=MESSAGE_ID,
        )
    )
    events = parse_sse_text(text)
    start = events[0]
    assert start["event"] == "message_start"
    msg = start["data"]["message"]
    assert msg["id"] == MESSAGE_ID
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["model"] == MODEL
    assert msg["content"] == []
    assert msg["usage"]["input_tokens"] == USAGE["input_tokens"]


# ---------------------------------------------------------------------------
# AC5: content_block_start shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_block_start_shape():
    """AC5: content_block_start has index=0 and content_block={type:text, text:''}."""
    from services.sse import anthropic_sse_stream

    text = await collect_stream(
        anthropic_sse_stream(
            _simple_deltas(),
            stop_reason=STOP_REASON,
            usage=USAGE,
            model=MODEL,
            message_id=MESSAGE_ID,
        )
    )
    events = parse_sse_text(text)
    cbs = events[1]
    assert cbs["event"] == "content_block_start"
    d = cbs["data"]
    assert d["index"] == 0
    assert d["content_block"] == {"type": "text", "text": ""}


# ---------------------------------------------------------------------------
# AC6 / UAT3: content_block_delta shapes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_block_delta_shapes():
    """AC6: each delta has index=0, delta.type=text_delta, delta.text matches source."""
    from services.sse import anthropic_sse_stream

    text = await collect_stream(
        anthropic_sse_stream(
            _simple_deltas(),
            stop_reason=STOP_REASON,
            usage=USAGE,
            model=MODEL,
            message_id=MESSAGE_ID,
        )
    )
    events = parse_sse_text(text)
    deltas = [e for e in events if e["event"] == "content_block_delta"]
    assert len(deltas) == len(DELTAS)
    for i, (evt, expected_text) in enumerate(zip(deltas, DELTAS)):
        d = evt["data"]
        assert d["index"] == 0, f"delta {i}: expected index=0"
        assert d["delta"]["type"] == "text_delta", f"delta {i}: wrong delta type"
        assert d["delta"]["text"] == expected_text, f"delta {i}: text mismatch"


# ---------------------------------------------------------------------------
# AC7: content_block_stop shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_content_block_stop_shape():
    """AC7: content_block_stop has index=0."""
    from services.sse import anthropic_sse_stream

    text = await collect_stream(
        anthropic_sse_stream(
            _simple_deltas(),
            stop_reason=STOP_REASON,
            usage=USAGE,
            model=MODEL,
            message_id=MESSAGE_ID,
        )
    )
    events = parse_sse_text(text)
    stop = next(e for e in events if e["event"] == "content_block_stop")
    assert stop["data"]["index"] == 0


# ---------------------------------------------------------------------------
# AC8 / UAT4: message_delta shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_delta_shape():
    """AC8: message_delta has delta.stop_reason, delta.stop_sequence=null, usage.output_tokens."""
    from services.sse import anthropic_sse_stream

    text = await collect_stream(
        anthropic_sse_stream(
            _simple_deltas(),
            stop_reason=STOP_REASON,
            usage=USAGE,
            model=MODEL,
            message_id=MESSAGE_ID,
        )
    )
    events = parse_sse_text(text)
    md = next(e for e in events if e["event"] == "message_delta")
    d = md["data"]
    assert d["delta"]["stop_reason"] == STOP_REASON
    assert d["delta"]["stop_sequence"] is None
    assert d["usage"]["output_tokens"] == USAGE["output_tokens"]


# ---------------------------------------------------------------------------
# AC9: message_stop shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_stop_shape():
    """AC9: message_stop data is {type: message_stop}."""
    from services.sse import anthropic_sse_stream

    text = await collect_stream(
        anthropic_sse_stream(
            _simple_deltas(),
            stop_reason=STOP_REASON,
            usage=USAGE,
            model=MODEL,
            message_id=MESSAGE_ID,
        )
    )
    events = parse_sse_text(text)
    ms = next(e for e in events if e["event"] == "message_stop")
    assert ms["data"] == {"type": "message_stop"}


# ---------------------------------------------------------------------------
# AC10 / UAT5: ping events emitted before first content event when configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ping_appears_before_first_content_delta():
    """AC10: with ping_interval set, at least one ping appears before first content_block_delta."""
    from services.sse import anthropic_sse_stream

    async def slow_deltas():
        await asyncio.sleep(0.05)
        yield "Hello"

    text = await collect_stream(
        anthropic_sse_stream(
            slow_deltas(),
            stop_reason="end_turn",
            usage={"input_tokens": 1, "output_tokens": 1},
            model="claude-3-haiku",
            message_id="msg_ping",
            ping_interval=0.01,
        )
    )
    events = parse_sse_text(text)
    event_types = [e["event"] for e in events]
    ping_indices = [i for i, t in enumerate(event_types) if t == "ping"]
    delta_indices = [i for i, t in enumerate(event_types) if t == "content_block_delta"]
    assert len(ping_indices) >= 1, "Expected at least one ping event"
    assert min(ping_indices) < min(delta_indices), "Ping must appear before first content_block_delta"


# ---------------------------------------------------------------------------
# AC11: no HTTP framework imports
# ---------------------------------------------------------------------------


def test_no_http_framework_imports():
    """AC11: services.sse must not import FastAPI, Starlette, or aiohttp."""
    import sys

    # Remove cached module to inspect freshly
    for key in list(sys.modules.keys()):
        if "services.sse" in key:
            del sys.modules[key]

    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "services" / "sse.py"
    tree = ast.parse(src.read_text())
    forbidden = {"fastapi", "starlette", "aiohttp"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                assert root not in forbidden, f"Forbidden import: from {node.module}"


# ---------------------------------------------------------------------------
# AC12 / UAT6: clean import without web framework
# ---------------------------------------------------------------------------


def test_import_without_web_framework():
    """AC12: services.sse imports cleanly (no web framework required at import time)."""
    import importlib
    import sys

    for key in list(sys.modules.keys()):
        if "services.sse" in key:
            del sys.modules[key]

    importlib.import_module("services.sse")
