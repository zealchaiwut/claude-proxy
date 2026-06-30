"""Tests for issue #17: OpenAI SSE streaming consumer (services/openai_sse_consumer.py)."""
import asyncio
import json
from typing import AsyncIterator


from services.openai_sse_consumer import consume_openai_sse_stream, ContentEvent, FinishEvent, UsageEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(data: str) -> bytes:
    return f"data: {data}\n\n".encode()


def _delta_chunk(content: str | None = None, finish_reason: str | None = None, usage: dict | None = None) -> bytes:
    payload: dict = {"choices": [{"delta": {}}]}
    if content is not None:
        payload["choices"][0]["delta"]["content"] = content
    if finish_reason is not None:
        payload["choices"][0]["finish_reason"] = finish_reason
    if usage is not None:
        payload["usage"] = usage
    return _chunk(json.dumps(payload))


async def _stream(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def collect(byte_chunks) -> list:
    events = []
    async for ev in consume_openai_sse_stream(byte_chunks):
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# AC: yields content strings in arrival order
# ---------------------------------------------------------------------------

def test_content_yielded_in_order():
    chunks = [
        _delta_chunk(content="Hello"),
        _delta_chunk(content=", "),
        _delta_chunk(content="world"),
        _delta_chunk(finish_reason="stop"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert [e.text for e in content_events] == ["Hello", ", ", "world"]


# AC: finish_reason is captured
def test_finish_reason_yielded():
    chunks = [
        _delta_chunk(content="hi"),
        _delta_chunk(finish_reason="stop"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    finish_events = [e for e in events if isinstance(e, FinishEvent)]
    assert len(finish_events) == 1
    assert finish_events[0].reason == "stop"


# AC: [DONE] produces no further events
def test_done_sentinel_terminates_iteration():
    chunks = [
        _delta_chunk(content="text"),
        b"data: [DONE]\n\n",
        _delta_chunk(content="should-not-appear"),
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    texts = [e.text for e in events if isinstance(e, ContentEvent)]
    assert "should-not-appear" not in texts
    assert "text" in texts


# AC: concatenated text matches expected
def test_concatenated_text_matches():
    words = ["The ", "quick ", "brown ", "fox"]
    chunks = [_delta_chunk(content=w) for w in words]
    chunks.append(_delta_chunk(finish_reason="stop"))
    chunks.append(b"data: [DONE]\n\n")
    events = asyncio.run(collect(_stream(*chunks)))
    full = "".join(e.text for e in events if isinstance(e, ContentEvent))
    assert full == "The quick brown fox"


# AC: role-only delta (no content key) is silently skipped
def test_role_only_delta_skipped():
    role_chunk = _chunk(json.dumps({"choices": [{"delta": {"role": "assistant"}}]}))
    chunks = [
        role_chunk,
        _delta_chunk(content="hi"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].text == "hi"


# AC: null content is silently skipped
def test_null_content_delta_skipped():
    null_chunk = _chunk(json.dumps({"choices": [{"delta": {"content": None}}]}))
    chunks = [
        null_chunk,
        _delta_chunk(content="ok"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].text == "ok"


# AC: blank keepalive/comment lines are silently skipped
def test_blank_and_comment_lines_skipped():
    chunks = [
        b"\n",
        b": ping\n\n",
        b"  \n",
        _delta_chunk(content="after-keepalive"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].text == "after-keepalive"


# AC: usage object yielded when present in any chunk
def test_usage_event_yielded():
    usage = {"prompt_tokens": 10, "completion_tokens": 25, "total_tokens": 35}
    chunks = [
        _delta_chunk(content="text"),
        _chunk(json.dumps({"choices": [{"delta": {}}], "usage": usage})),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage_events) == 1
    assert usage_events[0].usage == usage


# AC: mid-JSON split across two read boundaries is correctly reassembled
def test_mid_json_split_across_chunks():
    full_payload = json.dumps({"choices": [{"delta": {"content": "reassembled"}}]})
    full_line = f"data: {full_payload}\n\n".encode()
    # split in the middle of the JSON string
    split_at = len(full_line) // 2
    part1 = full_line[:split_at]
    part2 = full_line[split_at:]

    async def split_stream() -> AsyncIterator[bytes]:
        yield part1
        yield part2
        yield b"data: [DONE]\n\n"

    events = asyncio.run(collect(split_stream()))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].text == "reassembled"


# AC: empty content string is not yielded (only non-empty)
def test_empty_content_string_not_yielded():
    chunks = [
        _delta_chunk(content=""),
        _delta_chunk(content="real"),
        b"data: [DONE]\n\n",
    ]
    events = asyncio.run(collect(_stream(*chunks)))
    content_events = [e for e in events if isinstance(e, ContentEvent)]
    assert len(content_events) == 1
    assert content_events[0].text == "real"
