import json
import os

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from routers._proxy_utils import filter_headers, proxy_request
from schemas.anthropic import MessagesRequest, TextBlock
from schemas.openai import ChatResponse
from services.translator import from_openai_response, live_stream_to_anthropic_sse, to_openai_request

router = APIRouter()


def _filter_headers(headers) -> dict[str, str]:
    return filter_headers(headers)


def _wants_stream(request: Request, body: dict) -> bool:
    if request.headers.get("accept", "").lower() == "text/event-stream":
        return True
    return body.get("stream") is True


def _count_tokens_heuristic(body: dict) -> int:
    """Estimate token count from messages using total_chars / 4 heuristic."""
    messages = body.get("messages", [])
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
    return max(1, total_chars // 4)


@router.post("/v1/messages")
async def messages_passthrough(request: Request) -> Response:
    body_bytes = await request.body()
    headers = _filter_headers(request.headers)

    try:
        body_json = json.loads(body_bytes)
    except (ValueError, TypeError):
        body_json = {}

    profile = os.getenv("CCPROXY_PROFILE", "anthropic")

    if profile == "openai":
        return await _handle_openai_mode(request, body_json)

    # anthropic mode: byte-for-byte passthrough
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    if not _wants_stream(request, body_json):
        upstream_resp = await client.post(
            f"{settings.upstream_base_url}/v1/messages",
            content=body_bytes,
            headers=headers,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_filter_headers(upstream_resp.headers),
        )

    # streaming passthrough for anthropic mode
    stream_ctx = client.stream(
        "POST",
        f"{settings.upstream_base_url}/v1/messages",
        content=body_bytes,
        headers=headers,
    )
    upstream_resp = await stream_ctx.__aenter__()

    async def _iter():
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)

    return StreamingResponse(
        _iter(),
        status_code=upstream_resp.status_code,
        headers=_filter_headers(upstream_resp.headers),
    )


def _get_system_text(req: MessagesRequest) -> str | None:
    if req.system is None:
        return None
    if isinstance(req.system, str):
        return req.system
    return "".join(b.text for b in req.system if isinstance(b, TextBlock))


async def _handle_openai_mode(request: Request, body_json: dict) -> Response:
    # Read credentials at request time — never log these values
    openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
    tool_mode = os.getenv("CCPROXY_TOOL_MODE", "native")

    client: httpx.AsyncClient = request.app.state.http_client
    anthropic_req = MessagesRequest(**body_json)

    # XML mode: inject tool spec into system prompt and clear native tools
    if tool_mode == "xml" and anthropic_req.tools:
        from services.xml_tool_mode import build_xml_system_prompt
        xml_system = build_xml_system_prompt(_get_system_text(anthropic_req), list(anthropic_req.tools))
        anthropic_req = anthropic_req.model_copy(update={"system": xml_system, "tools": None, "tool_choice": None})

    if _wants_stream(request, body_json):
        return await _handle_openai_stream(
            client, anthropic_req, openai_base_url, openai_api_key, openai_model, tool_mode=tool_mode
        )

    openai_req = to_openai_request(anthropic_req, model=openai_model)
    upstream_resp = await client.post(
        f"{openai_base_url}/chat/completions",
        content=openai_req.model_dump_json().encode(),
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
    )

    openai_resp = ChatResponse(**json.loads(upstream_resp.content))
    anthropic_resp = from_openai_response(openai_resp)

    # XML mode: post-process — extract tool calls from text blocks
    if tool_mode == "xml":
        from services.xml_tool_mode import parse_xml_tool_calls
        from schemas.anthropic import ToolUseBlock
        full_text = "".join(b.text for b in anthropic_resp.content if isinstance(b, TextBlock))
        cleaned, tool_blocks = parse_xml_tool_calls(full_text)
        if tool_blocks:
            new_content: list = []
            if cleaned:
                new_content.append(TextBlock(type="text", text=cleaned))
            new_content.extend(tool_blocks)
            anthropic_resp = anthropic_resp.model_copy(
                update={"content": new_content, "stop_reason": "tool_use"}
            )

    return Response(
        content=anthropic_resp.model_dump_json(),
        status_code=200,
        media_type="application/json",
    )


async def _handle_openai_stream(
    client: httpx.AsyncClient,
    anthropic_req: MessagesRequest,
    openai_base_url: str,
    openai_api_key: str,
    openai_model: str,
    *,
    tool_mode: str = "native",
) -> Response:
    """Return a live StreamingResponse translating OpenAI SSE to Anthropic SSE."""
    from schemas.openai import ChatRequest

    openai_req = to_openai_request(anthropic_req, model=openai_model)
    req_body = ChatRequest(
        model=openai_model,
        messages=openai_req.messages,
        max_tokens=openai_req.max_tokens,
        stream=True,
    ).model_dump_json().encode()

    stream_ctx = client.stream(
        "POST",
        f"{openai_base_url}/chat/completions",
        content=req_body,
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
    )
    upstream = await stream_ctx.__aenter__()

    # Pre-content error: upstream returned non-2xx before any SSE data
    if upstream.status_code >= 400:
        try:
            body = b""
            async for chunk in upstream.aiter_bytes():
                body += chunk
        finally:
            await stream_ctx.__aexit__(None, None, None)
        return Response(content=body, status_code=upstream.status_code, media_type="application/json")

    if tool_mode == "xml":
        from services.xml_tool_mode import xml_buffered_sse

        async def _translate_xml():
            try:
                async for frame in xml_buffered_sse(upstream.aiter_bytes(), model=openai_model):
                    yield frame.encode() if isinstance(frame, str) else frame
            finally:
                await stream_ctx.__aexit__(None, None, None)

        return StreamingResponse(_translate_xml(), status_code=200, media_type="text/event-stream")

    async def _translate():
        content_sent = False
        try:
            async for frame in live_stream_to_anthropic_sse(
                upstream.aiter_bytes(),
                model=openai_model,
            ):
                if "content_block_delta" in frame:
                    content_sent = True
                yield frame.encode() if isinstance(frame, str) else frame
        except Exception as exc:
            if content_sent:
                err_frame = f'event: error\ndata: {json.dumps({"type": "error", "error": {"type": "stream_error", "message": str(exc)}})}\n\n'
                yield err_frame.encode()
        finally:
            await stream_ctx.__aexit__(None, None, None)

    return StreamingResponse(
        _translate(),
        status_code=200,
        media_type="text/event-stream",
    )


@router.post("/v1/messages/count_tokens")
async def count_tokens_passthrough(request: Request) -> Response:
    profile = os.getenv("CCPROXY_PROFILE", "anthropic")

    if profile == "openai":
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes)
        except (ValueError, TypeError):
            body_json = {}
        token_count = _count_tokens_heuristic(body_json)
        return Response(
            content=json.dumps({"input_tokens": token_count}),
            status_code=200,
            media_type="application/json",
        )

    # anthropic mode: passthrough unchanged
    settings = request.app.state.settings
    return await proxy_request(
        request,
        f"{settings.upstream_base_url}/v1/messages/count_tokens",
        method="POST",
    )
