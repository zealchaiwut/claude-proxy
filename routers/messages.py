import json
import os
import time
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from profiles import ProfileRegistry, resolve_profile_name
from routers._proxy_utils import filter_headers, proxy_request
from schemas.anthropic import MessagesRequest, TextBlock
from schemas.openai import ChatResponse
from services.request_logger import RequestLogger
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


def _get_profile_name(request: Request) -> str:
    """Resolve per-request profile name using 4-level precedence chain."""
    return resolve_profile_name(
        header=request.headers.get("x-ccproxy-profile"),
        query_param=request.query_params.get("profile"),
    )


def _hostname(url: str) -> str:
    return urlparse(url).hostname or url


async def _passthrough(
    request: Request,
    body_bytes: bytes,
    body_json: dict,
    headers: dict[str, str],
    upstream_base: str,
) -> Response:
    """Forward POST /v1/messages to upstream_base, streaming or non-streaming."""
    client: httpx.AsyncClient = request.app.state.http_client

    if not _wants_stream(request, body_json):
        upstream_resp = await client.post(
            f"{upstream_base}/v1/messages",
            content=body_bytes,
            headers=headers,
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=_filter_headers(upstream_resp.headers),
        )

    stream_ctx = client.stream(
        "POST",
        f"{upstream_base}/v1/messages",
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


async def _dispatch(
    request: Request,
    body_bytes: bytes,
    body_json: dict,
    headers: dict[str, str],
    profile_name: str,
) -> tuple[Response, dict]:
    """Route the request and return (response, log_meta).

    log_meta contains: profile_kind, upstream_model, upstream_host.
    """
    config_from_file: bool = getattr(request.app.state, "config_from_file", False)
    if config_from_file:
        registry: ProfileRegistry | None = getattr(request.app.state, "profile_registry", None)
        if registry is not None:
            try:
                kind, upstream_url, api_key, model, model_map = registry.resolve(profile_name)
                client_model = body_json.get("model", "")
                if kind == "openai":
                    upstream_model = (
                        model_map.get(client_model)
                        or model
                        or os.getenv("OPENAI_MODEL", "gpt-4o")
                    )
                    response = await _handle_openai_mode(
                        request,
                        body_json,
                        openai_base_url=upstream_url,
                        openai_api_key=api_key,
                        openai_model=upstream_model,
                    )
                    return response, {
                        "profile_kind": "openai",
                        "upstream_model": upstream_model,
                        "upstream_host": _hostname(upstream_url),
                    }
                # passthrough path (with optional model_map rewrite)
                upstream_model = client_model
                if model_map and client_model in model_map:
                    upstream_model = model_map[client_model]
                    body_json = {**body_json, "model": upstream_model}
                    body_bytes = json.dumps(body_json).encode()
                response = await _passthrough(request, body_bytes, body_json, headers, upstream_url)
                return response, {
                    "profile_kind": "passthrough",
                    "upstream_model": upstream_model,
                    "upstream_host": _hostname(upstream_url),
                }
            except KeyError:
                pass  # profile not in registry; fall through to legacy

    # Legacy path (env-var based, backward-compatible with M1/M2 behavior)
    if profile_name == "openai":
        openai_base_url = os.getenv("OPENAI_BASE_URL", "")
        upstream_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        response = await _handle_openai_mode(request, body_json)
        return response, {
            "profile_kind": "openai",
            "upstream_model": upstream_model,
            "upstream_host": _hostname(openai_base_url),
        }

    settings = request.app.state.settings
    response = await _passthrough(
        request, body_bytes, body_json, headers, settings.upstream_base_url
    )
    return response, {
        "profile_kind": "passthrough",
        "upstream_model": body_json.get("model", ""),
        "upstream_host": _hostname(settings.upstream_base_url),
    }


def _attach_logging(
    request_logger: RequestLogger,
    response: Response,
    *,
    profile_name: str,
    requested_model: str,
    method: str,
    path: str,
    start: float,
    log_meta: dict,
) -> Response:
    """Attach logging to response; wraps StreamingResponse body iterator for streamed requests."""
    if isinstance(response, StreamingResponse):
        original = response.body_iterator
        _start = start

        async def _wrapped():
            try:
                async for chunk in original:
                    yield chunk
            finally:
                latency_ms = (time.monotonic() - _start) * 1000
                record = request_logger.make_record(
                    profile_name=profile_name,
                    requested_model=requested_model,
                    method=method,
                    path=path,
                    status=response.status_code,
                    latency_ms=latency_ms,
                    streamed=True,
                    **log_meta,
                )
                request_logger.emit(record)

        response.body_iterator = _wrapped()
    else:
        latency_ms = (time.monotonic() - start) * 1000
        record = request_logger.make_record(
            profile_name=profile_name,
            requested_model=requested_model,
            method=method,
            path=path,
            status=response.status_code,
            latency_ms=latency_ms,
            streamed=False,
            **log_meta,
        )
        request_logger.emit(record)

    return response


@router.post("/v1/messages")
async def messages_passthrough(request: Request) -> Response:
    start = time.monotonic()
    body_bytes = await request.body()
    headers = _filter_headers(request.headers)

    try:
        body_json = json.loads(body_bytes)
    except (ValueError, TypeError):
        body_json = {}

    profile_name = _get_profile_name(request)
    requested_model = body_json.get("model", "")

    response, log_meta = await _dispatch(request, body_bytes, body_json, headers, profile_name)

    request_logger: RequestLogger | None = getattr(request.app.state, "request_logger", None)
    if request_logger is not None:
        response = _attach_logging(
            request_logger,
            response,
            profile_name=profile_name,
            requested_model=requested_model,
            method=request.method,
            path=request.url.path,
            start=start,
            log_meta=log_meta,
        )

    return response


def _get_system_text(req: MessagesRequest) -> str | None:
    if req.system is None:
        return None
    if isinstance(req.system, str):
        return req.system
    return "".join(b.text for b in req.system if isinstance(b, TextBlock))


async def _handle_openai_mode(
    request: Request,
    body_json: dict,
    *,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
) -> Response:
    # Fall back to env vars when values not supplied (legacy mode)
    if openai_base_url is None:
        openai_base_url = os.getenv("OPENAI_BASE_URL", "")
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if openai_model is None:
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
    body_bytes = await request.body()
    try:
        body_json = json.loads(body_bytes)
    except (ValueError, TypeError):
        body_json = {}

    profile_name = _get_profile_name(request)

    # Registry path
    config_from_file: bool = getattr(request.app.state, "config_from_file", False)
    if config_from_file:
        registry: ProfileRegistry | None = getattr(request.app.state, "profile_registry", None)
        if registry is not None:
            try:
                kind, upstream_url, _, _, _ = registry.resolve(profile_name)
                if kind == "openai":
                    token_count = _count_tokens_heuristic(body_json)
                    return Response(
                        content=json.dumps({"input_tokens": token_count}),
                        status_code=200,
                        media_type="application/json",
                    )
                return await proxy_request(
                    request,
                    f"{upstream_url}/v1/messages/count_tokens",
                    method="POST",
                )
            except KeyError:
                pass

    # Legacy path
    if profile_name == "openai":
        token_count = _count_tokens_heuristic(body_json)
        return Response(
            content=json.dumps({"input_tokens": token_count}),
            status_code=200,
            media_type="application/json",
        )

    settings = request.app.state.settings
    return await proxy_request(
        request,
        f"{settings.upstream_base_url}/v1/messages/count_tokens",
        method="POST",
    )
