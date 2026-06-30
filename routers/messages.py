import json

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from routers._proxy_utils import filter_headers, proxy_request

router = APIRouter()


def _filter_headers(headers) -> dict[str, str]:
    return filter_headers(headers)


def _wants_stream(request: Request, body: dict) -> bool:
    if request.headers.get("accept", "").lower() == "text/event-stream":
        return True
    return body.get("stream") is True


@router.post("/v1/messages")
async def messages_passthrough(request: Request) -> Response:
    body_bytes = await request.body()
    headers = _filter_headers(request.headers)

    try:
        body_json = json.loads(body_bytes)
    except (ValueError, TypeError):
        body_json = {}

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

    # Enter the streaming context to read response headers/status before yielding body.
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


@router.post("/v1/messages/count_tokens")
async def count_tokens_passthrough(request: Request) -> Response:
    settings = request.app.state.settings
    return await proxy_request(
        request,
        f"{settings.upstream_base_url}/v1/messages/count_tokens",
        method="POST",
    )
