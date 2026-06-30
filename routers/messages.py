import httpx
from fastapi import APIRouter, Request, Response

router = APIRouter()

_HOP_BY_HOP = frozenset({
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
})


def _filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


@router.post("/v1/messages")
async def messages_passthrough(request: Request) -> Response:
    body = await request.body()
    headers = _filter_headers(request.headers)

    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client

    upstream_resp = await client.post(
        f"{settings.upstream_base_url}/v1/messages",
        content=body,
        headers=headers,
    )

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=_filter_headers(upstream_resp.headers),
    )
