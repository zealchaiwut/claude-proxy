import httpx
from fastapi import APIRouter, Request, Response

from routers._proxy_utils import filter_headers, proxy_request

router = APIRouter()


def _filter_headers(headers) -> dict[str, str]:
    return filter_headers(headers)


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


@router.post("/v1/messages/count_tokens")
async def count_tokens_passthrough(request: Request) -> Response:
    settings = request.app.state.settings
    return await proxy_request(
        request,
        f"{settings.upstream_base_url}/v1/messages/count_tokens",
        method="POST",
    )
