import httpx
from fastapi import Request, Response

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

_502 = Response(
    content='{"error":"bad_gateway","message":"upstream unreachable"}',
    status_code=502,
    media_type="application/json",
)
_504 = Response(
    content='{"error":"gateway_timeout","message":"upstream timed out"}',
    status_code=504,
    media_type="application/json",
)


def filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP}


async def proxy_request(
    request: Request,
    upstream_url: str,
    method: str | None = None,
) -> Response:
    method = method or request.method
    body = await request.body()
    headers = filter_headers(request.headers)
    client: httpx.AsyncClient = request.app.state.http_client

    try:
        if method == "POST":
            upstream_resp = await client.post(upstream_url, content=body, headers=headers)
        elif method == "GET":
            upstream_resp = await client.get(upstream_url, headers=headers)
        else:
            upstream_resp = await client.request(
                method, upstream_url, content=body, headers=headers
            )
    except httpx.ConnectError:
        return _502
    except httpx.ReadTimeout:
        return _504

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=filter_headers(upstream_resp.headers),
    )
