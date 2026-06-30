from fastapi import APIRouter, Request, Response

from routers._proxy_utils import proxy_request

router = APIRouter()


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def catchall_passthrough(path: str, request: Request) -> Response:
    settings = request.app.state.settings
    return await proxy_request(
        request,
        f"{settings.upstream_base_url}/v1/{path}",
    )
