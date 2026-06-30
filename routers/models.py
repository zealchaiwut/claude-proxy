from fastapi import APIRouter, Request, Response

from routers._proxy_utils import proxy_request

router = APIRouter()


@router.get("/v1/models")
async def models_passthrough(request: Request) -> Response:
    settings = request.app.state.settings
    return await proxy_request(
        request,
        f"{settings.upstream_base_url}/v1/models",
        method="GET",
    )
