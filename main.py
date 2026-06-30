from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import Depends, FastAPI

from config import Settings, get_settings
from profiles import ProfileRegistry, get_or_load_config
from routers.messages import router as messages_router
from routers.metrics import router as metrics_router
from routers.models import router as models_router
from routers.passthrough import router as passthrough_router
from services.metrics_collector import MetricsCollector
from services.request_logger import RequestLogger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    proxy_config, config_from_file = get_or_load_config()
    app.state.proxy_config = proxy_config
    app.state.config_from_file = config_from_file
    app.state.profile_registry = ProfileRegistry(proxy_config)

    app.state.request_logger = RequestLogger()
    app.state.metrics_collector = MetricsCollector()

    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.upstream_read_timeout))
    app.state.http_client = client
    yield
    await client.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(messages_router)
app.include_router(metrics_router)
app.include_router(models_router)
app.include_router(passthrough_router)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "upstream": settings.upstream_base_url}


if __name__ == "__main__":
    proxy_config, _ = get_or_load_config()
    host = proxy_config.server.host
    port = proxy_config.server.port
    uvicorn.run("main:app", host=host, port=port)
