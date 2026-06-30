import sys
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI

from config import get_settings
from profiles import ProfileRegistry, get_or_load_config
from routers.health import router as health_router
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
app.include_router(health_router)
app.include_router(messages_router)
app.include_router(metrics_router)
app.include_router(models_router)
app.include_router(passthrough_router)


def main() -> None:
    """Console entrypoint: read config.toml and start uvicorn."""
    proxy_config, from_file = get_or_load_config()
    if not from_file:
        print(
            "warning: config.toml not found — using environment variable defaults.\n"
            "Copy config.example.toml to config.toml to configure the server.",
            file=sys.stderr,
        )
    uvicorn.run("main:app", host=proxy_config.server.host, port=proxy_config.server.port)


if __name__ == "__main__":
    main()
