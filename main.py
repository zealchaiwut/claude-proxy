import os
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import Depends, FastAPI

from config import Settings, get_settings
from routers.messages import router as messages_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    client = httpx.AsyncClient(timeout=httpx.Timeout(settings.upstream_read_timeout))
    app.state.http_client = client
    yield
    await client.aclose()


app = FastAPI(lifespan=lifespan)
app.include_router(messages_router)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "upstream": settings.upstream_base_url}


if __name__ == "__main__":
    host = os.getenv("CCPROXY_HOST", "127.0.0.1")
    port = int(os.getenv("CCPROXY_PORT", "8788"))
    uvicorn.run("main:app", host=host, port=port)
