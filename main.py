import os

import uvicorn
from fastapi import Depends, FastAPI

from config import Settings, get_settings

app = FastAPI()


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "upstream": settings.upstream_base_url}


if __name__ == "__main__":
    host = os.getenv("CCPROXY_HOST", "127.0.0.1")
    port = int(os.getenv("CCPROXY_PORT", "8788"))
    uvicorn.run("main:app", host=host, port=port)
