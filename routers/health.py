"""GET /health and GET /ready lifecycle endpoints (issue #50)."""
import asyncio
import time
from urllib.parse import urlparse

from fastapi import APIRouter, Depends

from config import Settings, get_settings
from profiles import get_or_load_config, resolve_profile_name

router = APIRouter()

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("claude-proxy")
except Exception:
    _VERSION = "0.1.0"

_ready_cache: dict = {"result": None, "expires_at": 0.0}


async def _tcp_probe(host: str, port: int, timeout: float) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


def _active_upstream(settings: Settings) -> tuple[str, str]:
    """Return (profile_name, upstream_url) for the active default profile."""
    profile_name = resolve_profile_name(None, None)
    proxy_config, _ = get_or_load_config()
    if profile_name in proxy_config.profiles:
        return profile_name, proxy_config.profiles[profile_name].upstream
    return profile_name, settings.upstream_base_url


@router.get("/health")
def health(settings: Settings = Depends(get_settings)):
    profile_name, upstream = _active_upstream(settings)
    return {
        "status": "ok",
        "version": _VERSION,
        "active_default_profile": profile_name,
        "upstream": upstream,
    }


@router.get("/ready")
async def ready(settings: Settings = Depends(get_settings)):
    now = time.monotonic()
    if _ready_cache["result"] is not None and now < _ready_cache["expires_at"]:
        return _ready_cache["result"]

    profile_name, upstream = _active_upstream(settings)

    parsed = urlparse(upstream)
    host = parsed.hostname or upstream
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    is_reachable = await _tcp_probe(host, port, timeout=2.0)
    result = {
        "status": "ok" if is_reachable else "degraded",
        "profile": profile_name,
    }

    _ready_cache["result"] = result
    _ready_cache["expires_at"] = now + settings.ready_cache_ttl

    return result
