"""GET /metrics endpoint — per-profile rolling summary (issue #43)."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/metrics")
def get_metrics(request: Request) -> dict:
    collector = getattr(request.app.state, "metrics_collector", None)
    if collector is None:
        return {"profiles": {}}
    return {"profiles": collector.snapshot()}
