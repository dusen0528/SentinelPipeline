from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()


def render_metrics() -> str:
    """
    메트릭 텍스트를 렌더링합니다.
    현재는 스텁이며, 추후 Prometheus 포맷으로 확장 가능합니다.
    """
    return "# metrics stub\n"


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics_endpoint() -> str:
    return render_metrics()

