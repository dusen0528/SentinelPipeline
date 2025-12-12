"""
헬스 체크 및 메트릭 엔드포인트
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends

from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.interface.api.dependencies import get_stream_manager

router = APIRouter()


def _summarize_streams(manager: StreamManager) -> dict[str, Any]:
    """스트림 상태 요약 정보를 반환합니다."""
    summary: dict[str, Any] = {}
    for state in manager.get_all_streams():
        summary[state.stream_id] = {
            "status": state.status.value,
            "fps": round(state.stats.fps, 1),
            "last_frame_ts": state.stats.last_frame_ts,
            "error_count": state.stats.error_count,
        }
    return summary


@router.get("/health/live")
async def health_live() -> dict[str, str]:
    """라이브니스 체크 (단순 200)."""
    return {"status": "live"}


@router.get("/health/ready")
async def health_ready(
    manager: StreamManager = Depends(get_stream_manager),
) -> dict[str, Any]:
    """레디니스 체크 (주요 컴포넌트 초기화 여부)."""
    return {
        "status": "ready",
        "streams": len(manager.get_all_streams()),
    }


@router.get("/health")
async def health(
    manager: StreamManager = Depends(get_stream_manager),
) -> dict[str, Any]:
    """통합 헬스 체크 (간단한 상태 + 스트림 요약)."""
    return {
        "status": "healthy",
        "ts": time.time(),
        "streams": _summarize_streams(manager),
    }

