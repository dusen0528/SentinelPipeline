"""
헬스 체크 및 메트릭 엔드포인트
"""

from __future__ import annotations

import os
import time
from typing import Any

import psutil
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


@router.get("/api/stats/system")
async def get_system_stats() -> dict[str, Any]:
    """시스템 리소스 사용률을 반환합니다."""
    try:
        # 현재 프로세스 정보
        process = psutil.Process(os.getpid())
        process_cpu = process.cpu_percent(interval=0.1)
        process_memory = process.memory_info()
        process_memory_mb = process_memory.rss / (1024 * 1024)
        
        # 전체 시스템 정보
        system_cpu = psutil.cpu_percent(interval=0.1)
        system_memory = psutil.virtual_memory()
        system_memory_total_mb = system_memory.total / (1024 * 1024)
        system_memory_used_mb = system_memory.used / (1024 * 1024)
        system_memory_percent = system_memory.percent
        
        return {
            "success": True,
            "data": {
                "system": {
                    "cpu_percent": round(system_cpu, 1),
                    "memory_total_mb": round(system_memory_total_mb, 0),
                    "memory_used_mb": round(system_memory_used_mb, 0),
                    "memory_percent": round(system_memory_percent, 1),
                },
                "process": {
                    "cpu_percent": round(process_cpu, 1),
                    "memory_mb": round(process_memory_mb, 0),
                }
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": {
                "system": {
                    "cpu_percent": 0,
                    "memory_total_mb": 0,
                    "memory_used_mb": 0,
                    "memory_percent": 0,
                },
                "process": {
                    "cpu_percent": 0,
                    "memory_mb": 0,
                }
            }
        }

