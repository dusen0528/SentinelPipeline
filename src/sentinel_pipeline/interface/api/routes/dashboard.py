from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psutil
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from sentinel_pipeline.application.pipeline.pipeline import PipelineEngine
from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.domain.models.stream import StreamState
from sentinel_pipeline.interface.api.dependencies import (
    get_pipeline_engine,
    get_stream_manager,
)

router = APIRouter()
_process = psutil.Process(os.getpid())


# --- DTOs ---

class SystemStats(BaseModel):
    cpu_percent: float
    memory_total_mb: float
    memory_used_mb: float
    memory_percent: float

class ProcessStats(BaseModel):
    cpu_percent: float
    memory_mb: float

class DashboardStream(BaseModel):
    stream_id: str
    status: str
    is_healthy: bool
    input_url: str | None = None
    output_url: str | None = None
    fps: float
    avg_latency_ms: float
    frame_count: int
    event_count: int
    error_count: int
    last_error: str | None = None
    config: dict[str, Any]

class DashboardStatsResponse(BaseModel):
    success: bool = True
    system: SystemStats
    process: ProcessStats
    streams: list[DashboardStream]
    modules: dict[str, Any]


# --- Routes ---

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def admin_dashboard_root(_: Request) -> HTMLResponse:
    """Serves the main dashboard HTML file."""
    base_dir = Path(__file__).resolve().parent.parent / "static" / "dashboard"
    html_path = base_dir / "index.html"
    return FileResponse(html_path)

@router.get("/api/dashboard/stats", response_model=DashboardStatsResponse)
async def get_dashboard_stats(
    stream_manager: StreamManager = Depends(get_stream_manager),
    pipeline_engine: PipelineEngine = Depends(get_pipeline_engine),
) -> dict[str, Any]:
    """
    Aggregates all necessary data for the dashboard in a single endpoint.
    """
    # 1. System and Process Stats
    sys_mem = psutil.virtual_memory()
    system_stats = SystemStats(
        cpu_percent=psutil.cpu_percent(),
        memory_total_mb=sys_mem.total / (1024 * 1024),
        memory_used_mb=sys_mem.used / (1024 * 1024),
        memory_percent=sys_mem.percent,
    )
    proc_mem_info = _process.memory_info()
    process_stats = ProcessStats(
        cpu_percent=_process.cpu_percent(),
        memory_mb=proc_mem_info.rss / (1024 * 1024),
    )

    # 2. Stream Stats
    stream_states = stream_manager.get_all_streams()
    dashboard_streams = [
        DashboardStream(
            stream_id=s.stream_id,
            status=s.status.value,
            is_healthy=s.is_healthy,
            input_url=s.config.rtsp_url,
            output_url=s.config.output_url,
            fps=s.stats.fps,
            avg_latency_ms=s.stats.avg_latency_ms,
            frame_count=s.stats.frame_count,
            event_count=s.stats.event_count,
            error_count=s.stats.error_count,
            last_error=s.last_error,
            config=s.config.to_dict(), # Include full config
        )
        for s in stream_states
    ]

    # 3. Module Stats
    module_stats = pipeline_engine.get_stats().get("modules", {})

    return {
        "system": system_stats,
        "process": process_stats,
        "streams": dashboard_streams,
        "modules": module_stats,
    }

