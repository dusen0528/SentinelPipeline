"""
FastAPI 애플리케이션 팩토리

Interface Layer에서만 FastAPI에 의존합니다.
예외 핸들러, CORS, 라우터 등록을 담당합니다.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import ValidationError

from sentinel_pipeline.common.errors import SentinelError
from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.interface.api.routes import (
    admin_ws,
    admin_debug,
    config,
    health,
    metrics,
    video,
    dashboard,
    audio,
)
from fastapi.staticfiles import StaticFiles
from pathlib import Path

logger = get_logger(__name__)


def create_app(allowed_origins: Iterable[str] | None = None) -> FastAPI:
    """
    FastAPI 애플리케이션을 생성합니다.

    Args:
        allowed_origins: CORS 허용 오리진 목록
    """
    app = FastAPI(title="SentinelPipeline API", version="0.1.0")

    @app.on_event("startup")
    async def on_startup():
        """애플리케이션 시작 시 이벤트 루프를 StreamManager에 설정합니다."""
        logger.info("Setting event loop for StreamManager")
        loop = asyncio.get_running_loop()
        if hasattr(app.state, "app_context"):
            app.state.app_context.stream_manager.set_event_loop(loop)
            app.state.app_context.audio_manager.set_event_loop(loop)
        else:
            logger.warning("AppContext not found in app.state")

    # CORS
    origins = list(allowed_origins) if allowed_origins else []
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 예외 핸들러 등록
    @app.exception_handler(SentinelError)
    async def handle_sentinel_error(_: Request, exc: SentinelError) -> JSONResponse:
        """SentinelError → JSON 응답 매핑."""
        logger.error(
            "SentinelError 발생",
            code=exc.code.value,
            error_msg=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.http_status,
            content={"success": False, "message": exc.message},
        )

    @app.exception_handler(ValidationError)
    async def handle_validation_error(_: Request, exc: ValidationError) -> JSONResponse:
        """Pydantic ValidationError → 422 응답."""
        logger.error("검증 오류", errors=exc.errors())
        error_msg = "입력 데이터 검증 실패"
        if exc.errors():
            first_error = exc.errors()[0]
            error_msg = f"{first_error.get('loc', [''])}: {first_error.get('msg', '검증 실패')}"
        return JSONResponse(
            status_code=422,
            content={"success": False, "message": error_msg},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        """알 수 없는 예외 → 500 응답."""
        logger.error("알 수 없는 오류", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": "서버 오류가 발생했습니다"},
        )

    # 라우터 등록
    app.include_router(health.router)
    app.include_router(video.router)
    app.include_router(config.router)
    app.include_router(metrics.router)
    app.include_router(admin_ws.router)
    app.include_router(admin_debug.router)
    app.include_router(dashboard.router)
    app.include_router(audio.router)

    # 정적 파일 서빙 (Clean Architecture Frontend)
    interface_dir = Path(__file__).resolve().parent.parent
    static_root = interface_dir / "static"

    if static_root.exists():
        # Common (Shared Kernel)
        common_dir = static_root / "common"
        if common_dir.exists():
            app.mount("/static/common", StaticFiles(directory=common_dir), name="static-common")

        # Audio Module
        audio_dir = static_root / "audio"
        if audio_dir.exists():
            app.mount("/static/audio", StaticFiles(directory=audio_dir), name="static-audio")

        # Video Module (Main Dashboard)
        video_dir = static_root / "video"
        if video_dir.exists():
            app.mount("/static/video", StaticFiles(directory=video_dir), name="static-video")
            
        # Root Redirect to Video Dashboard (Default)
        @app.get("/")
        async def root():
            return JSONResponse(status_code=307, headers={"Location": "/static/video/index.html"}, content=None)

    return app
