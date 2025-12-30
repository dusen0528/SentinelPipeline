"""
FastAPI dependency wiring.

Interface 계층에서 사용할 의존성을 관리합니다.
Composition Root(main.py)에서 set_app_context로 주입합니다.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sentinel_pipeline.application.config.manager import ConfigManager
from sentinel_pipeline.application.event.emitter import EventEmitter
from sentinel_pipeline.application.pipeline.pipeline import PipelineEngine
from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.application.stream.audio_manager import AudioManager
from sentinel_pipeline.interface.config.loader import ConfigLoader


@dataclass
class AppContext:
    stream_manager: StreamManager
    audio_manager: AudioManager
    pipeline_engine: PipelineEngine
    config_manager: ConfigManager
    config_loader: ConfigLoader
    event_emitter: EventEmitter
    transport_close_funcs: list[Callable[[], None]] = field(default_factory=list)


security = HTTPBasic()


def set_app_context(app: FastAPI, context: AppContext) -> None:
    """FastAPI app.state에 AppContext를 저장합니다"""
    app.state.app_context = context


def get_app_context(request: Request) -> AppContext:
    context: AppContext | None = getattr(request.app.state, "app_context", None)
    if context is None:
        raise RuntimeError("AppContext가 설정되지 않았습니다")
    return context


def get_stream_manager(context: AppContext = Depends(get_app_context)) -> StreamManager:
    return context.stream_manager


def get_audio_manager(context: AppContext = Depends(get_app_context)) -> AudioManager:
    return context.audio_manager


def get_pipeline_engine(context: AppContext = Depends(get_app_context)) -> PipelineEngine:
    return context.pipeline_engine


def get_config_manager(context: AppContext = Depends(get_app_context)) -> ConfigManager:
    return context.config_manager


def get_config_loader(context: AppContext = Depends(get_app_context)) -> ConfigLoader:
    return context.config_loader


def get_event_emitter(context: AppContext = Depends(get_app_context)) -> EventEmitter:
    return context.event_emitter


async def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """
    API Key 검증입니다.
    환경변수 API_KEY가 설정된 경우에만 검증합니다.
    """
    expected = os.getenv("API_KEY")
    if expected:
        if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )


async def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    """
    Admin 인증 (Basic Auth).
    환경변수 ADMIN_USER / ADMIN_PASSWORD 기준으로 검증합니다.
    """
    admin_user = os.getenv("ADMIN_USER", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD")

    if admin_password is None:
        # 설정되지 않으면 인증을 요구하지 않음
        return

    user_ok = secrets.compare_digest(credentials.username, admin_user)
    pass_ok = secrets.compare_digest(credentials.password, admin_password)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
