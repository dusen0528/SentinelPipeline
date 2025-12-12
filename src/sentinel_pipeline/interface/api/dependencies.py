"""
FastAPI 의존성 주입 헬퍼

Interface 계층에서만 외부 라이브러리(FastAPI, Pydantic)에 의존합니다.
실제 인스턴스는 Composition Root(main.py)에서 set_app_context로 주입합니다.
"""

from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sentinel_pipeline.application.config.manager import ConfigManager
from sentinel_pipeline.application.event.emitter import EventEmitter
from sentinel_pipeline.application.pipeline.pipeline import PipelineEngine
from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.interface.config.loader import ConfigLoader

_stream_manager: Optional[StreamManager] = None
_pipeline_engine: Optional[PipelineEngine] = None
_config_manager: Optional[ConfigManager] = None
_config_loader: Optional[ConfigLoader] = None
_event_emitter: Optional[EventEmitter] = None

security = HTTPBasic()


def set_app_context(
    stream_manager: StreamManager,
    pipeline_engine: PipelineEngine,
    config_manager: ConfigManager,
    config_loader: ConfigLoader,
    event_emitter: EventEmitter,
) -> None:
    """애플리케이션 의존성을 주입합니다."""
    global _stream_manager, _pipeline_engine, _config_manager, _config_loader, _event_emitter
    _stream_manager = stream_manager
    _pipeline_engine = pipeline_engine
    _config_manager = config_manager
    _config_loader = config_loader
    _event_emitter = event_emitter


@lru_cache(maxsize=1)
def get_stream_manager() -> StreamManager:
    if _stream_manager is None:
        raise RuntimeError("StreamManager가 설정되지 않았습니다")
    return _stream_manager


@lru_cache(maxsize=1)
def get_pipeline_engine() -> PipelineEngine:
    if _pipeline_engine is None:
        raise RuntimeError("PipelineEngine이 설정되지 않았습니다")
    return _pipeline_engine


@lru_cache(maxsize=1)
def get_config_manager() -> ConfigManager:
    if _config_manager is None:
        raise RuntimeError("ConfigManager가 설정되지 않았습니다")
    return _config_manager


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    if _config_loader is None:
        raise RuntimeError("ConfigLoader가 설정되지 않았습니다")
    return _config_loader


@lru_cache(maxsize=1)
def get_event_emitter() -> EventEmitter:
    if _event_emitter is None:
        raise RuntimeError("EventEmitter가 설정되지 않았습니다")
    return _event_emitter


async def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """
    API Key 검증 (선택적).
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
        # 설정되지 않았다면 인증을 요구하지 않음
        return

    user_ok = secrets.compare_digest(credentials.username, admin_user)
    pass_ok = secrets.compare_digest(credentials.password, admin_password)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )

