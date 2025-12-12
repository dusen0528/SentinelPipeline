"""
설정 조회/변경 API
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from sentinel_pipeline.application.config.manager import ConfigManager
from sentinel_pipeline.common.errors import ConfigError, ErrorCode
from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.interface.api.dependencies import (
    get_config_loader,
    get_config_manager,
    verify_admin,
)
from sentinel_pipeline.interface.config.loader import ConfigLoader
from sentinel_pipeline.interface.config.schema import AppConfig

router = APIRouter(prefix="/api/config", tags=["config"])
logger = get_logger(__name__)


# === DTO 정의 ===


class ConfigResponse(BaseModel):
    success: bool = True
    data: dict[str, Any]


class ModuleConfigResponse(BaseModel):
    success: bool = True
    data: dict[str, Any]


class UpdateConfigRequest(AppConfig):
    """전체 설정 교체 요청 (AppConfig 스키마 재사용)."""

    model_config = {"extra": "forbid"}


class UpdateModuleConfigRequest(BaseModel):
    enabled: bool | None = Field(None, description="활성화 여부")
    priority: int | None = Field(None, description="우선순위")
    timeout_ms: int | None = Field(None, description="타임아웃(ms)")
    options: dict[str, Any] | None = Field(None, description="옵션 병합")

    model_config = {"extra": "forbid"}


# === 라우트 ===


@router.get("", response_model=ConfigResponse)
async def get_config(
    manager: ConfigManager = Depends(get_config_manager),
) -> ConfigResponse:
    """현재 설정을 조회합니다 (전체 bundle 포함)."""
    return ConfigResponse(data=manager.to_full_dict())


@router.put("", response_model=ConfigResponse, dependencies=[Depends(verify_admin)])
async def update_config(
    payload: UpdateConfigRequest,
    manager: ConfigManager = Depends(get_config_manager),
    loader: ConfigLoader = Depends(get_config_loader),
) -> ConfigResponse:
    """
    전체 설정을 교체합니다.
    - Pydantic 스키마 검증
    - Runtime AppConfig로 변환 후 ConfigManager에 적용
    """
    config = loader.merge_with_defaults(payload)
    ok, errors = loader.validate(config)
    if not ok:
        raise ConfigError(
            ErrorCode.CONFIG_INVALID,
            "설정 검증에 실패했습니다",
            details={"errors": errors},
        )
    runtime_config = loader.to_runtime(config)
    bundle = loader.to_runtime_bundle(config)
    manager.load_config(runtime_config, bundle=bundle)

    # 런타임 컴포넌트에 재적용
    from sentinel_pipeline.interface.api.dependencies import (
        get_event_emitter,
        get_pipeline_engine,
        get_stream_manager,
    )
    from sentinel_pipeline.main import apply_bundle_to_components

    try:
        pipeline_engine = get_pipeline_engine()
        event_emitter = get_event_emitter()
        stream_manager = get_stream_manager()

        apply_bundle_to_components(
            bundle=bundle,
            pipeline_engine=pipeline_engine,
            event_emitter=event_emitter,
            stream_manager=stream_manager,
        )
    except Exception as e:
        # 재적용 실패해도 설정 저장은 유지, 경고만 남김
        logger.error(f"설정 재적용 실패: {e}", error=str(e))

    # 전체 설정 반환 (bundle 포함)
    return ConfigResponse(data=manager.to_full_dict())


@router.get("/modules/{name}", response_model=ModuleConfigResponse)
async def get_module_config(
    name: str,
    manager: ConfigManager = Depends(get_config_manager),
) -> ModuleConfigResponse:
    """특정 모듈 설정을 조회합니다."""
    module = manager.get_module_config(name)
    if not module:
        raise ConfigError(
            ErrorCode.CONFIG_INVALID,
            f"모듈을 찾을 수 없습니다: {name}",
            field_name=name,
        )
    return ModuleConfigResponse(
        data={
            "name": module.name,
            "enabled": module.enabled,
            "priority": module.priority,
            "timeout_ms": module.timeout_ms,
            "options": module.options,
        }
    )


@router.patch("/modules/{name}", response_model=ModuleConfigResponse, dependencies=[Depends(verify_admin)])
async def update_module_config(
    name: str,
    payload: UpdateModuleConfigRequest,
    manager: ConfigManager = Depends(get_config_manager),
) -> ModuleConfigResponse:
    """특정 모듈 설정을 부분 업데이트합니다."""
    updated = manager.update_module(
        name=name,
        enabled=payload.enabled,
        priority=payload.priority,
        timeout_ms=payload.timeout_ms,
        options=payload.options,
    )
    return ModuleConfigResponse(
        data={
            "name": updated.name,
            "enabled": updated.enabled,
            "priority": updated.priority,
            "timeout_ms": updated.timeout_ms,
            "options": updated.options,
        }
    )


@router.post("/modules/{name}/enable", response_model=ModuleConfigResponse, dependencies=[Depends(verify_admin)])
async def enable_module(
    name: str,
    manager: ConfigManager = Depends(get_config_manager),
) -> ModuleConfigResponse:
    """모듈을 활성화합니다."""
    updated = manager.update_module(name=name, enabled=True)
    return ModuleConfigResponse(
        data={
            "name": updated.name,
            "enabled": updated.enabled,
            "priority": updated.priority,
            "timeout_ms": updated.timeout_ms,
            "options": updated.options,
        }
    )


@router.post("/modules/{name}/disable", response_model=ModuleConfigResponse, dependencies=[Depends(verify_admin)])
async def disable_module(
    name: str,
    manager: ConfigManager = Depends(get_config_manager),
) -> ModuleConfigResponse:
    """모듈을 비활성화합니다."""
    updated = manager.update_module(name=name, enabled=False)
    return ModuleConfigResponse(
        data={
            "name": updated.name,
            "enabled": updated.enabled,
            "priority": updated.priority,
            "timeout_ms": updated.timeout_ms,
            "options": updated.options,
        }
    )

