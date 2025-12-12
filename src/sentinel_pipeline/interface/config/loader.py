"""
설정 로더

config.json을 로드하고 Pydantic 스키마로 검증합니다.
검증된 설정을 런타임(AppConfig, StreamConfig) 구조로 변환합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sentinel_pipeline.application.config.manager import (
    AppConfig as RuntimeAppConfig,
    GlobalConfig as RuntimeGlobalConfig,
    ModuleConfig as RuntimeModuleConfig,
)
from sentinel_pipeline.common.errors import ConfigError, ErrorCode
from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.domain.models.stream import StreamConfig as DomainStreamConfig

from .schema import AppConfig, GlobalConfig

logger = get_logger(__name__)


class ConfigLoader:
    """config.json 로딩 및 변환을 담당합니다."""

    def __init__(self, default_path: str = "config.json") -> None:
        self._default_path = Path(default_path)

    def load_from_file(self, path: str | Path | None = None) -> AppConfig:
        """파일에서 설정을 로드하고 검증합니다."""
        target = Path(path) if path else self._default_path

        if not target.exists():
            raise ConfigError(
                ErrorCode.CONFIG_NOT_FOUND,
                f"설정 파일을 찾을 수 없습니다: {target}",
                config_path=str(target),
            )

        try:
            content = target.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ConfigError(
                ErrorCode.CONFIG_PARSE_ERROR,
                f"설정 파일 파싱에 실패했습니다: {e}",
                config_path=str(target),
                details={"error": str(e)},
            ) from e

        return self.load_from_dict(data, config_path=str(target))

    def load_from_dict(
        self,
        data: dict[str, Any],
        config_path: str | None = None,
    ) -> AppConfig:
        """딕셔너리에서 설정을 검증합니다."""
        try:
            config = AppConfig.model_validate(data)
            return config
        except ValidationError as e:
            logger.error("설정 검증 실패", errors=e.errors(), config_path=config_path)
            raise ConfigError(
                ErrorCode.CONFIG_INVALID,
                "설정 검증에 실패했습니다",
                config_path=config_path,
                details={"errors": e.errors()},
            ) from e

    def merge_with_defaults(self, config: AppConfig) -> AppConfig:
        """기본값이 적용된 설정을 반환합니다."""
        return config.model_copy()

    def validate(self, config: AppConfig) -> tuple[bool, list[str]]:
        """
        추가 교차 검증.
        - 이벤트 배치 크기 <= 큐 크기
        """
        errors: list[str] = []

        if config.event.batch_size > config.event.max_queue_size:
            errors.append("event.batch_size는 event.max_queue_size 이하여야 합니다")

        return (len(errors) == 0), errors

    def to_runtime(self, config: AppConfig) -> RuntimeAppConfig:
        """
        Application Layer에서 사용하는 런타임 설정으로 변환합니다.
        (modules, global_config만 포함 - 나머지는 별도 반환)
        """
        runtime_modules = [
            RuntimeModuleConfig(
                name=module.name,
                enabled=module.enabled,
                priority=module.priority,
                timeout_ms=module.timeout_ms,
                options=dict(module.options),
            )
            for module in config.modules
        ]

        global_cfg: GlobalConfig = config.global_config
        runtime_global = RuntimeGlobalConfig(
            max_fps=global_cfg.max_fps,
            downscale=global_cfg.downscale,
            queue_max=global_cfg.queue_max,
            drop_strategy=global_cfg.drop_strategy,
            inference_timeout_ms=global_cfg.inference_timeout_ms,
        )

        return RuntimeAppConfig(
            modules=runtime_modules,
            global_config=runtime_global,
        )

    def to_runtime_bundle(self, config: AppConfig) -> dict[str, Any]:
        """
        파이프라인/이벤트/관측 설정을 포함한 번들을 반환합니다.
        EntryPoint에서 각 컴포넌트 초기화에 사용합니다.
        """
        return {
            "app": self.to_runtime(config),
            "pipeline": config.pipeline.model_dump(),
            "event": config.event.model_dump(),
            "observability": config.observability.model_dump(),
            "streams": self.to_domain_streams(config),
        }

    def to_domain_streams(self, config: AppConfig) -> list[DomainStreamConfig]:
        """StreamConfig(Pydantic)를 Domain StreamConfig로 변환합니다."""
        streams: list[DomainStreamConfig] = []
        for stream in config.streams:
            streams.append(
                DomainStreamConfig(
                    stream_id=stream.stream_id,
                    rtsp_url=stream.rtsp_url,
                    enabled=stream.enabled,
                    max_fps=stream.max_fps,
                    downscale=stream.downscale,
                    buffer_size=stream.buffer_size,
                    reconnect_enabled=stream.reconnect_enabled,
                    reconnect_max_retries=stream.reconnect_max_retries,
                    reconnect_base_delay=stream.reconnect_base_delay,
                    reconnect_max_delay=stream.reconnect_max_delay,
                    output_url=stream.output_url,
                    output_enabled=stream.output_enabled,
                )
            )
        return streams


