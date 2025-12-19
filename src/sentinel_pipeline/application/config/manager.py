"""
설정 관리자

런타임 설정 변경과 컴포넌트 간 전파를 담당합니다.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Callable
from dataclasses import dataclass, field, asdict

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.common.errors import ConfigError, ErrorCode

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class ModuleConfig:
    """모듈 설정 (런타임용)"""
    name: str
    enabled: bool = True
    priority: int = 100
    timeout_ms: int = 50
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalConfig:
    """전역 설정 (런타임용)"""
    max_fps: int = 15
    downscale: float = 0.5
    queue_max: int = 2
    drop_strategy: str = "drop_oldest"
    inference_timeout_ms: int = 50
    target_width: int | None = None
    target_height: int | None = None


@dataclass
class AppConfig:
    """애플리케이션 설정 (런타임용)"""
    modules: list[ModuleConfig] = field(default_factory=list)
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    # 전체 설정 bundle (pipeline/event/observability/streams 포함)
    _bundle: dict[str, Any] | None = None


class ConfigManager:
    """
    설정 관리자
    
    애플리케이션 설정을 관리하고, 변경 시 관련 컴포넌트에 전파합니다.
    
    주요 기능:
    - 설정 조회/변경
    - 모듈 설정 개별 변경
    - 변경 콜백 등록
    - 스레드 안전 보장
    
    Example:
        >>> manager = ConfigManager()
        >>> manager.load_config(app_config)
        >>> 
        >>> # 설정 변경
        >>> manager.update_module("FireDetect", enabled=False)
        >>> 
        >>> # 콜백 등록
        >>> manager.add_on_change(lambda: print("설정 변경됨"))
    """
    
    def __init__(self) -> None:
        """설정 관리자 초기화"""
        self._config: AppConfig | None = None
        self._bundle: dict[str, Any] | None = None  # 전체 설정 bundle
        self._lock = threading.RLock()
        
        # 변경 콜백 목록
        self._on_change_callbacks: list[Callable[[], None]] = []
        
        # 모듈별 변경 콜백
        self._on_module_change_callbacks: list[Callable[[str, ModuleConfig], None]] = []
        
        # 전역 설정 변경 콜백
        self._on_global_change_callbacks: list[Callable[[GlobalConfig], None]] = []
    
    def load_config(self, config: AppConfig, bundle: dict[str, Any] | None = None) -> None:
        """
        설정을 로드합니다.
        
        Args:
            config: 애플리케이션 설정
            bundle: 전체 설정 bundle (선택)
        """
        with self._lock:
            self._config = config
            self._bundle = bundle
            if bundle is not None:
                config._bundle = bundle
            logger.info(
                f"설정 로드 완료: 모듈 {len(config.modules)}개",
                module_count=len(config.modules),
            )
    
    def get_config(self) -> AppConfig | None:
        """현재 설정을 반환합니다."""
        with self._lock:
            return self._config
    
    def get_module_config(self, name: str) -> ModuleConfig | None:
        """
        특정 모듈의 설정을 반환합니다.
        
        Args:
            name: 모듈 이름
        
        Returns:
            모듈 설정 또는 None
        """
        with self._lock:
            if not self._config:
                return None
            
            for module in self._config.modules:
                if module.name == name:
                    return module
            
            return None
    
    def get_global_config(self) -> GlobalConfig | None:
        """전역 설정을 반환합니다."""
        with self._lock:
            return self._config.global_config if self._config else None
    
    def update_config(self, new_config: AppConfig, bundle: dict[str, Any] | None = None) -> None:
        """
        전체 설정을 업데이트합니다.
        
        Args:
            new_config: 새 설정
            bundle: 전체 설정 bundle (선택)
        """
        with self._lock:
            self._config = new_config
            if bundle is not None:
                self._bundle = bundle
                new_config._bundle = bundle
            
            logger.info("설정 전체 업데이트")
        
        # 콜백 호출 (락 밖에서)
        self._notify_change()
    
    def update_module(
        self,
        name: str,
        enabled: bool | None = None,
        priority: int | None = None,
        timeout_ms: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ModuleConfig:
        """
        특정 모듈의 설정을 업데이트합니다.
        
        Args:
            name: 모듈 이름
            enabled: 활성화 여부 (None이면 변경 안함)
            priority: 우선순위 (None이면 변경 안함)
            timeout_ms: 타임아웃 (None이면 변경 안함)
            options: 옵션 (None이면 변경 안함, 병합됨)
        
        Returns:
            업데이트된 모듈 설정
        
        Raises:
            ConfigError: 모듈을 찾을 수 없을 때
        """
        with self._lock:
            if not self._config:
                raise ConfigError(
                    ErrorCode.CONFIG_NOT_FOUND,
                    "설정이 로드되지 않았습니다",
                )
            
            module_config = None
            for module in self._config.modules:
                if module.name == name:
                    module_config = module
                    break
            
            if not module_config:
                raise ConfigError(
                    ErrorCode.CONFIG_INVALID,
                    f"모듈을 찾을 수 없습니다: {name}",
                    field_name=name,
                )
            
            # 설정 업데이트
            if enabled is not None:
                module_config.enabled = enabled
            if priority is not None:
                module_config.priority = priority
            if timeout_ms is not None:
                module_config.timeout_ms = timeout_ms
            if options is not None:
                module_config.options.update(options)
            
            logger.info(
                f"모듈 설정 변경: {name}",
                module_name=name,
                enabled=module_config.enabled,
                priority=module_config.priority,
            )
        
        # 콜백 호출 (락 밖에서)
        self._notify_module_change(name, module_config)
        
        return module_config
    
    def update_global(
        self,
        max_fps: int | None = None,
        downscale: float | None = None,
        queue_max: int | None = None,
        drop_strategy: str | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> GlobalConfig:
        """
        전역 설정을 업데이트합니다.
        
        Args:
            max_fps: 최대 FPS
            downscale: 다운스케일 비율
            queue_max: 최대 큐 크기
            drop_strategy: 드롭 전략
            target_width: 목표 출력 너비
            target_height: 목표 출력 높이
        
        Returns:
            업데이트된 전역 설정
        """
        with self._lock:
            if not self._config:
                raise ConfigError(
                    ErrorCode.CONFIG_NOT_FOUND,
                    "설정이 로드되지 않았습니다",
                )
            
            global_config = self._config.global_config
            
            if max_fps is not None:
                global_config.max_fps = max_fps
            if downscale is not None:
                global_config.downscale = downscale
            if queue_max is not None:
                global_config.queue_max = queue_max
            if drop_strategy is not None:
                global_config.drop_strategy = drop_strategy
            if target_width is not None:
                global_config.target_width = target_width
            if target_height is not None:
                global_config.target_height = target_height
            
            logger.info(
                f"전역 설정 변경: max_fps={global_config.max_fps}, "
                f"downscale={global_config.downscale}, "
                f"target_resolution={global_config.target_width}x{global_config.target_height}"
            )
        
        # 콜백 호출 (락 밖에서)
        self._notify_global_change(global_config)
        
        return global_config
    
    def enable_module(self, name: str) -> bool:
        """모듈을 활성화합니다."""
        try:
            self.update_module(name, enabled=True)
            return True
        except ConfigError:
            return False
    
    def disable_module(self, name: str) -> bool:
        """모듈을 비활성화합니다."""
        try:
            self.update_module(name, enabled=False)
            return True
        except ConfigError:
            return False
    
    def add_on_change(self, callback: Callable[[], None]) -> None:
        """설정 변경 콜백을 추가합니다."""
        self._on_change_callbacks.append(callback)
    
    def add_on_module_change(
        self,
        callback: Callable[[str, ModuleConfig], None],
    ) -> None:
        """모듈 설정 변경 콜백을 추가합니다."""
        self._on_module_change_callbacks.append(callback)
    
    def add_on_global_change(
        self,
        callback: Callable[[GlobalConfig], None],
    ) -> None:
        """전역 설정 변경 콜백을 추가합니다."""
        self._on_global_change_callbacks.append(callback)
    
    def _notify_change(self) -> None:
        """설정 변경을 알립니다."""
        for callback in self._on_change_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"설정 변경 콜백 오류: {e}")
    
    def _notify_module_change(self, name: str, config: ModuleConfig) -> None:
        """모듈 설정 변경을 알립니다."""
        for callback in self._on_module_change_callbacks:
            try:
                callback(name, config)
            except Exception as e:
                logger.error(f"모듈 설정 변경 콜백 오류: {e}")
    
    def _notify_global_change(self, config: GlobalConfig) -> None:
        """전역 설정 변경을 알립니다."""
        for callback in self._on_global_change_callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(f"전역 설정 변경 콜백 오류: {e}")
    
    def to_dict(self) -> dict[str, Any]:
        """설정을 딕셔너리로 반환합니다."""
        with self._lock:
            if not self._config:
                return {}
            
            return {
                "modules": [
                    {
                        "name": m.name,
                        "enabled": m.enabled,
                        "priority": m.priority,
                        "timeout_ms": m.timeout_ms,
                        "options": m.options,
                    }
                    for m in self._config.modules
                ],
                "global": {
                    "max_fps": self._config.global_config.max_fps,
                    "downscale": self._config.global_config.downscale,
                    "queue_max": self._config.global_config.queue_max,
                    "drop_strategy": self._config.global_config.drop_strategy,
                    "inference_timeout_ms": self._config.global_config.inference_timeout_ms,
                },
            }

    def get_bundle(self) -> dict[str, Any] | None:
        """전체 설정 bundle을 반환합니다."""
        with self._lock:
            return self._bundle

    def to_full_dict(self) -> dict[str, Any]:
        """전체 설정을 딕셔너리로 반환합니다 (bundle 포함)."""
        with self._lock:
            if not self._config:
                return {}

            # App (dataclass) → dict
            app_dict = asdict(self._config)
            app_dict.pop("_bundle", None)  # 내부 필드 제거

            result: dict[str, Any] = {"app": app_dict}

            if self._bundle:
                bundle = self._bundle.copy()

                # app (RuntimeAppConfig dataclass) → dict
                app_bundle = bundle.get("app")
                if app_bundle is not None:
                    try:
                        bundle["app"] = asdict(app_bundle)
                    except TypeError:
                        pass

                # streams (Domain StreamConfig list) → dict list
                streams_bundle = bundle.get("streams")
                if streams_bundle is not None:
                    stream_dicts: list[dict[str, Any]] = []
                    for s in streams_bundle:
                        if hasattr(s, "to_dict"):
                            stream_dicts.append(s.to_dict())
                        else:
                            try:
                                stream_dicts.append(asdict(s))
                            except TypeError:
                                stream_dicts.append({"stream": str(s)})
                    bundle["streams"] = stream_dicts

                result["bundle"] = bundle
            return result

