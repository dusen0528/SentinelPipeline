"""
플러그인 레지스트리 및 로더

현재는 더미/패스스루 모듈을 등록합니다.
실제 모델 연동 시 이 레지스트리에 구현체를 추가하세요.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Type

from sentinel_pipeline.domain.interfaces.module import ModuleBase
from sentinel_pipeline.domain.models.event import Event
from sentinel_pipeline.common.logging import get_logger

logger = get_logger(__name__)

class BasePlugin:
    """기본 플러그인 베이스 (ModuleBase 구현)."""

    name: str = "BasePlugin"
    enabled: bool = True
    priority: int = 100
    timeout_ms: int = 50
    options: dict[str, Any]

    def __init__(self, **options: Any) -> None:
        self.options = options

    @staticmethod
    def validate_options(options: dict[str, Any]) -> None:
        """옵션 검증 기본 구현 (확장 클래스에서 override)."""
        return None

    def process_frame(self, frame, metadata) -> tuple[Any, list[Event], dict]:
        # 패스스루: 프레임 그대로, 이벤트 없음, metadata 그대로 반환
        return frame, [], metadata

    def process_audio(self, chunk, metadata) -> tuple[list[Event], dict]:
        # 오디오 미지원 기본 구현 (metadata 그대로 반환)
        return [], metadata


class FireDetectModule(BasePlugin):
    name = "FireDetectModule"
    priority = 100
    timeout_ms = 45

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.6,
        min_consecutive: int = 3,
        window_ms: int = 2000,
        **kw: Any,
    ) -> None:
        super().__init__(model_path=model_path, threshold=threshold, min_consecutive=min_consecutive, window_ms=window_ms, **kw)
        self.model_path = model_path
        self.threshold = float(threshold)
        self.min_consecutive = int(min_consecutive)
        self.window_ms = int(window_ms)
        # 간단한 상태 버퍼 (실제 구현에서 히스토리/윈도우 로직으로 대체)
        self._last_score: float = 0.0
        self._last_ts: float = 0.0

    @staticmethod
    def validate_options(options: dict[str, Any]) -> None:
        model_path = options.get("model_path")
        if not model_path or not isinstance(model_path, str):
            raise ValueError("FireDetectModule.model_path는 필수 문자열입니다")
        if not os.path.exists(model_path):
            raise ValueError(f"FireDetectModule.model_path 파일을 찾을 수 없습니다: {model_path}")

        threshold = options.get("threshold", 0.6)
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("FireDetectModule.threshold는 0~1 범위의 숫자여야 합니다")

        min_consecutive = options.get("min_consecutive", 3)
        if not isinstance(min_consecutive, int) or min_consecutive < 1:
            raise ValueError("FireDetectModule.min_consecutive는 1 이상의 정수여야 합니다")

        window_ms = options.get("window_ms", 2000)
        if not isinstance(window_ms, int) or window_ms < 1:
            raise ValueError("FireDetectModule.window_ms는 1 이상의 정수여야 합니다")

    def process_frame(self, frame, metadata):
        # TODO: 실제 모델 추론/윈도우 로직으로 대체
        # 여기서는 dummy: 항상 score=0.0 반환
        self._last_score = 0.0
        metadata_out = dict(metadata)
        metadata_out["fire_score"] = self._last_score
        metadata_out["fire_detected"] = False
        return frame, [], metadata_out


class ScreamDetectModule(BasePlugin):
    name = "ScreamDetectModule"
    priority = 120
    timeout_ms = 40

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.75,
        cooldown_ms: int = 5000,
        **kw: Any,
    ) -> None:
        super().__init__(model_path=model_path, threshold=threshold, cooldown_ms=cooldown_ms, **kw)
        self.model_path = model_path
        self.threshold = float(threshold)
        self.cooldown_ms = int(cooldown_ms)
        self._last_alert_ts: float = 0.0

    @staticmethod
    def validate_options(options: dict[str, Any]) -> None:
        model_path = options.get("model_path")
        if not model_path or not isinstance(model_path, str):
            raise ValueError("ScreamDetectModule.model_path는 필수 문자열입니다")
        if not os.path.exists(model_path):
            raise ValueError(f"ScreamDetectModule.model_path 파일을 찾을 수 없습니다: {model_path}")

        threshold = options.get("threshold", 0.75)
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError("ScreamDetectModule.threshold는 0~1 범위의 숫자여야 합니다")

        cooldown_ms = options.get("cooldown_ms", 5000)
        if not isinstance(cooldown_ms, int) or cooldown_ms < 0:
            raise ValueError("ScreamDetectModule.cooldown_ms는 0 이상 정수여야 합니다")

    def process_audio(self, chunk, metadata):
        # TODO: 실제 오디오 추론/멜스펙 처리로 대체
        metadata_out = dict(metadata)
        metadata_out["scream_score"] = 0.0
        metadata_out["cooldown_remaining"] = max(0, self.cooldown_ms - 0)
        return [], metadata_out


class FaceBlurModule(BasePlugin):
    name = "FaceBlurModule"
    priority = 200
    timeout_ms = 50

    def process_frame(self, frame, metadata):
        # 실제 블러링은 추후 구현, 현재는 통과
        return frame, [], metadata

    def __init__(
        self,
        model_path: str,
        blur_kernel: int = 99,
        confidence_threshold: float = 0.5,
        min_face_size: int = 30,
        use_tracker: bool = True,
        tracker_max_age: int = 25,
        tracker_smoothing: float = 0.5,
        **kw: Any,
    ) -> None:
        super().__init__(
            model_path=model_path,
            blur_kernel=blur_kernel,
            confidence_threshold=confidence_threshold,
            min_face_size=min_face_size,
            use_tracker=use_tracker,
            tracker_max_age=tracker_max_age,
            tracker_smoothing=tracker_smoothing,
            **kw,
        )
        self.model_path = model_path
        self.blur_kernel = int(blur_kernel)
        self.confidence_threshold = float(confidence_threshold)
        self.min_face_size = int(min_face_size)
        self.use_tracker = bool(use_tracker)
        self.tracker_max_age = int(tracker_max_age)
        self.tracker_smoothing = float(tracker_smoothing)

    @staticmethod
    def validate_options(options: dict[str, Any]) -> None:
        model_path = options.get("model_path")
        if not model_path or not isinstance(model_path, str):
            raise ValueError("FaceBlurModule.model_path는 필수 문자열입니다")
        if not os.path.exists(model_path):
            raise ValueError(f"FaceBlurModule.model_path 파일을 찾을 수 없습니다: {model_path}")

        kernel = options.get("blur_kernel", 99)
        if not isinstance(kernel, int) or kernel < 1:
            raise ValueError("FaceBlurModule.blur_kernel은 1 이상의 정수여야 합니다")

        conf = options.get("confidence_threshold", 0.5)
        if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
            raise ValueError("FaceBlurModule.confidence_threshold는 0~1 범위의 숫자여야 합니다")

        min_face = options.get("min_face_size", 30)
        if not isinstance(min_face, int) or min_face < 1:
            raise ValueError("FaceBlurModule.min_face_size는 1 이상의 정수여야 합니다")

        tracker_max_age = options.get("tracker_max_age", 25)
        if not isinstance(tracker_max_age, int) or tracker_max_age < 0:
            raise ValueError("FaceBlurModule.tracker_max_age는 0 이상 정수여야 합니다")

        tracker_smoothing = options.get("tracker_smoothing", 0.5)
        if not isinstance(tracker_smoothing, (int, float)) or not (0 <= tracker_smoothing <= 1):
            raise ValueError("FaceBlurModule.tracker_smoothing은 0~1 범위의 숫자여야 합니다")


PLUGIN_REGISTRY: Dict[str, Type[ModuleBase]] = {
    FireDetectModule.name: FireDetectModule,
    ScreamDetectModule.name: ScreamDetectModule,
    FaceBlurModule.name: FaceBlurModule,
}


def load_plugins_from_config(
    modules: list[dict[str, Any]],
    *,
    strict_missing: bool = False,
) -> List[ModuleBase]:
    """
    Config modules 항목을 기반으로 플러그인 인스턴스를 생성합니다.

    Args:
        modules: 모듈 설정 딕셔너리 리스트
        strict_missing: 레지스트리에 없는 모듈이 있을 때 예외 발생 여부 (기본 False)
    """
    loaded: List[ModuleBase] = []
    for m in modules:
        name = m.get("name")
        if not name:
            continue
        cls = PLUGIN_REGISTRY.get(name)
        if not cls:
            msg = f"등록되지 않은 플러그인: {name}"
            if strict_missing:
                raise ValueError(msg)
            logger.warning("등록되지 않은 플러그인 스킵", module=name)
            continue

        if m.get("enabled") is False:
            logger.info("비활성 모듈 스킵", module=name)
            continue
        try:
            options = m.get("options", {}) or {}
            # 옵션 검증 (필수/타입)
            cls.validate_options(options)

            instance = cls(**m.get("options", {}))
            instance.enabled = m.get("enabled", True)
            instance.priority = m.get("priority", getattr(instance, "priority", 100))
            instance.timeout_ms = m.get("timeout_ms", getattr(instance, "timeout_ms", 50))
            loaded.append(instance)
        except Exception as e:
            logger.warning("플러그인 로드 실패, 스킵", module=name, error=str(e))
            continue
    return loaded

