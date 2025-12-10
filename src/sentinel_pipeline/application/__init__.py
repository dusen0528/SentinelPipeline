"""
Application Layer

비즈니스 로직과 유스케이스를 구현합니다.
Domain Layer만 참조하며, Infrastructure와 Interface Layer에 의존하지 않습니다.

구성 요소:
- pipeline: 파이프라인 엔진, 모듈 스케줄링
- stream: 스트림 관리, 헬스 체크
- config: 설정 동적 변경 관리
- event: 이벤트 발행
"""

from sentinel_pipeline.application.pipeline.pipeline import PipelineEngine
from sentinel_pipeline.application.pipeline.scheduler import ModuleScheduler
from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.application.stream.health import HealthWatcher
from sentinel_pipeline.application.config.manager import ConfigManager
from sentinel_pipeline.application.event.emitter import EventEmitter

__all__ = [
    "PipelineEngine",
    "ModuleScheduler",
    "StreamManager",
    "HealthWatcher",
    "ConfigManager",
    "EventEmitter",
]

