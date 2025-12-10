"""
Domain Layer

순수 비즈니스 규칙과 엔티티를 정의합니다.
외부 라이브러리에 의존하지 않으며, 표준 라이브러리만 사용합니다.

구성 요소:
- interfaces: 모듈 인터페이스 (Protocol)
- models: 데이터 모델 (Event, Stream)
"""

from sentinel_pipeline.domain.interfaces.module import ModuleBase
from sentinel_pipeline.domain.models.event import Event, EventType, EventStage
from sentinel_pipeline.domain.models.stream import (
    StreamConfig,
    StreamState,
    StreamStatus,
)

__all__ = [
    # 인터페이스
    "ModuleBase",
    # 이벤트 모델
    "Event",
    "EventType",
    "EventStage",
    # 스트림 모델
    "StreamConfig",
    "StreamState",
    "StreamStatus",
]

