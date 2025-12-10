"""
이벤트 모델

파이프라인 모듈이 생성하는 감지 이벤트의 데이터 구조를 정의합니다.
이 모듈은 외부 라이브러리에 의존하지 않습니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """
    이벤트 유형
    
    감지 모듈이 생성하는 이벤트의 종류를 정의합니다.
    """
    
    # 화재 관련
    FIRE = "FIRE"                       # 화재 감지
    SMOKE = "SMOKE"                     # 연기 감지
    
    # 침입 관련
    INTRUSION = "INTRUSION"             # 침입 감지
    LOITERING = "LOITERING"             # 배회 감지
    PERIMETER = "PERIMETER"             # 경계선 침범
    
    # 음향 관련
    SCREAM = "SCREAM"                   # 비명 감지
    GUNSHOT = "GUNSHOT"                 # 총성 감지
    GLASS_BREAK = "GLASS_BREAK"         # 유리 깨짐 감지
    
    # 객체 관련
    FACE_DETECTED = "FACE_DETECTED"     # 얼굴 감지
    PERSON_DETECTED = "PERSON_DETECTED" # 사람 감지
    VEHICLE_DETECTED = "VEHICLE_DETECTED"  # 차량 감지
    OBJECT_LEFT = "OBJECT_LEFT"         # 유기물 감지
    OBJECT_REMOVED = "OBJECT_REMOVED"   # 도난 감지
    
    # 행동 관련
    FALL_DETECTED = "FALL_DETECTED"     # 쓰러짐 감지
    FIGHT_DETECTED = "FIGHT_DETECTED"   # 싸움 감지
    CROWDING = "CROWDING"               # 군중 밀집
    
    # 시스템 관련
    SYSTEM_ALERT = "SYSTEM_ALERT"       # 시스템 알림
    CUSTOM = "CUSTOM"                   # 사용자 정의


class EventStage(str, Enum):
    """
    이벤트 단계
    
    이벤트의 확정 수준을 나타냅니다.
    """
    
    DETECTED = "DETECTED"       # 최초 감지 (단일 프레임)
    CONFIRMED = "CONFIRMED"     # 확정 (연속 감지 조건 충족)
    CLEARED = "CLEARED"         # 해제 (이벤트 종료)


@dataclass
class BoundingBox:
    """
    바운딩 박스
    
    감지된 객체의 위치를 나타냅니다.
    
    Attributes:
        x: 좌상단 X 좌표 (픽셀)
        y: 좌상단 Y 좌표 (픽셀)
        width: 너비 (픽셀)
        height: 높이 (픽셀)
    """
    
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        """우하단 X 좌표"""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """우하단 Y 좌표"""
        return self.y + self.height
    
    @property
    def center(self) -> tuple[int, int]:
        """중심점 좌표"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """면적"""
        return self.width * self.height
    
    def to_list(self) -> list[int]:
        """[x, y, width, height] 형식으로 반환"""
        return [self.x, self.y, self.width, self.height]
    
    def to_xyxy(self) -> list[int]:
        """[x1, y1, x2, y2] 형식으로 반환"""
        return [self.x, self.y, self.x2, self.y2]
    
    @classmethod
    def from_xyxy(cls, x1: int, y1: int, x2: int, y2: int) -> "BoundingBox":
        """[x1, y1, x2, y2] 형식에서 생성"""
        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
    
    def to_dict(self) -> dict[str, int]:
        """딕셔너리로 변환"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Event:
    """
    감지 이벤트
    
    파이프라인 모듈이 생성하는 이벤트 데이터 구조입니다.
    VMS로 전송되어 알람, 녹화, 정책 적용 등에 사용됩니다.
    
    Attributes:
        type: 이벤트 유형 (FIRE, SCREAM 등)
        stage: 이벤트 단계 (DETECTED, CONFIRMED, CLEARED)
        confidence: 신뢰도 (0.0 ~ 1.0)
        ts: 타임스탬프 (Unix timestamp, 초 단위)
        stream_id: 이벤트가 발생한 스트림 ID
        module_name: 이벤트를 생성한 모듈 이름
        latency_ms: 이벤트 생성까지의 처리 시간 (밀리초)
        details: 추가 상세 정보 (bbox, tracking_id 등)
        event_id: 이벤트 고유 ID (자동 생성)
    
    Example:
        >>> event = Event(
        ...     type=EventType.FIRE,
        ...     stage=EventStage.DETECTED,
        ...     confidence=0.87,
        ...     stream_id="camera_01",
        ...     module_name="FireDetectModule",
        ...     details={"bbox": [100, 200, 300, 400]}
        ... )
        >>> print(event.to_dict())
    """
    
    type: EventType
    stage: EventStage
    confidence: float
    stream_id: str
    module_name: str
    latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: _generate_event_id())
    
    def __post_init__(self) -> None:
        """유효성 검사"""
        # confidence 범위 검사
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence는 0.0~1.0 범위여야 합니다: {self.confidence}")
        
        # EventType 변환 (문자열로 전달된 경우)
        if isinstance(self.type, str):
            self.type = EventType(self.type)
        
        # EventStage 변환 (문자열로 전달된 경우)
        if isinstance(self.stage, str):
            self.stage = EventStage(self.stage)
    
    @property
    def bbox(self) -> BoundingBox | None:
        """
        바운딩 박스를 반환합니다.
        
        details에 bbox 정보가 있으면 BoundingBox 객체로 변환합니다.
        """
        bbox_data = self.details.get("bbox")
        if bbox_data is None:
            return None
        
        if isinstance(bbox_data, BoundingBox):
            return bbox_data
        
        if isinstance(bbox_data, list) and len(bbox_data) == 4:
            return BoundingBox(
                x=bbox_data[0],
                y=bbox_data[1],
                width=bbox_data[2],
                height=bbox_data[3],
            )
        
        if isinstance(bbox_data, dict):
            return BoundingBox(**bbox_data)
        
        return None
    
    @property
    def tracking_id(self) -> str | None:
        """추적 ID (객체 추적 시 사용)"""
        return self.details.get("tracking_id")
    
    @property
    def frame_number(self) -> int | None:
        """이벤트가 발생한 프레임 번호"""
        return self.details.get("frame_number")
    
    def to_dict(self) -> dict[str, Any]:
        """
        딕셔너리로 변환합니다.
        
        REST API 응답 및 VMS 전송에 사용됩니다.
        """
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "stage": self.stage.value,
            "confidence": round(self.confidence, 4),
            "ts": self.ts,
            "stream_id": self.stream_id,
            "module_name": self.module_name,
            "latency_ms": round(self.latency_ms, 2),
            "details": self._serialize_details(),
        }
    
    def _serialize_details(self) -> dict[str, Any]:
        """details를 직렬화 가능한 형태로 변환"""
        result: dict[str, Any] = {}
        for key, value in self.details.items():
            if isinstance(value, BoundingBox):
                result[key] = value.to_dict()
            elif hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """딕셔너리에서 Event 객체를 생성합니다."""
        return cls(
            type=EventType(data["type"]),
            stage=EventStage(data["stage"]),
            confidence=data["confidence"],
            ts=data.get("ts", time.time()),
            stream_id=data["stream_id"],
            module_name=data["module_name"],
            latency_ms=data.get("latency_ms", 0.0),
            details=data.get("details", {}),
            event_id=data.get("event_id", _generate_event_id()),
        )
    
    def with_stage(self, stage: EventStage) -> "Event":
        """단계가 변경된 새 이벤트를 반환합니다."""
        return Event(
            type=self.type,
            stage=stage,
            confidence=self.confidence,
            ts=time.time(),  # 새 타임스탬프
            stream_id=self.stream_id,
            module_name=self.module_name,
            latency_ms=self.latency_ms,
            details=self.details.copy(),
            event_id=self.event_id,  # 동일 이벤트 ID 유지
        )
    
    def __str__(self) -> str:
        return (
            f"Event({self.type.value}/{self.stage.value}, "
            f"conf={self.confidence:.2f}, stream={self.stream_id})"
        )


def _generate_event_id() -> str:
    """
    이벤트 고유 ID를 생성합니다.
    
    형식: {timestamp_ms}_{random_hex}
    예: 1702195200123_a1b2c3d4
    """
    import random
    timestamp_ms = int(time.time() * 1000)
    random_hex = format(random.getrandbits(32), "08x")
    return f"{timestamp_ms}_{random_hex}"


@dataclass
class EventFilter:
    """
    이벤트 필터
    
    이벤트 로그 조회 시 필터링 조건을 정의합니다.
    
    Attributes:
        types: 포함할 이벤트 유형 목록 (None이면 전체)
        stages: 포함할 이벤트 단계 목록 (None이면 전체)
        stream_ids: 포함할 스트림 ID 목록 (None이면 전체)
        module_names: 포함할 모듈 이름 목록 (None이면 전체)
        min_confidence: 최소 신뢰도
        start_ts: 시작 타임스탬프
        end_ts: 종료 타임스탬프
    """
    
    types: list[EventType] | None = None
    stages: list[EventStage] | None = None
    stream_ids: list[str] | None = None
    module_names: list[str] | None = None
    min_confidence: float = 0.0
    start_ts: float | None = None
    end_ts: float | None = None
    
    def matches(self, event: Event) -> bool:
        """이벤트가 필터 조건에 맞는지 확인합니다."""
        if self.types and event.type not in self.types:
            return False
        
        if self.stages and event.stage not in self.stages:
            return False
        
        if self.stream_ids and event.stream_id not in self.stream_ids:
            return False
        
        if self.module_names and event.module_name not in self.module_names:
            return False
        
        if event.confidence < self.min_confidence:
            return False
        
        if self.start_ts and event.ts < self.start_ts:
            return False
        
        if self.end_ts and event.ts > self.end_ts:
            return False
        
        return True

