"""
오디오 스트림 모델

오디오 스트림의 설정과 상태를 나타내는 데이터 구조를 정의합니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from sentinel_pipeline.interface.config.schema import AudioStreamConfig


class AudioStreamStatus(str, Enum):
    """
    오디오 스트림 상태
    """
    
    IDLE = "IDLE"                   # 유휴 상태
    STARTING = "STARTING"           # 시작 중
    RUNNING = "RUNNING"             # 실행 중
    STOPPING = "STOPPING"           # 중지 중
    STOPPED = "STOPPED"             # 중지됨
    ERROR = "ERROR"                 # 오류 발생


@dataclass
class AudioStreamStats:
    """
    오디오 스트림 통계
    """
    
    chunk_count: int = 0
    event_count: int = 0
    error_count: int = 0
    last_chunk_ts: float | None = None
    start_ts: float | None = None
    scream_detected_count: int = 0
    keyword_detected_count: int = 0
    
    def record_chunk(self) -> None:
        """청크 처리를 기록합니다."""
        self.chunk_count += 1
        self.last_chunk_ts = time.time()
        
        if self.start_ts is None:
            self.start_ts = self.last_chunk_ts
    
    def record_event(self) -> None:
        """이벤트 생성을 기록합니다."""
        self.event_count += 1
    
    def record_error(self) -> None:
        """에러 발생을 기록합니다."""
        self.error_count += 1
        
    @property
    def uptime_seconds(self) -> float:
        """가동 시간 (초)"""
        if self.start_ts is None:
            return 0.0
        return time.time() - self.start_ts
    
    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "chunk_count": self.chunk_count,
            "event_count": self.event_count,
            "error_count": self.error_count,
            "scream_detected_count": self.scream_detected_count,
            "keyword_detected_count": self.keyword_detected_count,
            "last_chunk_ts": self.last_chunk_ts,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }
    
    def reset(self) -> None:
        """통계를 초기화합니다."""
        self.chunk_count = 0
        self.event_count = 0
        self.error_count = 0
        self.scream_detected_count = 0
        self.keyword_detected_count = 0
        self.last_chunk_ts = None
        self.start_ts = None


@dataclass
class AudioStreamState:
    """
    오디오 스트림 상태
    
    스트림의 현재 상태와 통계를 포함하는 종합 정보입니다.
    """
    
    config: AudioStreamConfig
    status: AudioStreamStatus = AudioStreamStatus.IDLE
    stats: AudioStreamStats = field(default_factory=AudioStreamStats)
    last_error: str | None = None
    
    @property
    def stream_id(self) -> str:
        """스트림 ID"""
        return self.config.stream_id
    
    @property
    def is_active(self) -> bool:
        """스트림이 활성 상태인지 확인"""
        return self.status in (AudioStreamStatus.STARTING, AudioStreamStatus.RUNNING)
    
    def set_status(self, status: AudioStreamStatus, error: str | None = None) -> None:
        """상태를 변경합니다."""
        self.status = status
        if error:
            self.last_error = error
            
    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "stream_id": self.stream_id,
            "config": self.config.model_dump(),
            "status": self.status.value,
            "stats": self.stats.to_dict(),
            "last_error": self.last_error,
            "is_active": self.is_active,
        }
