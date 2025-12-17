"""
스트림 모델

RTSP 스트림의 설정과 상태를 나타내는 데이터 구조를 정의합니다.
이 모듈은 외부 라이브러리에 의존하지 않습니다.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any



class StreamStatus(str, Enum):
    """
    스트림 상태
    
    스트림의 현재 동작 상태를 나타냅니다.
    """
    
    IDLE = "IDLE"                   # 유휴 상태 (초기 상태)
    STARTING = "STARTING"           # 시작 중
    RUNNING = "RUNNING"             # 실행 중
    STOPPING = "STOPPING"           # 중지 중
    STOPPED = "STOPPED"             # 중지됨
    ERROR = "ERROR"                 # 오류 발생
    RECONNECTING = "RECONNECTING"   # 재연결 중


@dataclass
class StreamConfig:
    """
    스트림 설정
    
    RTSP 스트림 연결 및 처리에 필요한 설정을 정의합니다.
    
    Attributes:
        stream_id: 스트림 고유 식별자
        rtsp_url: RTSP 스트림 URL
        enabled: 활성화 여부
        max_fps: 최대 처리 FPS (원본보다 높으면 원본 사용)
        downscale: 프레임 축소 비율 (0.0 < downscale <= 1.0)
        buffer_size: 프레임 버퍼 크기
        reconnect_enabled: 자동 재연결 활성화
        reconnect_max_retries: 최대 재연결 시도 횟수
        reconnect_base_delay: 재연결 기본 대기 시간 (초)
        reconnect_max_delay: 재연결 최대 대기 시간 (초)
    
    Example:
        >>> config = StreamConfig(
        ...     stream_id="camera_01",
        ...     rtsp_url="rtsp://admin:pass@192.168.1.100:554/stream1",
        ...     max_fps=15,
        ...     downscale=0.5,
        ... )
    """
    
    stream_id: str
    rtsp_url: str
    enabled: bool = True
    max_fps: int = 15
    downscale: float = 1.0
    buffer_size: int = 2
    
    # 재연결 설정
    reconnect_enabled: bool = True
    reconnect_max_retries: int = 5
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 8.0
    
    # FFmpeg 출력 설정 (선택)
    output_url: str | None = None
    output_enabled: bool = True
    
    def __post_init__(self) -> None:
        """유효성 검사"""
        if not self.stream_id:
            raise ValueError("stream_id는 필수입니다")
        
        if not self.rtsp_url:
            raise ValueError("rtsp_url은 필수입니다")
        
        if self.max_fps <= 0:
            raise ValueError(f"max_fps는 양수여야 합니다: {self.max_fps}")
        
        if not 0.0 < self.downscale <= 1.0:
            raise ValueError(f"downscale은 0.0~1.0 범위여야 합니다: {self.downscale}")
        
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size는 1 이상이어야 합니다: {self.buffer_size}")
    
    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "stream_id": self.stream_id,
            "rtsp_url": self._mask_url(self.rtsp_url),  # 비밀번호 마스킹
            "enabled": self.enabled,
            "max_fps": self.max_fps,
            "downscale": self.downscale,
            "buffer_size": self.buffer_size,
            "reconnect_enabled": self.reconnect_enabled,
            "reconnect_max_retries": self.reconnect_max_retries,
            "reconnect_base_delay": self.reconnect_base_delay,
            "reconnect_max_delay": self.reconnect_max_delay,
            "output_url": self._mask_url(self.output_url) if self.output_url else None,
            "output_enabled": self.output_enabled,
        }
    
    @staticmethod
    def _mask_url(url: str) -> str:
        """URL에서 비밀번호를 마스킹합니다."""
        # rtsp://admin:password@host:port/path
        # -> rtsp://admin:****@host:port/path
        return re.sub(r"(://[^:]+:)[^@]+(@)", r"\1****\2", url)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamConfig":
        """딕셔너리에서 StreamConfig 객체를 생성합니다."""
        return cls(
            stream_id=data["stream_id"],
            rtsp_url=data["rtsp_url"],
            enabled=data.get("enabled", True),
            max_fps=data.get("max_fps", 15),
            downscale=data.get("downscale", 1.0),
            buffer_size=data.get("buffer_size", 2),
            reconnect_enabled=data.get("reconnect_enabled", True),
            reconnect_max_retries=data.get("reconnect_max_retries", 5),
            reconnect_base_delay=data.get("reconnect_base_delay", 1.0),
            reconnect_max_delay=data.get("reconnect_max_delay", 8.0),
            output_url=data.get("output_url"),
            output_enabled=data.get("output_enabled", True),
        )


@dataclass
class StreamStats:
    """
    스트림 통계
    
    스트림 처리의 실시간 통계 정보를 관리합니다.
    
    Attributes:
        frame_count: 처리된 프레임 수
        event_count: 생성된 이벤트 수
        error_count: 발생한 에러 수
        reconnect_count: 재연결 횟수
        fps: 현재 FPS
        avg_latency_ms: 평균 처리 지연 (밀리초)
        last_frame_ts: 마지막 프레임 타임스탬프
        start_ts: 스트림 시작 타임스탬프
    """
    
    frame_count: int = 0
    event_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    last_frame_ts: float | None = None
    start_ts: float | None = None
    
    # 내부 계산용
    _fps_frame_times: list[float] = field(default_factory=list)
    _latency_sum: float = 0.0
    
    def record_frame(self, latency_ms: float) -> None:
        """
        프레임 처리를 기록합니다.
        
        Args:
            latency_ms: 처리 지연 (밀리초)
        """
        now = time.time()
        self.frame_count += 1
        self.last_frame_ts = now
        self._latency_sum += latency_ms
        
        if self.start_ts is None:
            self.start_ts = now
        
        # FPS 계산 (최근 1초 기준)
        self._fps_frame_times.append(now)
        cutoff = now - 1.0
        self._fps_frame_times = [t for t in self._fps_frame_times if t > cutoff]
        self.fps = len(self._fps_frame_times)
        
        # 평균 지연 계산
        self.avg_latency_ms = self._latency_sum / self.frame_count
    
    def record_event(self, count: int = 1) -> None:
        """이벤트 생성을 기록합니다."""
        self.event_count += count
    
    def record_error(self) -> None:
        """에러 발생을 기록합니다."""
        self.error_count += 1
    
    def record_reconnect(self) -> None:
        """재연결을 기록합니다."""
        self.reconnect_count += 1
    
    @property
    def uptime_seconds(self) -> float:
        """가동 시간 (초)"""
        if self.start_ts is None:
            return 0.0
        return time.time() - self.start_ts
    
    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "frame_count": self.frame_count,
            "event_count": self.event_count,
            "error_count": self.error_count,
            "reconnect_count": self.reconnect_count,
            "fps": round(self.fps, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_frame_ts": self.last_frame_ts,
            "start_ts": self.start_ts,
            "uptime_seconds": round(self.uptime_seconds, 1),
        }
    
    def reset(self) -> None:
        """통계를 초기화합니다."""
        self.frame_count = 0
        self.event_count = 0
        self.error_count = 0
        self.reconnect_count = 0
        self.fps = 0.0
        self.avg_latency_ms = 0.0
        self.last_frame_ts = None
        self.start_ts = None
        self._fps_frame_times = []
        self._latency_sum = 0.0


@dataclass
class StreamState:
    """
    스트림 상태
    
    스트림의 현재 상태와 통계를 포함하는 종합 정보입니다.
    
    Attributes:
        config: 스트림 설정
        status: 현재 상태
        stats: 통계 정보
        last_error: 마지막 에러 메시지
        retry_count: 현재 재연결 시도 횟수
        next_retry_ts: 다음 재연결 시도 타임스탬프
    
    Example:
        >>> state = StreamState(
        ...     config=StreamConfig(stream_id="cam_01", rtsp_url="rtsp://..."),
        ...     status=StreamStatus.RUNNING,
        ... )
        >>> print(state.to_dict())
    """
    
    config: StreamConfig
    status: StreamStatus = StreamStatus.IDLE
    stats: StreamStats = field(default_factory=StreamStats)
    last_error: str | None = None
    retry_count: int = 0
    next_retry_ts: float | None = None
    
    @property
    def stream_id(self) -> str:
        """스트림 ID (편의 속성)"""
        return self.config.stream_id
    
    @property
    def is_active(self) -> bool:
        """스트림이 활성 상태인지 확인"""
        return self.status in (
            StreamStatus.STARTING,
            StreamStatus.RUNNING,
            StreamStatus.RECONNECTING,
        )
    
    @property
    def is_healthy(self) -> bool:
        """
        스트림이 정상 상태인지 확인
        
        최근 5초 내에 프레임이 수신되었으면 정상으로 판단합니다.
        """
        if self.status != StreamStatus.RUNNING:
            return False
        
        if self.stats.last_frame_ts is None:
            return False
        
        return (time.time() - self.stats.last_frame_ts) < 5.0
    
    def set_status(self, status: StreamStatus, error: str | None = None) -> None:
        """
        상태를 변경합니다.
        
        Args:
            status: 새 상태
            error: 에러 메시지 (ERROR 상태일 때)
        """
        self.status = status
        if error:
            self.last_error = error
    
    def calculate_next_retry_delay(self) -> float:
        """
        다음 재연결까지의 대기 시간을 계산합니다 (지수 백오프).
        
        Returns:
            대기 시간 (초)
        """
        delay = self.config.reconnect_base_delay * (2 ** self.retry_count)
        return min(delay, self.config.reconnect_max_delay)
    
    def record_retry(self) -> None:
        """재연결 시도를 기록합니다."""
        self.retry_count += 1
        self.stats.record_reconnect()
        self.next_retry_ts = time.time() + self.calculate_next_retry_delay()
    
    def reset_retry(self) -> None:
        """재연결 카운터를 초기화합니다."""
        self.retry_count = 0
        self.next_retry_ts = None
    
    def to_dict(self) -> dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "stream_id": self.stream_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "stats": self.stats.to_dict(),
            "last_error": self.last_error,
            "retry_count": self.retry_count,
            "next_retry_ts": self.next_retry_ts,
            "is_active": self.is_active,
            "is_healthy": self.is_healthy,
        }
    
    def to_summary(self) -> dict[str, Any]:
        """요약 정보를 딕셔너리로 변환 (대시보드용)"""
        return {
            "stream_id": self.stream_id,
            "status": self.status.value,
            "fps": round(self.stats.fps, 1),
            "frame_count": self.stats.frame_count,
            "event_count": self.stats.event_count,
            "error_count": self.stats.error_count,
            "last_error": self.last_error,
            "is_healthy": self.is_healthy,
        }

