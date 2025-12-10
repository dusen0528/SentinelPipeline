"""
헬스 워처

스트림과 시스템의 상태를 모니터링합니다.
"""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING, Any, Callable

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.domain.models.stream import StreamStatus

if TYPE_CHECKING:
    from sentinel_pipeline.application.stream.manager import StreamManager

logger = get_logger(__name__)


class HealthWatcher:
    """
    헬스 워처
    
    스트림의 상태를 주기적으로 모니터링하고,
    문제가 감지되면 적절한 조치를 취합니다.
    
    모니터링 항목:
    - 프레임 수신 여부 (last_frame_ts)
    - FPS 저하
    - 에러 카운트
    - 메모리/CPU 사용량 (선택)
    
    Attributes:
        check_interval_seconds: 체크 주기 (초)
        frame_timeout_seconds: 프레임 타임아웃 (초)
        min_fps_threshold: 최소 FPS 임계치
    
    Example:
        >>> watcher = HealthWatcher(stream_manager)
        >>> watcher.start()
        >>> ...
        >>> watcher.stop()
    """
    
    DEFAULT_CHECK_INTERVAL = 5.0
    DEFAULT_FRAME_TIMEOUT = 10.0
    DEFAULT_MIN_FPS = 1.0
    
    def __init__(
        self,
        stream_manager: "StreamManager",
        check_interval_seconds: float = DEFAULT_CHECK_INTERVAL,
        frame_timeout_seconds: float = DEFAULT_FRAME_TIMEOUT,
        min_fps_threshold: float = DEFAULT_MIN_FPS,
    ) -> None:
        """
        헬스 워처 초기화
        
        Args:
            stream_manager: 스트림 관리자
            check_interval_seconds: 체크 주기 (초)
            frame_timeout_seconds: 프레임 타임아웃 (초)
            min_fps_threshold: 최소 FPS 임계치
        """
        self._stream_manager = stream_manager
        self.check_interval_seconds = check_interval_seconds
        self.frame_timeout_seconds = frame_timeout_seconds
        self.min_fps_threshold = min_fps_threshold
        
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        # 콜백
        self._on_unhealthy: Callable[[str, str], None] | None = None
        self._on_recovered: Callable[[str], None] | None = None
        
        # 상태 추적
        self._unhealthy_streams: set[str] = set()
    
    def start(self) -> None:
        """헬스 워처를 시작합니다."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="health_watcher",
            daemon=True,
        )
        self._thread.start()
        
        logger.info("헬스 워처 시작")
    
    def stop(self) -> None:
        """헬스 워처를 중지합니다."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        logger.info("헬스 워처 중지")
    
    def set_on_unhealthy(
        self,
        callback: Callable[[str, str], None] | None,
    ) -> None:
        """
        비정상 상태 감지 콜백을 설정합니다.
        
        Args:
            callback: (stream_id, reason) -> None
        """
        self._on_unhealthy = callback
    
    def set_on_recovered(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        """
        복구 감지 콜백을 설정합니다.
        
        Args:
            callback: (stream_id) -> None
        """
        self._on_recovered = callback
    
    def _watch_loop(self) -> None:
        """모니터링 루프"""
        while self._running and not self._stop_event.is_set():
            try:
                self._check_streams()
            except Exception as e:
                logger.error(f"헬스 체크 오류: {e}")
            
            self._stop_event.wait(self.check_interval_seconds)
    
    def _check_streams(self) -> None:
        """모든 스트림의 상태를 확인합니다."""
        now = time.time()
        
        for state in self._stream_manager.get_all_streams():
            stream_id = state.stream_id
            
            # 실행 중인 스트림만 확인
            if state.status != StreamStatus.RUNNING:
                continue
            
            issues = []
            
            # 프레임 타임아웃 확인
            if state.stats.last_frame_ts:
                elapsed = now - state.stats.last_frame_ts
                if elapsed > self.frame_timeout_seconds:
                    issues.append(f"프레임 타임아웃 ({elapsed:.1f}초)")
            
            # FPS 확인
            if state.stats.fps < self.min_fps_threshold:
                issues.append(f"낮은 FPS ({state.stats.fps:.1f})")
            
            # 문제 감지됨
            if issues:
                if stream_id not in self._unhealthy_streams:
                    self._unhealthy_streams.add(stream_id)
                    reason = ", ".join(issues)
                    
                    logger.warning(
                        f"스트림 비정상: {stream_id} - {reason}",
                        stream_id=stream_id,
                        issues=issues,
                    )
                    
                    if self._on_unhealthy:
                        try:
                            self._on_unhealthy(stream_id, reason)
                        except Exception as e:
                            logger.error(f"비정상 콜백 오류: {e}")
            else:
                # 복구됨
                if stream_id in self._unhealthy_streams:
                    self._unhealthy_streams.discard(stream_id)
                    
                    logger.info(
                        f"스트림 복구: {stream_id}",
                        stream_id=stream_id,
                    )
                    
                    if self._on_recovered:
                        try:
                            self._on_recovered(stream_id)
                        except Exception as e:
                            logger.error(f"복구 콜백 오류: {e}")
    
    def check_stream(self, stream_id: str) -> dict[str, Any]:
        """
        특정 스트림의 헬스 상태를 확인합니다.
        
        Args:
            stream_id: 스트림 ID
        
        Returns:
            헬스 상태 딕셔너리
        """
        state = self._stream_manager.get_stream_state(stream_id)
        if not state:
            return {
                "stream_id": stream_id,
                "status": "not_found",
                "healthy": False,
            }
        
        now = time.time()
        issues = []
        
        if state.status != StreamStatus.RUNNING:
            issues.append(f"상태: {state.status.value}")
        
        if state.stats.last_frame_ts:
            elapsed = now - state.stats.last_frame_ts
            if elapsed > self.frame_timeout_seconds:
                issues.append(f"프레임 타임아웃 ({elapsed:.1f}초)")
        
        if state.stats.fps < self.min_fps_threshold:
            issues.append(f"낮은 FPS ({state.stats.fps:.1f})")
        
        return {
            "stream_id": stream_id,
            "status": state.status.value,
            "healthy": len(issues) == 0,
            "issues": issues,
            "fps": round(state.stats.fps, 1),
            "last_frame_ts": state.stats.last_frame_ts,
            "error_count": state.stats.error_count,
        }
    
    def get_overall_health(self) -> dict[str, Any]:
        """
        전체 시스템 헬스 상태를 반환합니다.
        
        Returns:
            전체 헬스 상태 딕셔너리
        """
        streams = self._stream_manager.get_all_streams()
        active_count = sum(1 for s in streams if s.is_active)
        healthy_count = sum(1 for s in streams if s.is_healthy)
        
        return {
            "status": "healthy" if healthy_count == active_count else "degraded",
            "total_streams": len(streams),
            "active_streams": active_count,
            "healthy_streams": healthy_count,
            "unhealthy_streams": list(self._unhealthy_streams),
        }


def calculate_exponential_backoff(
    retry_count: int,
    base_delay: float,
    max_delay: float,
    jitter: bool = True,
) -> float:
    """
    지수 백오프 지연 시간을 계산합니다.
    
    Args:
        retry_count: 재시도 횟수 (0부터 시작)
        base_delay: 기본 지연 시간 (초)
        max_delay: 최대 지연 시간 (초)
        jitter: 지터 추가 여부 (충돌 방지)
    
    Returns:
        계산된 지연 시간 (초)
    
    Example:
        >>> calculate_exponential_backoff(0, 1.0, 8.0)  # ~1초
        >>> calculate_exponential_backoff(1, 1.0, 8.0)  # ~2초
        >>> calculate_exponential_backoff(2, 1.0, 8.0)  # ~4초
        >>> calculate_exponential_backoff(3, 1.0, 8.0)  # ~8초 (최대)
    """
    delay = base_delay * (2 ** retry_count)
    delay = min(delay, max_delay)
    
    if jitter:
        import random
        # 0.5 ~ 1.5 범위의 지터 적용
        jitter_factor = 0.5 + random.random()
        delay *= jitter_factor
    
    return delay

