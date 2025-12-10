"""
이벤트 발행자

감지 이벤트의 큐잉, 배치 처리, 전송을 담당합니다.
"""

from __future__ import annotations

import time
import queue
import threading
from typing import TYPE_CHECKING, Any, Callable, Protocol
from enum import Enum

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.domain.models.event import Event

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class DropStrategy(str, Enum):
    """큐 드롭 전략"""
    DROP_OLDEST = "drop_oldest"  # 가장 오래된 이벤트 폐기 (실시간성 우선)
    DROP_NEWEST = "drop_newest"  # 새 이벤트 거부 (데이터 손실 최소화)


class TransportProtocol(Protocol):
    """이벤트 전송 인터페이스 (Infrastructure에서 구현)"""
    
    async def send_event(self, event: Event) -> bool: ...
    async def send_events(self, events: list[Event]) -> bool: ...


class EventEmitter:
    """
    이벤트 발행자
    
    감지 이벤트를 수집하고, 배치로 묶어 전송합니다.
    큐 백프레셔와 드롭 정책을 관리합니다.
    
    Attributes:
        max_queue_size: 최대 큐 크기
        batch_size: 배치 크기
        flush_interval_ms: 플러시 주기 (밀리초)
        drop_strategy: 큐 초과 시 드롭 전략
    
    Example:
        >>> emitter = EventEmitter(max_queue_size=1000)
        >>> emitter.set_transport(http_client)
        >>> emitter.start()
        >>> 
        >>> emitter.emit(event)
        >>> 
        >>> emitter.stop()
    """
    
    DEFAULT_MAX_QUEUE_SIZE = 1000
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_FLUSH_INTERVAL_MS = 100
    
    def __init__(
        self,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval_ms: int = DEFAULT_FLUSH_INTERVAL_MS,
        drop_strategy: DropStrategy | str = DropStrategy.DROP_OLDEST,
    ) -> None:
        """
        이벤트 발행자 초기화
        
        Args:
            max_queue_size: 최대 큐 크기
            batch_size: 배치 크기
            flush_interval_ms: 플러시 주기 (밀리초)
            drop_strategy: 큐 초과 시 드롭 전략
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        
        if isinstance(drop_strategy, str):
            self.drop_strategy = DropStrategy(drop_strategy)
        else:
            self.drop_strategy = drop_strategy
        
        self._queue: queue.Queue[Event] = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        # 전송 클라이언트 (Infrastructure에서 설정)
        self._sync_transport: Callable[[list[Event]], bool] | None = None
        
        # 통계
        self._total_emitted = 0
        self._total_dropped = 0
        self._total_sent = 0
        self._total_failed = 0
        
        # 콜백
        self._on_events_emitted: Callable[[list[Event]], None] | None = None
    
    def set_transport(
        self,
        transport: Callable[[list[Event]], bool],
    ) -> None:
        """
        전송 클라이언트를 설정합니다.
        
        Args:
            transport: 이벤트 목록을 받아 전송하는 함수
        """
        self._sync_transport = transport
    
    def set_on_events_emitted(
        self,
        callback: Callable[[list[Event]], None] | None,
    ) -> None:
        """이벤트 발행 콜백을 설정합니다."""
        self._on_events_emitted = callback
    
    def start(self) -> None:
        """이벤트 발행자를 시작합니다."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._thread = threading.Thread(
            target=self._emit_loop,
            name="event_emitter",
            daemon=True,
        )
        self._thread.start()
        
        logger.info(
            f"이벤트 발행자 시작: queue_max={self.max_queue_size}, "
            f"batch_size={self.batch_size}"
        )
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        이벤트 발행자를 중지합니다.
        
        Args:
            timeout: 종료 타임아웃 (초)
        """
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        
        logger.info("이벤트 발행자 중지")
    
    def emit(self, event: Event) -> bool:
        """
        이벤트를 발행합니다.
        
        Args:
            event: 발행할 이벤트
        
        Returns:
            성공 여부 (큐에 추가됨)
        """
        return self.emit_batch([event])
    
    def emit_batch(self, events: list[Event]) -> bool:
        """
        여러 이벤트를 발행합니다.
        
        Args:
            events: 발행할 이벤트 목록
        
        Returns:
            성공 여부 (모두 큐에 추가됨)
        """
        if not events:
            return True
        
        success = True
        
        for event in events:
            try:
                self._queue.put_nowait(event)
                self._total_emitted += 1
            except queue.Full:
                # 큐가 가득 찬 경우 드롭 전략 적용
                if self.drop_strategy == DropStrategy.DROP_OLDEST:
                    # 오래된 이벤트 제거하고 새 이벤트 추가
                    try:
                        dropped = self._queue.get_nowait()
                        self._total_dropped += 1
                        logger.warning(
                            f"이벤트 드롭 (drop_oldest): {dropped.type.value}",
                            event_type=dropped.type.value,
                        )
                        self._queue.put_nowait(event)
                        self._total_emitted += 1
                    except queue.Empty:
                        pass
                else:
                    # 새 이벤트 거부
                    self._total_dropped += 1
                    logger.warning(
                        f"이벤트 드롭 (drop_newest): {event.type.value}",
                        event_type=event.type.value,
                    )
                    success = False
        
        return success
    
    def _emit_loop(self) -> None:
        """이벤트 발행 루프"""
        flush_interval_sec = self.flush_interval_ms / 1000.0
        
        while self._running and not self._stop_event.is_set():
            try:
                # 배치 수집
                batch = self._collect_batch()
                
                if batch:
                    # 콜백 호출
                    if self._on_events_emitted:
                        try:
                            self._on_events_emitted(batch)
                        except Exception as e:
                            logger.error(f"이벤트 발행 콜백 오류: {e}")
                    
                    # 전송
                    if self._sync_transport:
                        try:
                            success = self._sync_transport(batch)
                            if success:
                                self._total_sent += len(batch)
                            else:
                                self._total_failed += len(batch)
                        except Exception as e:
                            self._total_failed += len(batch)
                            logger.error(f"이벤트 전송 오류: {e}")
                
                # 다음 플러시까지 대기
                self._stop_event.wait(flush_interval_sec)
                
            except Exception as e:
                logger.error(f"이벤트 발행 루프 오류: {e}")
    
    def _collect_batch(self) -> list[Event]:
        """배치 크기만큼 이벤트를 수집합니다."""
        batch = []
        
        while len(batch) < self.batch_size:
            try:
                event = self._queue.get_nowait()
                batch.append(event)
            except queue.Empty:
                break
        
        return batch
    
    def flush(self, timeout: float = 10.0) -> int:
        """
        큐에 있는 모든 이벤트를 즉시 전송합니다.
        
        Args:
            timeout: 타임아웃 (초)
        
        Returns:
            전송된 이벤트 수
        """
        sent_count = 0
        start_time = time.time()
        
        while not self._queue.empty():
            if time.time() - start_time > timeout:
                logger.warning("이벤트 플러시 타임아웃")
                break
            
            batch = self._collect_batch()
            if not batch:
                break
            
            if self._sync_transport:
                try:
                    success = self._sync_transport(batch)
                    if success:
                        sent_count += len(batch)
                        self._total_sent += len(batch)
                    else:
                        self._total_failed += len(batch)
                except Exception as e:
                    self._total_failed += len(batch)
                    logger.error(f"플러시 중 전송 오류: {e}")
        
        logger.info(f"이벤트 플러시 완료: {sent_count}개 전송")
        return sent_count
    
    @property
    def queue_size(self) -> int:
        """현재 큐 크기"""
        return self._queue.qsize()
    
    @property
    def is_full(self) -> bool:
        """큐가 가득 찼는지 확인"""
        return self._queue.full()
    
    def get_stats(self) -> dict[str, Any]:
        """
        통계를 반환합니다.
        
        Returns:
            통계 딕셔너리
        """
        return {
            "queue_size": self.queue_size,
            "max_queue_size": self.max_queue_size,
            "total_emitted": self._total_emitted,
            "total_dropped": self._total_dropped,
            "total_sent": self._total_sent,
            "total_failed": self._total_failed,
            "drop_rate": (
                self._total_dropped / self._total_emitted
                if self._total_emitted > 0 else 0.0
            ),
            "success_rate": (
                self._total_sent / (self._total_sent + self._total_failed)
                if (self._total_sent + self._total_failed) > 0 else 1.0
            ),
        }
    
    def reset_stats(self) -> None:
        """통계를 초기화합니다."""
        self._total_emitted = 0
        self._total_dropped = 0
        self._total_sent = 0
        self._total_failed = 0

