"""
파이프라인 엔진

프레임과 오디오를 처리하는 모듈형 파이프라인을 구현합니다.
"""

from __future__ import annotations

import asyncio
import time
import threading
from typing import TYPE_CHECKING, Any, Callable

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.domain.interfaces.module import ModuleBase
from sentinel_pipeline.domain.models.event import Event
from sentinel_pipeline.application.pipeline.scheduler import ModuleScheduler

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# 타입 정의
FrameType = Any
AudioChunkType = Any
MetadataType = dict[str, Any]


class PipelineEngine:
    """
    파이프라인 엔진
    
    등록된 모듈들을 순차적으로 실행하여 프레임/오디오를 처리합니다.
    모듈 간 예외 격리, 타임아웃 관리, 메타데이터 전달을 담당합니다.
    
    Attributes:
        scheduler: 모듈 스케줄러
    
    Example:
        >>> engine = PipelineEngine()
        >>> engine.register_module(FireDetectModule())
        >>> engine.register_module(FaceBlurModule())
        >>> 
        >>> frame, events = engine.process_frame(frame, metadata)
        >>> for event in events:
        ...     print(f"감지: {event.type}")
    """
    
    def __init__(
        self,
        max_consecutive_errors: int = 5,
        max_consecutive_timeouts: int = 10,
        max_workers: int | None = None,
        error_window_seconds: float = 60.0,
        cooldown_seconds: float = 300.0,
    ) -> None:
        """
        파이프라인 엔진 초기화
        
        Args:
            max_consecutive_errors: 연속 에러 시 모듈 비활성화 임계치
            max_consecutive_timeouts: 연속 타임아웃 시 모듈 비활성화 임계치
            max_workers: 모듈 실행 스레드 풀 크기
            error_window_seconds: 에러 카운트 윈도우 (초)
            cooldown_seconds: 비활성화 후 재활성화 대기 시간 (초)
        """
        self.scheduler = ModuleScheduler(
            max_consecutive_errors=max_consecutive_errors,
            max_consecutive_timeouts=max_consecutive_timeouts,
            max_workers=max_workers,
            error_window_seconds=error_window_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        
        self._lock = threading.RLock()
        self._running = False
        
        # 이벤트 콜백
        self._on_events: Callable[[list[Event]], None] | None = None
        
        # 통계
        self._total_frames = 0
        self._total_events = 0
        self._start_time: float | None = None
        self._stats_task_started = False
    
    def register_module(self, module: ModuleBase) -> None:
        """
        모듈을 파이프라인에 등록합니다.
        
        Args:
            module: 등록할 모듈
        """
        self.scheduler.register(module)
    
    def unregister_module(self, name: str) -> bool:
        """
        모듈을 파이프라인에서 제거합니다.
        
        Args:
            name: 모듈 이름
        
        Returns:
            성공 여부
        """
        return self.scheduler.unregister(name)
    
    def process_frame(
        self,
        frame: FrameType,
        metadata: MetadataType,
    ) -> tuple[FrameType, list[Event]]:
        """
        프레임을 파이프라인으로 처리합니다.
        
        등록된 모듈을 priority 순으로 실행하며,
        각 모듈의 출력 프레임과 메타데이터가 다음 모듈로 전달됩니다.
        
        Args:
            frame: 입력 프레임
            metadata: 프레임 메타데이터
                - stream_id: 스트림 ID
                - frame_number: 프레임 번호
                - timestamp: 타임스탬프
        
        Returns:
            (처리된 프레임, 생성된 이벤트 목록)
        """
        if self._start_time is None:
            self._start_time = time.time()
        
        all_events: list[Event] = []
        current_frame = frame
        current_metadata = metadata.copy()
        
        # 처리 시작 시간 기록
        pipeline_start = time.perf_counter()
        current_metadata["_pipeline_start_ts"] = pipeline_start
        
        # priority 순으로 모듈 실행
        for context in self.scheduler.get_execution_order():
            module = context.module
            
            # 메타데이터에 현재 모듈 정보 추가
            current_metadata["_current_module"] = module.name
            
            # 타임아웃 적용하여 실행
            result = self.scheduler.execute_with_timeout(
                context, current_frame, current_metadata
            )
            
            if result is None:
                # 타임아웃 또는 에러 - 현재 프레임/메타데이터 유지, 다음 모듈 실행
                continue
            
            # 결과 적용
            current_frame, events, current_metadata = result
            
            # 이벤트에 처리 시간 기록
            if events:
                latency_ms = (time.perf_counter() - pipeline_start) * 1000
                for event in events:
                    event.latency_ms = latency_ms
                all_events.extend(events)
        
        # 통계 업데이트
        self._total_frames += 1
        self._total_events += len(all_events)
        
        # 이벤트 콜백 호출
        if all_events and self._on_events:
            try:
                self._on_events(all_events)
            except Exception as e:
                logger.error(f"이벤트 콜백 오류: {e}")
        
        return current_frame, all_events
    
    def process_audio(
        self,
        chunk: AudioChunkType,
        metadata: MetadataType,
    ) -> list[Event]:
        """
        오디오 청크를 파이프라인으로 처리합니다.
        
        Args:
            chunk: 오디오 청크
            metadata: 오디오 메타데이터
        
        Returns:
            생성된 이벤트 목록
        """
        all_events: list[Event] = []
        current_metadata = metadata.copy()
        
        # priority 순으로 모듈 실행
        for context in self.scheduler.get_execution_order():
            module = context.module
            
            try:
                events, current_metadata = module.process_audio(chunk, current_metadata)
                all_events.extend(events)
            except Exception as e:
                logger.error(
                    f"오디오 처리 오류: {module.name} - {e}",
                    module_name=module.name,
                    error=str(e),
                )
        
        # 이벤트 콜백 호출
        if all_events and self._on_events:
            try:
                self._on_events(all_events)
            except Exception as e:
                logger.error(f"이벤트 콜백 오류: {e}")
        
        return all_events
    
    def reload_modules(self, module_configs: list[dict[str, Any]]) -> None:
        """
        모듈 설정을 다시 로드합니다.
        
        런타임에 설정이 변경되었을 때 호출됩니다.
        
        Args:
            module_configs: 새 모듈 설정 목록
        """
        with self._lock:
            for config in module_configs:
                name = config.get("name")
                if not name:
                    continue
                
                ctx = self.scheduler.get_module(name)
                if not ctx:
                    logger.warning(f"설정 변경 대상 모듈 없음: {name}")
                    continue
                
                module = ctx.module
                
                # enabled 변경
                if "enabled" in config:
                    module.enabled = config["enabled"]
                
                # priority 변경
                if "priority" in config and hasattr(module, "priority"):
                    module.priority = config["priority"]
                
                # timeout_ms 변경
                if "timeout_ms" in config and hasattr(module, "timeout_ms"):
                    module.timeout_ms = config["timeout_ms"]
                
                # options 변경
                if "options" in config and hasattr(module, "options"):
                    module.options.update(config["options"])
                
                logger.info(
                    f"모듈 설정 변경: {name}",
                    module_name=name,
                    enabled=module.enabled,
                )
    
    def enable_module(self, name: str) -> bool:
        """모듈을 활성화합니다."""
        return self.scheduler.enable_module(name)
    
    def disable_module(self, name: str) -> bool:
        """모듈을 비활성화합니다."""
        return self.scheduler.disable_module(name)
    
    def set_on_events(self, callback: Callable[[list[Event]], None] | None) -> None:
        """
        이벤트 발생 콜백을 설정합니다.
        
        Args:
            callback: 이벤트 목록을 받는 콜백 함수
        """
        self._on_events = callback
    
    def get_module_stats(self) -> dict[str, dict[str, Any]]:
        """모든 모듈의 통계를 반환합니다."""
        return self.scheduler.get_stats()
    
    def get_stats(self) -> dict[str, Any]:
        """
        파이프라인 전체 통계를 반환합니다.
        
        Returns:
            통계 딕셔너리
        """
        uptime = 0.0
        if self._start_time:
            uptime = time.time() - self._start_time
        
        return {
            "total_frames": self._total_frames,
            "total_events": self._total_events,
            "uptime_seconds": round(uptime, 1),
            "modules": self.get_module_stats(),
        }

    def start_stats_publisher(self, interval_sec: float = 5.0) -> None:
        """모듈 통계를 주기적으로 WS로 전송합니다."""
        if self._stats_task_started:
            return
        self._stats_task_started = True

        async def _loop():
            from sentinel_pipeline.interface.api.ws_bus import publish_module_stats

            while True:
                stats = self.get_module_stats()
                payload = {"modules": stats, "ts": time.time()}
                try:
                    await publish_module_stats(payload)
                except Exception as e:
                    logger.debug("module_stats publish failed", error=str(e))
                await asyncio.sleep(interval_sec)

        try:
            # 실행 중인 이벤트 루프가 있는지 확인
            loop = asyncio.get_running_loop()
            loop.create_task(_loop())
        except RuntimeError:
            # 이벤트 루프가 없으면 나중에 FastAPI가 시작되면 자동으로 시작됨
            logger.debug("event loop not running; module_stats publisher will start when FastAPI starts")
    
    def shutdown(self) -> None:
        """파이프라인을 종료합니다."""
        self.scheduler.shutdown()
        logger.info("파이프라인 엔진 종료")

