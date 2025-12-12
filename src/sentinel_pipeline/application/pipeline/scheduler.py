"""
모듈 스케줄러

파이프라인 모듈의 실행 순서와 타임아웃을 관리합니다.
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TYPE_CHECKING, Any, Callable

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.domain.interfaces.module import ModuleBase, ModuleContext

if TYPE_CHECKING:
    from sentinel_pipeline.domain.models.event import Event

logger = get_logger(__name__)

# 타입 정의
FrameType = Any
MetadataType = dict[str, Any]
ProcessResult = tuple[FrameType, list["Event"], MetadataType]


class ModuleScheduler:
    """
    모듈 스케줄러
    
    등록된 모듈들을 priority 순으로 정렬하고,
    타임아웃 및 에러 관리를 수행합니다.
    
    Attributes:
        max_consecutive_errors: 연속 에러 시 비활성화 임계치
        max_consecutive_timeouts: 연속 타임아웃 시 비활성화 임계치
        error_window_seconds: 에러 카운트 윈도우 (초)
    
    Example:
        >>> scheduler = ModuleScheduler()
        >>> scheduler.register(fire_detect_module)
        >>> scheduler.register(face_blur_module)
        >>> 
        >>> for ctx in scheduler.get_execution_order():
        ...     result = scheduler.execute_with_timeout(ctx, frame, metadata)
    """
    
    # 기본 설정값
    DEFAULT_MAX_CONSECUTIVE_ERRORS = 5
    DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 10
    DEFAULT_ERROR_WINDOW_SECONDS = 60.0
    DEFAULT_COOLDOWN_SECONDS = 300.0
    DEFAULT_MAX_WORKERS = 4  # 기본 스레드 풀 크기
    
    def __init__(
        self,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        max_consecutive_timeouts: int = DEFAULT_MAX_CONSECUTIVE_TIMEOUTS,
        error_window_seconds: float = DEFAULT_ERROR_WINDOW_SECONDS,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        max_workers: int | None = None,
        auto_scale_workers: bool = True,
    ) -> None:
        """
        스케줄러 초기화
        
        Args:
            max_consecutive_errors: 연속 에러 시 비활성화 임계치
            max_consecutive_timeouts: 연속 타임아웃 시 비활성화 임계치
            error_window_seconds: 에러 카운트 윈도우 (초)
            cooldown_seconds: 비활성화 후 재활성화 대기 시간 (초)
            max_workers: ThreadPoolExecutor 최대 워커 수 (None이면 자동 계산)
            auto_scale_workers: 모듈 수에 따라 자동으로 워커 수 조정 (기본 True)
        """
        self._modules: dict[str, ModuleContext] = {}
        self._lock = threading.RLock()
        self._auto_scale_workers = auto_scale_workers
        
        # max_workers 결정: 설정값 또는 자동 계산
        if max_workers is None:
            if auto_scale_workers:
                # 초기에는 기본값 사용, 모듈 등록 시 조정
                max_workers = self.DEFAULT_MAX_WORKERS
            else:
                max_workers = self.DEFAULT_MAX_WORKERS
        
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="module_exec"
        )
        self._max_workers = max_workers
        
        self.max_consecutive_errors = max_consecutive_errors
        self.max_consecutive_timeouts = max_consecutive_timeouts
        self.error_window_seconds = error_window_seconds
        self.cooldown_seconds = cooldown_seconds
        
        # 비활성화된 모듈의 쿨다운 타임스탬프
        self._disabled_until: dict[str, float] = {}
        
        # 모듈 상태 변경 콜백
        self._on_module_disabled: Callable[[str, str], None] | None = None
        self._on_module_enabled: Callable[[str], None] | None = None
    
    def register(self, module: ModuleBase) -> None:
        """
        모듈을 등록합니다.
        
        Args:
            module: 등록할 모듈
        
        Raises:
            ValueError: 이미 등록된 모듈 이름인 경우
        """
        with self._lock:
            if module.name in self._modules:
                raise ValueError(f"이미 등록된 모듈입니다: {module.name}")
            
            context = ModuleContext(module)
            self._modules[module.name] = context
            
            logger.info(
                f"모듈 등록: {module.name}",
                module_name=module.name,
                priority=module.priority,
                enabled=module.enabled,
            )
            
            # 모듈 수에 따라 스레드 풀 크기 조정 (선택적)
            self._adjust_thread_pool_size()
    
    def unregister(self, name: str) -> bool:
        """
        모듈을 등록 해제합니다.
        
        Args:
            name: 모듈 이름
        
        Returns:
            성공 여부
        """
        with self._lock:
            if name not in self._modules:
                return False
            
            del self._modules[name]
            self._disabled_until.pop(name, None)
            
            logger.info(f"모듈 등록 해제: {name}", module_name=name)
            return True
    
    def get_module(self, name: str) -> ModuleContext | None:
        """모듈 컨텍스트를 반환합니다."""
        return self._modules.get(name)
    
    def get_all_modules(self) -> list[ModuleContext]:
        """모든 모듈 컨텍스트를 반환합니다."""
        with self._lock:
            return list(self._modules.values())
    
    def get_execution_order(self) -> list[ModuleContext]:
        """
        실행 순서에 따라 정렬된 활성 모듈 목록을 반환합니다.
        
        Returns:
            priority 오름차순으로 정렬된 enabled=True인 모듈 목록
        """
        with self._lock:
            now = time.time()
            
            # 쿨다운이 끝난 모듈 재활성화
            for name, until_ts in list(self._disabled_until.items()):
                if now >= until_ts:
                    ctx = self._modules.get(name)
                    if ctx:
                        ctx.module.enabled = True
                        ctx.reset_counters()
                        del self._disabled_until[name]
                        logger.info(
                            f"모듈 쿨다운 종료, 재활성화: {name}",
                            module_name=name,
                        )
                        if self._on_module_enabled:
                            self._on_module_enabled(name)
            
            # enabled=True인 모듈만 필터링하고 priority로 정렬
            active_modules = [
                ctx for ctx in self._modules.values()
                if ctx.module.enabled
            ]
            active_modules.sort(key=lambda ctx: ctx.module.priority)
            
            return active_modules
    
    def execute_with_timeout(
        self,
        context: ModuleContext,
        frame: FrameType,
        metadata: MetadataType,
    ) -> ProcessResult | None:
        """
        타임아웃을 적용하여 모듈을 실행합니다.
        
        Args:
            context: 모듈 컨텍스트
            frame: 입력 프레임
            metadata: 메타데이터
        
        Returns:
            처리 결과 또는 None (타임아웃/에러 시)
        """
        module = context.module
        timeout_sec = module.timeout_ms / 1000.0
        
        start_time = time.perf_counter()
        
        try:
            # ThreadPoolExecutor로 타임아웃 적용
            future = self._executor.submit(
                module.process_frame, frame, metadata
            )
            result = future.result(timeout=timeout_sec)
            
            # 성공 기록
            latency_ms = (time.perf_counter() - start_time) * 1000
            event_count = len(result[1]) if result else 0
            context.record_success(latency_ms, event_count)
            
            return result
            
        except FuturesTimeoutError:
            # 타임아웃 처리
            latency_ms = (time.perf_counter() - start_time) * 1000
            context.record_timeout()
            
            logger.warning(
                f"모듈 타임아웃: {module.name} ({latency_ms:.1f}ms > {module.timeout_ms}ms)",
                module_name=module.name,
                latency_ms=latency_ms,
                timeout_ms=module.timeout_ms,
                timeout_count=context.timeout_count,
            )
            
            self._check_disable(context, "timeout")
            return None
            
        except Exception as e:
            # 에러 처리
            latency_ms = (time.perf_counter() - start_time) * 1000
            context.record_error()
            
            logger.error(
                f"모듈 실행 오류: {module.name} - {e}",
                module_name=module.name,
                error=str(e),
                error_count=context.error_count,
            )
            
            self._check_disable(context, f"error: {e}")
            return None
    
    def _check_disable(self, context: ModuleContext, reason: str) -> None:
        """
        모듈 비활성화 조건을 확인하고 필요시 비활성화합니다.
        
        Args:
            context: 모듈 컨텍스트
            reason: 비활성화 사유
        """
        if context.should_disable(
            self.max_consecutive_errors,
            self.max_consecutive_timeouts,
        ):
            module_name = context.module.name
            context.module.enabled = False
            self._disabled_until[module_name] = time.time() + self.cooldown_seconds
            
            logger.warning(
                f"모듈 자동 비활성화: {module_name} (사유: {reason}, "
                f"쿨다운: {self.cooldown_seconds}초)",
                module_name=module_name,
                reason=reason,
                cooldown_seconds=self.cooldown_seconds,
            )
            
            if self._on_module_disabled:
                self._on_module_disabled(module_name, reason)
    
    def enable_module(self, name: str) -> bool:
        """
        모듈을 수동으로 활성화합니다.
        
        Args:
            name: 모듈 이름
        
        Returns:
            성공 여부
        """
        with self._lock:
            ctx = self._modules.get(name)
            if not ctx:
                return False
            
            ctx.module.enabled = True
            ctx.reset_counters()
            self._disabled_until.pop(name, None)
            
            logger.info(f"모듈 수동 활성화: {name}", module_name=name)
            return True
    
    def disable_module(self, name: str) -> bool:
        """
        모듈을 수동으로 비활성화합니다.
        
        Args:
            name: 모듈 이름
        
        Returns:
            성공 여부
        """
        with self._lock:
            ctx = self._modules.get(name)
            if not ctx:
                return False
            
            ctx.module.enabled = False
            self._disabled_until.pop(name, None)  # 쿨다운 제거
            
            logger.info(f"모듈 수동 비활성화: {name}", module_name=name)
            return True
    
    def set_on_module_disabled(
        self,
        callback: Callable[[str, str], None] | None,
    ) -> None:
        """모듈 비활성화 콜백을 설정합니다."""
        self._on_module_disabled = callback
    
    def set_on_module_enabled(
        self,
        callback: Callable[[str], None] | None,
    ) -> None:
        """모듈 활성화 콜백을 설정합니다."""
        self._on_module_enabled = callback
    
    def get_stats(self) -> dict[str, dict[str, Any]]:
        """
        모든 모듈의 통계를 반환합니다.
        
        Returns:
            {module_name: stats_dict} 형태의 딕셔너리
        """
        with self._lock:
            return {
                name: ctx.to_dict()
                for name, ctx in self._modules.items()
            }
    
    def _adjust_thread_pool_size(self) -> None:
        """
        모듈 수에 따라 스레드 풀 크기를 동적으로 조정합니다.
        
        auto_scale_workers가 True이면 모듈 수에 따라 워커 수를 계산하고,
        필요시 ThreadPoolExecutor를 재생성합니다.
        """
        if not self._auto_scale_workers:
            return
        
        active_count = len([m for m in self._modules.values() if m.module.enabled])
        
        # 모듈 수의 1.5배, 최소 2, 최대 16
        suggested_workers = max(2, min(16, int(active_count * 1.5)))
        
        if suggested_workers != self._max_workers:
            logger.info(
                "스레드 풀 크기 조정",
                active_modules=active_count,
                old_max_workers=self._max_workers,
                new_max_workers=suggested_workers,
            )
            
            # 기존 executor 종료
            old_executor = self._executor
            old_executor.shutdown(wait=False, cancel_futures=True)
            
            # 새 executor 생성
            self._max_workers = suggested_workers
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="module_exec"
            )
            
            logger.info("스레드 풀 크기 조정 완료")
        elif active_count > self._max_workers * 2:
            logger.warning(
                "활성 모듈 수가 스레드 풀 크기의 2배를 초과",
                active_modules=active_count,
                max_workers=self._max_workers,
                suggestion="max_workers를 늘리거나 모듈을 비활성화하세요",
            )
    
    def shutdown(self) -> None:
        """스케줄러를 종료합니다."""
        self._executor.shutdown(wait=True, cancel_futures=True)
        logger.info("모듈 스케줄러 종료")

