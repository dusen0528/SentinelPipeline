# -*- coding: utf-8 -*-
"""
HTTP 이벤트 전송 클라이언트.

httpx를 사용하여 이벤트를 VMS 서버로 전송합니다.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sentinel_pipeline.common.errors import ErrorCode, TransportError
from sentinel_pipeline.common.logging import get_logger

if TYPE_CHECKING:
    from sentinel_pipeline.domain.models.event import Event


class HttpEventClient:
    """
    HTTP 이벤트 전송 클라이언트.
    
    VMS 서버로 이벤트를 HTTP POST로 전송합니다.
    배치 전송 및 재시도를 지원합니다.
    """
    
    def __init__(
        self,
        base_url: str,
        endpoint: str = "/api/events",
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        headers: dict[str, str] | None = None,
    ):
        """
        HttpEventClient 초기화.
        
        Args:
            base_url: VMS 서버 기본 URL (예: http://vms.example.com)
            endpoint: 이벤트 전송 엔드포인트 (기본 /api/events)
            timeout_seconds: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            headers: 추가 HTTP 헤더
        """
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._headers = headers or {}
        
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._running = False
        self._lock = threading.Lock()
        
        # 통계
        self._sent_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0
        
        self._logger = get_logger(__name__)
    
    @property
    def url(self) -> str:
        """전체 URL."""
        return f"{self._base_url}{self._endpoint}"
    
    @property
    def sent_count(self) -> int:
        """전송된 이벤트 수."""
        return self._sent_count
    
    @property
    def error_count(self) -> int:
        """오류 수."""
        return self._error_count
    
    @property
    def avg_latency_ms(self) -> float:
        """평균 전송 지연 시간 (밀리초)."""
        if self._sent_count == 0:
            return 0.0
        return self._total_latency_ms / self._sent_count
    
    def start(self) -> None:
        """동기 클라이언트를 시작합니다."""
        with self._lock:
            if self._running:
                return
            
            self._sync_client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                headers=self._headers,
            )
            self._running = True
            self._logger.info("HTTP 클라이언트 시작", url=self.url)
    
    def stop(self) -> None:
        """클라이언트를 중지합니다."""
        with self._lock:
            self._running = False
            
            if self._sync_client is not None:
                self._sync_client.close()
                self._sync_client = None
            
            self._logger.info("HTTP 클라이언트 중지")
    
    async def start_async(self) -> None:
        """비동기 클라이언트를 시작합니다."""
        with self._lock:
            if self._client is not None:
                return
            
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                headers=self._headers,
            )
            self._running = True
            self._logger.info("비동기 HTTP 클라이언트 시작", url=self.url)
    
    async def stop_async(self) -> None:
        """비동기 클라이언트를 중지합니다."""
        with self._lock:
            self._running = False
            
            if self._client is not None:
                await self._client.aclose()
                self._client = None
            
            self._logger.info("비동기 HTTP 클라이언트 중지")
    
    def send_event(self, event: Event) -> bool:
        """
        이벤트를 동기적으로 전송합니다.
        
        Args:
            event: 전송할 이벤트
            
        Returns:
            성공 여부
        """
        return self.send_events([event])
    
    def send_events(self, events: list[Event]) -> bool:
        """
        이벤트 배치를 동기적으로 전송합니다.
        
        Args:
            events: 전송할 이벤트 리스트
            
        Returns:
            성공 여부
        """
        if not self._running or self._sync_client is None:
            self._logger.warning("HTTP 클라이언트가 실행 중이 아님")
            return False
        
        if not events:
            return True
        
        start_time = time.perf_counter()
        
        try:
            # 이벤트를 JSON으로 변환
            payload = [event.to_dict() for event in events]
            
            # 재시도 포함 전송
            response = self._send_with_retry(payload)
            
            if response.status_code >= 400:
                self._error_count += 1
                self._logger.warning(
                    "이벤트 전송 실패",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return False
            
            # 통계 업데이트
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._sent_count += len(events)
            self._total_latency_ms += latency_ms
            
            self._logger.debug(
                "이벤트 전송 완료",
                count=len(events),
                latency_ms=f"{latency_ms:.1f}",
            )
            return True
            
        except Exception as e:
            self._error_count += 1
            self._logger.error("이벤트 전송 중 오류", error=str(e))
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    def _send_with_retry(self, payload: list[dict]) -> httpx.Response:
        """재시도 포함 전송."""
        return self._sync_client.post(  # type: ignore
            self._endpoint,
            json=payload,
        )
    
    async def send_event_async(self, event: Event) -> bool:
        """
        이벤트를 비동기적으로 전송합니다.
        
        Args:
            event: 전송할 이벤트
            
        Returns:
            성공 여부
        """
        return await self.send_events_async([event])
    
    async def send_events_async(self, events: list[Event]) -> bool:
        """
        이벤트 배치를 비동기적으로 전송합니다.
        
        Args:
            events: 전송할 이벤트 리스트
            
        Returns:
            성공 여부
        """
        if not self._running or self._client is None:
            self._logger.warning("비동기 HTTP 클라이언트가 실행 중이 아님")
            return False
        
        if not events:
            return True
        
        start_time = time.perf_counter()
        
        try:
            # 이벤트를 JSON으로 변환
            payload = [event.to_dict() for event in events]
            
            # 전송
            response = await self._send_with_retry_async(payload)
            
            if response.status_code >= 400:
                self._error_count += 1
                self._logger.warning(
                    "이벤트 전송 실패",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return False
            
            # 통계 업데이트
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._sent_count += len(events)
            self._total_latency_ms += latency_ms
            
            return True
            
        except Exception as e:
            self._error_count += 1
            self._logger.error("이벤트 전송 중 오류", error=str(e))
            return False
    
    async def _send_with_retry_async(
        self,
        payload: list[dict],
        retries: int = 3,
    ) -> httpx.Response:
        """재시도 포함 비동기 전송."""
        last_exception: Exception | None = None
        
        for attempt in range(retries):
            try:
                return await self._client.post(  # type: ignore
                    self._endpoint,
                    json=payload,
                )
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < retries - 1:
                    delay = 0.5 * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        raise last_exception  # type: ignore
    
    def get_stats(self) -> dict:
        """통계 정보 반환."""
        return {
            "url": self.url,
            "running": self._running,
            "sent_count": self._sent_count,
            "error_count": self._error_count,
            "avg_latency_ms": self.avg_latency_ms,
        }
    
    def __enter__(self) -> HttpEventClient:
        """컨텍스트 매니저 진입."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료."""
        self.stop()


class HttpTransport:
    """
    HTTP 전송 래퍼 클래스.
    
    EventEmitter의 sync_transport로 사용할 수 있으며,
    리소스 정리를 위한 close() 메서드를 제공합니다.
    
    Example:
        transport = HttpTransport(base_url="http://vms.example.com")
        emitter.set_sync_transport(transport)
        # ... 사용 후
        transport.close()
    """
    
    def __init__(
        self,
        base_url: str,
        endpoint: str = "/api/events",
        headers: dict[str, str] | None = None,
    ):
        self._client = HttpEventClient(base_url, endpoint, headers=headers)
        self._client.start()
    
    def __call__(self, events: list[Event]) -> bool:
        """이벤트 전송 (EventEmitter 콜백으로 사용)."""
        return self._client.send_events(events)
    
    def close(self) -> None:
        """리소스 정리."""
        self._client.stop()
    
    def get_stats(self) -> dict:
        """통계 정보 반환."""
        return self._client.get_stats()
    
    def __enter__(self) -> HttpTransport:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# 동기 전송 함수 (EventEmitter에서 사용) - deprecated
def create_http_transport(
    base_url: str,
    endpoint: str = "/api/events",
    headers: dict[str, str] | None = None,
) -> tuple[Callable[[list[Event]], bool], Callable[[], None]]:
    """
    HTTP 전송 함수를 생성합니다.
    
    **주의**: 이 함수는 deprecated입니다. HttpTransport 클래스를 사용하세요.
    
    Args:
        base_url: VMS 서버 URL
        endpoint: 이벤트 엔드포인트
        headers: 추가 헤더
        
    Returns:
        (전송 함수, 종료 함수) 튜플
    """
    transport = HttpTransport(base_url, endpoint, headers=headers)
    return transport, transport.close

