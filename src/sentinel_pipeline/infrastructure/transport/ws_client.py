# -*- coding: utf-8 -*-
"""
WebSocket 이벤트 전송 클라이언트.

websockets를 사용하여 이벤트를 VMS 서버로 실시간 전송합니다.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from sentinel_pipeline.common.errors import ErrorCode, TransportError
from sentinel_pipeline.common.logging import get_logger

if TYPE_CHECKING:
    import websockets
    from websockets.legacy.client import WebSocketClientProtocol
    
    from sentinel_pipeline.domain.models.event import Event


class ConnectionState(str, Enum):
    """WebSocket 연결 상태."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"


class WebSocketEventClient:
    """
    WebSocket 이벤트 전송 클라이언트.
    
    VMS 서버로 이벤트를 WebSocket을 통해 실시간 전송합니다.
    자동 재연결을 지원합니다.
    """
    
    def __init__(
        self,
        url: str,
        reconnect_interval: float = 1.0,
        reconnect_max_interval: float = 30.0,
        reconnect_max_retries: int = 10,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
        headers: dict[str, str] | None = None,
    ):
        """
        WebSocketEventClient 초기화.
        
        Args:
            url: WebSocket URL (예: ws://vms.example.com/ws/events)
            reconnect_interval: 초기 재연결 간격 (초)
            reconnect_max_interval: 최대 재연결 간격 (초)
            reconnect_max_retries: 최대 재연결 시도 횟수
            ping_interval: Ping 간격 (초)
            ping_timeout: Ping 타임아웃 (초)
            headers: 추가 HTTP 헤더
        """
        self._url = url
        self._reconnect_interval = reconnect_interval
        self._reconnect_max_interval = reconnect_max_interval
        self._reconnect_max_retries = reconnect_max_retries
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._headers = headers or {}
        
        self._ws: WebSocketClientProtocol | None = None
        self._state = ConnectionState.DISCONNECTED
        self._running = False
        self._lock = threading.Lock()
        self._reconnect_count = 0
        
        # 콜백
        self._on_connected: Callable[[], None] | None = None
        self._on_disconnected: Callable[[str], None] | None = None
        self._on_message: Callable[[dict], None] | None = None
        
        # 통계
        self._sent_count = 0
        self._error_count = 0
        self._received_count = 0
        
        self._logger = get_logger(__name__)
    
    @property
    def state(self) -> ConnectionState:
        """연결 상태."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """연결 여부."""
        return self._state == ConnectionState.CONNECTED
    
    @property
    def sent_count(self) -> int:
        """전송된 메시지 수."""
        return self._sent_count
    
    @property
    def error_count(self) -> int:
        """오류 수."""
        return self._error_count
    
    def set_on_connected(self, callback: Callable[[], None]) -> None:
        """연결 완료 콜백 설정."""
        self._on_connected = callback
    
    def set_on_disconnected(self, callback: Callable[[str], None]) -> None:
        """연결 종료 콜백 설정."""
        self._on_disconnected = callback
    
    def set_on_message(self, callback: Callable[[dict], None]) -> None:
        """메시지 수신 콜백 설정."""
        self._on_message = callback
    
    async def connect(self) -> bool:
        """
        WebSocket 서버에 연결합니다.
        
        Returns:
            연결 성공 여부
        """
        import websockets
        
        if self._state == ConnectionState.CONNECTED:
            return True
        
        self._state = ConnectionState.CONNECTING
        self._logger.info("WebSocket 연결 시도", url=self._url)
        
        try:
            self._ws = await websockets.connect(
                self._url,
                extra_headers=self._headers,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            )
            
            self._state = ConnectionState.CONNECTED
            self._reconnect_count = 0
            self._running = True
            
            self._logger.info("WebSocket 연결 성공")
            
            if self._on_connected:
                self._on_connected()
            
            return True
            
        except Exception as e:
            self._state = ConnectionState.DISCONNECTED
            self._error_count += 1
            self._logger.error("WebSocket 연결 실패", error=str(e))
            return False
    
    async def disconnect(self) -> None:
        """WebSocket 연결을 종료합니다."""
        self._running = False
        self._state = ConnectionState.CLOSED
        
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        
        self._logger.info("WebSocket 연결 종료")
    
    async def send_event(self, event: Event) -> bool:
        """
        이벤트를 전송합니다.
        
        Args:
            event: 전송할 이벤트
            
        Returns:
            성공 여부
        """
        return await self.send_events([event])
    
    async def send_events(self, events: list[Event]) -> bool:
        """
        이벤트 배치를 전송합니다.
        
        Args:
            events: 전송할 이벤트 리스트
            
        Returns:
            성공 여부
        """
        if not self.is_connected or self._ws is None:
            self._logger.warning("WebSocket이 연결되어 있지 않음")
            return False
        
        if not events:
            return True
        
        try:
            # 이벤트를 JSON으로 변환
            payload = {
                "type": "events",
                "ts": time.time(),
                "events": [event.to_dict() for event in events],
            }
            
            await self._ws.send(json.dumps(payload))
            self._sent_count += len(events)
            
            return True
            
        except Exception as e:
            self._error_count += 1
            self._logger.error("이벤트 전송 실패", error=str(e))
            
            # 연결 끊김 처리
            if self._running:
                self._state = ConnectionState.RECONNECTING
                asyncio.create_task(self._reconnect())
            
            return False
    
    async def send_message(self, message: dict) -> bool:
        """
        일반 메시지를 전송합니다.
        
        Args:
            message: 전송할 메시지 딕셔너리
            
        Returns:
            성공 여부
        """
        if not self.is_connected or self._ws is None:
            return False
        
        try:
            await self._ws.send(json.dumps(message))
            return True
        except Exception as e:
            self._error_count += 1
            self._logger.error("메시지 전송 실패", error=str(e))
            return False
    
    async def receive_loop(self) -> None:
        """
        메시지 수신 루프.
        
        백그라운드에서 메시지를 수신하고 콜백을 호출합니다.
        """
        import websockets
        
        while self._running and self.is_connected and self._ws is not None:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=1.0,
                )
                
                self._received_count += 1
                
                # JSON 파싱
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    data = {"raw": message}
                
                # 콜백 호출
                if self._on_message:
                    self._on_message(data)
                    
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed as e:
                self._logger.warning("WebSocket 연결 끊김", reason=str(e))
                if self._running:
                    self._state = ConnectionState.RECONNECTING
                    await self._reconnect()
                break
            except Exception as e:
                self._logger.error("메시지 수신 중 오류", error=str(e))
    
    async def _reconnect(self) -> None:
        """재연결을 시도합니다."""
        import websockets
        
        while self._running and self._reconnect_count < self._reconnect_max_retries:
            self._reconnect_count += 1
            
            # 지수 백오프 계산
            delay = min(
                self._reconnect_interval * (2 ** (self._reconnect_count - 1)),
                self._reconnect_max_interval,
            )
            
            self._logger.info(
                "WebSocket 재연결 대기",
                attempt=self._reconnect_count,
                delay=f"{delay:.1f}s",
            )
            
            await asyncio.sleep(delay)
            
            if not self._running:
                break
            
            try:
                self._ws = await websockets.connect(
                    self._url,
                    extra_headers=self._headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                
                self._state = ConnectionState.CONNECTED
                self._reconnect_count = 0
                
                self._logger.info("WebSocket 재연결 성공")
                
                if self._on_connected:
                    self._on_connected()
                
                return
                
            except Exception as e:
                self._logger.warning("WebSocket 재연결 실패", error=str(e))
        
        # 최대 재시도 초과
        self._state = ConnectionState.DISCONNECTED
        self._logger.error("WebSocket 재연결 최대 횟수 초과")
        
        if self._on_disconnected:
            self._on_disconnected("최대 재연결 횟수 초과")
    
    async def run(self) -> None:
        """
        클라이언트를 실행합니다.
        
        연결 및 메시지 수신 루프를 시작합니다.
        """
        self._running = True
        
        if not await self.connect():
            return
        
        await self.receive_loop()
    
    def get_stats(self) -> dict:
        """통계 정보 반환."""
        return {
            "url": self._url,
            "state": self._state.value,
            "sent_count": self._sent_count,
            "received_count": self._received_count,
            "error_count": self._error_count,
            "reconnect_count": self._reconnect_count,
        }


# 비동기 실행 헬퍼
def run_ws_client(client: WebSocketEventClient) -> None:
    """
    WebSocket 클라이언트를 별도 스레드에서 실행합니다.
    
    Args:
        client: WebSocketEventClient 인스턴스
    """
    async def _run():
        await client.run()
    
    asyncio.run(_run())


def create_ws_client_thread(
    url: str,
    on_connected: Callable[[], None] | None = None,
    on_disconnected: Callable[[str], None] | None = None,
    on_message: Callable[[dict], None] | None = None,
) -> tuple[WebSocketEventClient, threading.Thread]:
    """
    WebSocket 클라이언트와 실행 스레드를 생성합니다.
    
    Args:
        url: WebSocket URL
        on_connected: 연결 완료 콜백
        on_disconnected: 연결 종료 콜백
        on_message: 메시지 수신 콜백
        
    Returns:
        (클라이언트, 스레드) 튜플
    """
    client = WebSocketEventClient(url)
    
    if on_connected:
        client.set_on_connected(on_connected)
    if on_disconnected:
        client.set_on_disconnected(on_disconnected)
    if on_message:
        client.set_on_message(on_message)
    
    thread = threading.Thread(
        target=run_ws_client,
        args=(client,),
        daemon=True,
    )
    
    return client, thread

