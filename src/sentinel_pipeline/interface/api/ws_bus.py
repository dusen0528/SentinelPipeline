"""
WebSocket 브로드캐스트 버스 (스켈레톤)

stream_update / module_stats / event 세 가지 메시지 타입을 기본으로 지원합니다.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Set

from fastapi import WebSocket

from sentinel_pipeline.common.logging import get_logger

logger = get_logger(__name__)


_connections: Set[WebSocket] = set()
_lock = asyncio.Lock()


async def register(ws: WebSocket) -> None:
    await ws.accept()
    async with _lock:
        _connections.add(ws)
    logger.info("WS 연결 등록", connections=len(_connections))


async def unregister(ws: WebSocket) -> None:
    async with _lock:
        if ws in _connections:
            _connections.remove(ws)
    logger.info("WS 연결 해제", connections=len(_connections))


async def _broadcast(message: Dict[str, Any]) -> None:
    async with _lock:
        conns = list(_connections)
    to_remove: list[WebSocket] = []
    for ws in conns:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.warning("WS 전송 실패, 연결 제거", error=str(e))
            to_remove.append(ws)
    if to_remove:
        async with _lock:
            for ws in to_remove:
                _connections.discard(ws)


async def publish_stream_update(payload: Dict[str, Any]) -> None:
    await _broadcast({"type": "stream_update", **payload})


async def publish_module_stats(payload: Dict[str, Any]) -> None:
    await _broadcast({"type": "module_stats", **payload})


async def publish_event(payload: Dict[str, Any]) -> None:
    await _broadcast({"type": "event", **payload})

