from __future__ import annotations

import asyncio
from typing import Any

import base64
import os
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from sentinel_pipeline.interface.api.ws_bus import (
    publish_event,
    publish_module_stats,
    publish_stream_update,
    register,
    unregister,
)

router = APIRouter()


async def _auth_ws(ws: WebSocket) -> bool:
    """
    간단한 인증:
    1) ADMIN_WS_API_KEY 가 설정되어 있으면 X-API-Key로 검증
    2) 없으면 ADMIN_WS_USER/PASSWORD가 설정된 경우 Basic Auth 검증
    3) 둘 다 없으면 인증을 생략
    """
    # 1) API Key: query param token 또는 X-API-Key 헤더
    token = ws.query_params.get("token")
    api_key = os.getenv("ADMIN_WS_API_KEY")
    if api_key:
        hdr = ws.headers.get("x-api-key")
        if (hdr or token) != api_key:
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return False
        return True

    user = os.getenv("ADMIN_WS_USER")
    password = os.getenv("ADMIN_WS_PASSWORD")
    if user and password:
        auth_hdr = ws.headers.get("authorization")
        if not auth_hdr or not auth_hdr.lower().startswith("basic "):
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return False
        try:
            encoded = auth_hdr.split(" ", 1)[1]
            raw = base64.b64decode(encoded).decode("utf-8")
            u, p = raw.split(":", 1)
            if u != user or p != password:
                await ws.close(code=status.WS_1008_POLICY_VIOLATION)
                return False
        except Exception:
            await ws.close(code=status.WS_1008_POLICY_VIOLATION)
            return False
    return True


@router.websocket("/api/ws/admin")
async def admin_ws(ws: WebSocket) -> None:
    if not await _auth_ws(ws):
        return
    await register(ws)
    try:
        # 초기 핸드셰이크 메시지
        await ws.send_json({"type": "welcome", "message": "admin websocket connected"})
        while True:
            try:
                data: Any = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                # 단순 에코/헬스 체크
                if isinstance(data, str) and data.strip().lower() == "ping":
                    await ws.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # keep-alive
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        # 다른 예외는 무시하고 종료
        pass
    finally:
        await unregister(ws)

