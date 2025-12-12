from __future__ import annotations

import base64
import os
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from sentinel_pipeline.interface.api.ws_bus import (
    publish_event,
    publish_module_stats,
    publish_stream_update,
)

router = APIRouter(prefix="/admin/debug", tags=["admin-debug"])


def _auth_http(request: Request) -> None:
    """
    간단한 인증:
    1) ADMIN_WS_API_KEY 설정 시 X-API-Key 헤더 또는 token 쿼리로 검증
    2) 없으면 ADMIN_WS_USER/PASSWORD가 설정된 경우 Basic Auth 검증
    3) 둘 다 없으면 인증 생략
    """
    api_key = os.getenv("ADMIN_WS_API_KEY")
    if api_key:
        hdr = request.headers.get("x-api-key")
        token = request.query_params.get("token")
        if (hdr or token) != api_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")
        return

    user = os.getenv("ADMIN_WS_USER")
    password = os.getenv("ADMIN_WS_PASSWORD")
    if user and password:
        auth_hdr = request.headers.get("authorization")
        if not auth_hdr or not auth_hdr.lower().startswith("basic "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="basic auth required")
        try:
            encoded = auth_hdr.split(" ", 1)[1]
            raw = base64.b64decode(encoded).decode("utf-8")
            u, p = raw.split(":", 1)
            if u != user or p != password:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid basic header")


class BroadcastRequest(BaseModel):
    kind: Literal["stream_update", "module_stats", "event"]
    payload: dict


@router.post("/broadcast")
async def admin_debug_broadcast(
    body: BroadcastRequest,
    _: None = Depends(_auth_http),
) -> dict:
    """디버그용: HTTP로 WS 브로드캐스트를 트리거합니다."""
    if body.kind == "stream_update":
        await publish_stream_update(body.payload)
    elif body.kind == "module_stats":
        await publish_module_stats(body.payload)
    elif body.kind == "event":
        await publish_event(body.payload)
    return {"success": True}
