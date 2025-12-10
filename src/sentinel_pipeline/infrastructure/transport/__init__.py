# -*- coding: utf-8 -*-
"""
Transport Infrastructure 패키지.

이벤트 전송 클라이언트를 담당합니다:
- HTTP 클라이언트
- WebSocket 클라이언트
"""

from sentinel_pipeline.infrastructure.transport.http_client import HttpEventClient
from sentinel_pipeline.infrastructure.transport.ws_client import WebSocketEventClient

__all__ = [
    "HttpEventClient",
    "WebSocketEventClient",
]

