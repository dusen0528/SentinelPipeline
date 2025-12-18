"""
스트림 제어 API
"""

from __future__ import annotations

from typing import Any
import time
from urllib.parse import urlparse
import uuid

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.common.errors import ErrorCode, StreamError
from sentinel_pipeline.domain.models.stream import StreamStatus, StreamConfig
from sentinel_pipeline.interface.api.dependencies import get_stream_manager

router = APIRouter(prefix="/api/streams", tags=["streams"])

WAIT_TIMEOUT_SECONDS = 10.0
WAIT_POLL_SECONDS = 0.2


# === DTO 정의 ===


class StreamSummary(BaseModel):
    stream_id: str
    status: str
    fps: float | None = None
    last_frame_ts: float | None = None
    error_count: int | None = None


class StreamListResponse(BaseModel):
    success: bool = True
    data: list[StreamSummary]


class StreamDetailResponse(BaseModel):
    success: bool = True
    data: dict[str, Any]


class StreamControlResponse(BaseModel):
    success: bool = True
    data: dict[str, Any]


class StreamStartRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP URL")
    max_fps: int | None = Field(None, description="최대 FPS")
    downscale: float | None = Field(None, description="프레임 축소 비율")
    output_url: str | None = Field(None, description="출력 URL (선택)")

    model_config = {"extra": "forbid"}


# === 유틸 ===


def _state_to_summary(state: Any) -> StreamSummary:
    return StreamSummary(
        stream_id=state.stream_id,
        status=state.status.value,
        fps=state.stats.fps,
        last_frame_ts=state.stats.last_frame_ts,
        error_count=state.stats.error_count,
    )


def _state_to_dict(state: Any) -> dict[str, Any]:
    return {
        "stream_id": state.stream_id,
        "status": state.status.value,
        "config": state.config.to_dict(),
        "stats": state.stats.to_dict(),
        "last_error": state.last_error,
        "retry_count": state.retry_count,
        "next_retry_ts": state.next_retry_ts,
        "is_active": state.is_active,
        "is_healthy": state.is_healthy,
    }


def _validate_rtsp_url(url: str) -> str:
    trimmed = (url or "").strip()
    parsed = urlparse(trimmed)
    if not parsed.scheme.startswith("rtsp"):
        raise StreamError(ErrorCode.CONFIG_INVALID, "RTSP URL 형식이 아닙니다", stream_id="auto")
    if not parsed.hostname:
        raise StreamError(ErrorCode.CONFIG_INVALID, "RTSP URL에 호스트가 없습니다", stream_id="auto")
    if not parsed.path or parsed.path == "/":
        raise StreamError(ErrorCode.CONFIG_INVALID, "RTSP URL 경로가 비어 있습니다", stream_id="auto")
    return trimmed

# === 라우트 ===

@router.get("", response_model=StreamListResponse)
async def list_streams(manager: StreamManager = Depends(get_stream_manager)) -> StreamListResponse:
    """모든 스트림 상태를 조회합니다."""
    states = manager.get_all_streams()
    return StreamListResponse(data=[_state_to_summary(s) for s in states])


@router.get("/{stream_id}", response_model=StreamDetailResponse)
async def get_stream(
    stream_id: str,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamDetailResponse:
    """특정 스트림 상태를 조회합니다."""
    state = manager.get_stream_state(stream_id)
    if not state:
        raise StreamError(
            code=ErrorCode.STREAM_NOT_FOUND,
            message=f"스트림을 찾을 수 없습니다: {stream_id}",
            stream_id=stream_id,
        )
    return StreamDetailResponse(data=_state_to_dict(state))


@router.post("/{stream_id}/start", response_model=StreamControlResponse)
async def start_stream(
    stream_id: str,
    payload: StreamStartRequest,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    스트림을 시작하고 RUNNING 상태가 될 때까지 대기합니다.
    
    10초 내에 RUNNING 상태가 되면 200 응답, 
    ERROR 상태면 502, 타임아웃이면 503을 반환합니다.
    """
    # RTSP URL 검증
    validated_url = _validate_rtsp_url(payload.rtsp_url)
    if payload.output_url:
        _validate_rtsp_url(payload.output_url)
    
    try:
        final_state = manager.start_stream(
            stream_id=stream_id,
            rtsp_url=validated_url,
            max_fps=payload.max_fps,
            downscale=payload.downscale,
            output_url=payload.output_url,
            wait=True,
        )
        return StreamControlResponse(data=_state_to_dict(final_state))
    except (StreamError, ValueError) as e:
        # ValueError from StreamConfig validation
        raise StreamError(
            e.code if isinstance(e, StreamError) else ErrorCode.CONFIG_INVALID,
            str(e),
            stream_id=stream_id,
        ) from e

@router.post("/{stream_id}/stop", response_model=StreamControlResponse)
async def stop_stream(
    stream_id: str,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """스트림을 중지합니다."""
    manager.stop_stream(stream_id)
    state = manager.get_stream_state(stream_id)
    data = _state_to_dict(state) if state else {"stream_id": stream_id, "status": "STOPPED"}
    return StreamControlResponse(data=data)


@router.post("/{stream_id}/restart", response_model=StreamControlResponse)
async def restart_stream(
    stream_id: str,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """스트림을 재시작합니다."""
    state = manager.restart_stream(stream_id, wait=True)
    return StreamControlResponse(data=_state_to_dict(state))


@router.delete("/{stream_id}", response_model=StreamControlResponse)
async def delete_stream(
    stream_id: str,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    스트림을 삭제합니다.
    
    스트림이 실행 중이면 먼저 중지한 후 삭제합니다.
    """
    manager.delete_stream(stream_id)
    return StreamControlResponse(data={
        "stream_id": stream_id,
        "status": "DELETED",
        "message": "스트림이 삭제되었습니다",
    })





@router.get("/by-input", response_model=StreamControlResponse)
async def get_output_url_by_input(
    input_url: str = Query(..., description="입력 RTSP URL"),
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    입력 RTSP URL로 스트림을 찾거나 생성하고, 실행 상태로 만듭니다.
    
    - 기존 스트림이 있으면 재시작하거나 현재 상태 반환
    - 없으면 새로 생성하고 시작
    - 최종적으로 RUNNING 상태가 될 때까지 10초간 대기
    """
    # RTSP URL 형식 검증
    validated_input = _validate_rtsp_url(input_url)

    try:
        final_state = manager.get_or_create_by_input_url(validated_input)
        output_url = final_state.config.output_url or StreamConfig.generate_output_url(validated_input)
        
        data = {
            "input_url": validated_input,
            "output_url": output_url,
            **_state_to_dict(final_state),
        }
        return StreamControlResponse(data=data)
    except (StreamError, ValueError) as e:
        raise StreamError(
            e.code if isinstance(e, StreamError) else ErrorCode.STREAM_CONNECTION_FAILED,
            f"입력 URL '{input_url}'로 스트림을 생성/실행할 수 없습니다: {e}",
            stream_id="auto",
        ) from e
