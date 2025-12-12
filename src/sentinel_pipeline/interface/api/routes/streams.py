"""
스트림 제어 API
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.common.errors import ErrorCode, StreamError
from sentinel_pipeline.interface.api.dependencies import get_stream_manager

router = APIRouter(prefix="/api/streams", tags=["streams"])


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
    """스트림을 시작합니다."""
    try:
        state = manager.start_stream(
            stream_id=stream_id,
            rtsp_url=payload.rtsp_url,
            max_fps=payload.max_fps,
            downscale=payload.downscale,
            output_url=payload.output_url,
        )
    except ValueError as e:
        # StreamConfig __post_init__ 유효성 오류
        raise StreamError(
            ErrorCode.CONFIG_INVALID,
            f"스트림 설정이 올바르지 않습니다: {e}",
            stream_id=stream_id,
        ) from e
    return StreamControlResponse(data=_state_to_dict(state))


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
    state = manager.restart_stream(stream_id)
    return StreamControlResponse(data=_state_to_dict(state))

