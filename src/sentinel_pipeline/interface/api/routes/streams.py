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
from sentinel_pipeline.domain.models.stream import StreamStatus
from sentinel_pipeline.interface.api.dependencies import get_stream_manager

router = APIRouter(prefix="/api/streams", tags=["streams"])

MAX_CONCURRENT_STREAMS = 10
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


def _wait_for_running(manager: StreamManager, stream_id: str) -> Any:
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    while time.time() < deadline:
        state = manager.get_stream_state(stream_id)
        if state is None:
            raise StreamError(ErrorCode.STREAM_NOT_FOUND, "스트림을 찾을 수 없습니다", stream_id=stream_id)
        if state.status == StreamStatus.RUNNING:
            return state
        if state.status == StreamStatus.ERROR:
            raise StreamError(
                ErrorCode.STREAM_CONNECTION_FAILED,
                state.last_error or "스트림 연결 오류",
                stream_id=stream_id,
            )
        time.sleep(WAIT_POLL_SECONDS)
    raise StreamError(
        ErrorCode.TRANSPORT_TIMEOUT,
        "10초 내에 스트림이 RUNNING 상태가 되지 않았습니다",
        stream_id=stream_id,
    )





def _enforce_capacity(manager: StreamManager) -> JSONResponse | None:
    active = sum(1 for s in manager.get_all_streams() if s.is_active)
    if active >= MAX_CONCURRENT_STREAMS:
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "message": f"동시 스트림 한도({MAX_CONCURRENT_STREAMS}개)를 초과했습니다",
            },
        )
    return None


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


async def _start_and_wait_stream(
    manager: StreamManager,
    stream_id: str,
    rtsp_url: str,
    max_fps: int | None,
    downscale: float | None,
    output_url: str | None,
) -> dict[str, Any]:
    """Helper to start a stream and wait for it to run."""
    try:
        manager.start_stream(
            stream_id=stream_id,
            rtsp_url=rtsp_url,
            max_fps=max_fps,
            downscale=downscale,
            output_url=output_url,
        )
    except ValueError as e:
        raise StreamError(
            ErrorCode.CONFIG_INVALID,
            f"스트림 설정이 올바르지 않습니다: {e}",
            stream_id=stream_id,
        ) from e

    try:
        final_state = _wait_for_running(manager, stream_id)
        return _state_to_dict(final_state)
    except StreamError:
        raise


@router.post("/{stream_id}/start", response_model=StreamControlResponse)
async def start_stream(
    stream_id: str,
    payload: StreamStartRequest,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    스트림을 시작합니다.
    
    10초 내에 RUNNING 상태가 되면 200 응답, 
    ERROR 상태면 502, 타임아웃이면 503을 반환합니다.
    """
    # RTSP URL 검증
    validated_url = _validate_rtsp_url(payload.rtsp_url)
    if payload.output_url:
        _validate_rtsp_url(payload.output_url)
    
    # 동시 스트림 한도 확인
    capacity_check = _enforce_capacity(manager)
    if capacity_check:
        return capacity_check
    
    data = await _start_and_wait_stream(
        manager=manager,
        stream_id=stream_id,
        rtsp_url=validated_url,
        max_fps=payload.max_fps,
        downscale=payload.downscale,
        output_url=payload.output_url,
    )
    return StreamControlResponse(data=data)


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


def _generate_output_url(input_url: str) -> str:
    """입력 RTSP URL을 기반으로 블러 출력 URL 생성"""
    parsed = urlparse(input_url.strip())
    if not parsed.scheme.startswith("rtsp"):
        raise ValueError("RTSP URL이 아닙니다.")
    
    path = parsed.path or ""
    if not path:
        raise ValueError("RTSP URL에 경로가 없습니다.")
    
    if path.endswith("/"):
        path = path[:-1]
    # 마지막 세그먼트에 -blur 추가
    segments = path.split("/")
    last = segments[-1]
    if not last:
        raise ValueError("RTSP URL 경로가 비어 있습니다.")
    segments[-1] = f"{last}-blur"
    new_path = "/".join(segments)
    
    port_part = f":{parsed.port}" if parsed.port else ""
    return f"rtsp://{parsed.hostname}{port_part}{new_path}"


@router.get("/by-input", response_model=StreamControlResponse)
async def get_output_url_by_input(
    input_url: str = Query(..., description="입력 RTSP URL"),
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    입력 RTSP URL로 출력 RTSP URL 조회 및 자동 시작
    
    동작:
    1. 기존 스트림이 있으면:
       - STOPPED/ERROR 상태면 즉시 재시작 시도
       - RUNNING이면 즉시 반환
    2. 없으면 새로 생성하고 자동 시작
    3. 10초 내에 RUNNING이면 output RTSP 반환
    4. ERROR면 502, 타임아웃이면 503 반환
    
    검증:
    - 입력/출력 RTSP URL 형식 검증
    - 중복 생성 방지 (기존 스트림 재사용)
    - 동시 스트림 한도 초과 시 429 반환
    """
    # RTSP URL 형식 검증
    validated_input = _validate_rtsp_url(input_url)
    
    # 기존 스트림 찾기 (입력 URL로)
    existing_state = None
    all_streams = manager.get_all_streams()
    for state in all_streams:
        if state.config.rtsp_url.strip() == validated_input:
            existing_state = state
            break
    
    # 중복 생성 방지: 기존 스트림이 있으면 재사용
    if existing_state:
        stream_id = existing_state.stream_id
        status = existing_state.status.value
        
        # STOPPED/ERROR 상태면 즉시 재시작
        if status in ("STOPPED", "ERROR"):
            try:
                manager.restart_stream(stream_id)
            except Exception as e:
                raise StreamError(
                    ErrorCode.STREAM_CONNECTION_FAILED,
                    f"스트림 재시작 실패: {e}",
                    stream_id=stream_id,
                ) from e
        
        # 10초 대기 후 RUNNING 확인
        try:
            final_state = _wait_for_running(manager, stream_id)
            output_url = final_state.config.output_url or _generate_output_url(validated_input)
            return StreamControlResponse(data={
                "input_url": validated_input,
                "output_url": output_url,
                "stream_id": stream_id,
                "status": final_state.status.value,
                "config": final_state.config.to_dict(),
                "stats": final_state.stats.to_dict(),
            })
        except StreamError:
            raise
    
    # 기존 스트림이 없으면 새로 생성
    # 동시 스트림 한도 확인
    capacity_check = _enforce_capacity(manager)
    if capacity_check:
        return capacity_check
    
    try:
        # stream_id 자동 생성
        stream_id = f"auto-{str(uuid.uuid4())[:8]}"
        
        # output_url 자동 생성
        generated_output = _generate_output_url(validated_input)
        
        # 스트림 시작 및 대기
        final_state_dict = await _start_and_wait_stream(
            manager=manager,
            stream_id=stream_id,
            rtsp_url=validated_input,
            max_fps=None,
            downscale=None,
            output_url=generated_output,
        )
        
        return StreamControlResponse(data={
            "input_url": validated_input,
            "output_url": generated_output,
            **final_state_dict,
        })
    except (StreamError, ValueError) as e:
        raise StreamError(
            code=ErrorCode.STREAM_CONNECTION_FAILED,
            message=f"입력 URL '{input_url}'로 스트림을 생성할 수 없습니다: {e}",
            stream_id="auto",
        ) from e
