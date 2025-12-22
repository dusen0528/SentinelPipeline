"""
스트림 제어 API
"""

from __future__ import annotations

from typing import Any
import time
from urllib.parse import urlparse
import uuid

from fastapi import APIRouter, Body, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from sentinel_pipeline.application.stream.manager import StreamManager
from sentinel_pipeline.common.errors import ErrorCode, StreamError
from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.domain.models.stream import StreamStatus, StreamConfig
from sentinel_pipeline.interface.api.dependencies import get_stream_manager

logger = get_logger(__name__)

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


class StreamStartByInputRequest(BaseModel):
    """RTSP URL로 스트림 시작 시 사용 (rtsp_url은 선택, 쿼리 파라미터 input_url 사용)"""
    rtsp_url: str | None = Field(None, description="RTSP URL (선택, 쿼리 파라미터 input_url 사용 가능)")
    max_fps: int | None = Field(None, description="최대 FPS")
    downscale: float | None = Field(None, description="프레임 축소 비율")
    output_url: str | None = Field(None, description="출력 URL (선택)")

    model_config = {"extra": "forbid"}


class StreamRegisterRequest(BaseModel):
    rtsp_url: str = Field(..., description="RTSP URL")

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

# === 개선된 유틸 함수 ===

def _disconnect_stream_internal(manager: StreamManager, stream_id: str) -> None:
    """
    스트림을 중지하고 모든 리소스를 정리한 후 삭제합니다 (내부 헬퍼).
    
    """
    # 존재 여부 확인
    if not manager.get_stream_state(stream_id):
        # 이미 없으면 그냥 반환 (에러 발생 안 함)
        return

    # 중지 및 삭제 (예외 처리 포함)
    try:
        manager.stop_stream(stream_id, force=False)
        manager.delete_stream(stream_id)
    except Exception as e:
        # 이미 삭제되었거나 하는 경우에도 무시
        # (리소스는 정리되었을 가능성이 높음)
        logger.warning(f"disconnect 중 예외 발생 (무시): {stream_id} - {e}")


async def _wait_for_stream_running(manager: StreamManager, stream_id: str) -> str:
    """
    스트림이 RUNNING 상태가 될 때까지 대기하고, 최종 output_url을 반환합니다.
    실패 시 StreamError를 발생시킵니다.
    """
    deadline = time.time() + WAIT_TIMEOUT_SECONDS
    
    while time.time() < deadline:
        state = manager.get_stream_state(stream_id)
        if not state:
            raise StreamError(
                ErrorCode.STREAM_NOT_FOUND,
                f"스트림 소멸됨: {stream_id}",
                stream_id=stream_id
            )
        
        if state.status == StreamStatus.RUNNING:
            # 설정된 output_url이 있으면 우선 사용, 없으면 생성 규칙 따름
            return state.config.output_url or StreamConfig.generate_output_url(state.config.rtsp_url)
        
        if state.status == StreamStatus.ERROR:
            raise StreamError(
                ErrorCode.STREAM_CONNECTION_FAILED,
                f"스트림 시작 실패 (ERROR): {state.last_error or 'Unknown'}",
                stream_id=stream_id
            )
            
        time.sleep(WAIT_POLL_SECONDS)

    raise StreamError(
        ErrorCode.TRANSPORT_TIMEOUT,
        f"{WAIT_TIMEOUT_SECONDS}초 내에 스트림이 준비되지 않았습니다.",
        stream_id=stream_id
    )


def _get_stream_id_by_url(manager: StreamManager, input_url: str) -> str:
    """
    입력 RTSP URL로 스트림을 찾아 stream_id를 반환합니다.
    스트림이 없으면 StreamError를 발생시킵니다.
    """
    validated_input = _validate_rtsp_url(input_url)
    state = manager.get_stream_by_url(validated_input)
    
    if not state:
        raise StreamError(
            ErrorCode.STREAM_NOT_FOUND,
            f"입력 URL '{input_url}'에 해당하는 스트림을 찾을 수 없습니다.",
            stream_id="auto"
        )
    
    return state.stream_id

# === 라우트 ===
# 주의: 구체적인 경로(/by-input)를 동적 경로(/{stream_id})보다 먼저 등록해야 합니다.

@router.get("", response_model=StreamListResponse)
async def list_streams(manager: StreamManager = Depends(get_stream_manager)) -> StreamListResponse:
    """모든 스트림 상태를 조회합니다."""
    states = manager.get_all_streams()
    return StreamListResponse(data=[_state_to_summary(s) for s in states])


@router.get("/by-input", response_model=StreamControlResponse)
async def get_output_url_by_input(
    input_url: str = Query(..., description="입력 RTSP URL"),
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    입력 URL로 스트림을 찾거나 생성하고, 출력 URL을 반환합니다.
    
    - 기존 스트림이 있으면: 출력 URL만 반환 (상태와 무관)
    - 스트림이 없으면: 새로 생성하고 출력 URL 반환
    """
    validated_input = _validate_rtsp_url(input_url)
    
    # 1. 기존 스트림 조회
    state = manager.get_stream_by_url(validated_input)
    
    if state:
        # 기존 스트림이 있으면 출력 URL만 반환
        output_url = state.config.output_url or StreamConfig.generate_output_url(validated_input)
        return StreamControlResponse(data={
            "stream_id": state.stream_id,
            "input_url": validated_input,
            "output_url": output_url,
            "name": state.stream_id,
            "status": state.status.value,
            "message": "기존 스트림을 찾았습니다."
        })
    
    # 2. 스트림이 없으면 새로 생성
    new_stream_id = f"auto-{str(uuid.uuid4())[:8]}"
    
    try:
        manager.start_stream(
            stream_id=new_stream_id,
            rtsp_url=validated_input,
            wait=False
        )
        
        # 3. RUNNING 대기 및 URL 획득
        final_output_url = await _wait_for_stream_running(manager, new_stream_id)
        
        return StreamControlResponse(data={
            "stream_id": new_stream_id,
            "input_url": validated_input,
            "output_url": final_output_url,
            "name": new_stream_id,
            "status": "running",
            "message": "새로운 스트림이 생성되었습니다."
        })
        
    except Exception as e:
        # 실패 시 정리 시도
        try:
            manager.stop_stream(new_stream_id, force=False)
            manager.delete_stream(new_stream_id)
        except:
            pass
        raise StreamError(
            ErrorCode.STREAM_CONNECTION_FAILED if not isinstance(e, StreamError) else e.code,
            f"입력 URL '{input_url}'로 스트림을 생성할 수 없습니다: {e}",
            stream_id="auto",
        ) from e


@router.post("/by-input/start", response_model=StreamControlResponse)
async def start_stream_by_input(
    input_url: str = Query(..., description="입력 RTSP URL"),
    payload: StreamStartByInputRequest | None = Body(default=None, description="스트림 설정 (선택)"),
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    입력 RTSP URL로 스트림을 찾아 시작합니다.
    
    - 기존 스트림이 있으면: 해당 스트림 시작
    - 스트림이 없으면: 새로 생성하고 시작
    
    Body의 rtsp_url이 있으면 그걸 사용하고, 없으면 input_url 쿼리 파라미터를 사용합니다.
    """
    # RTSP URL 결정: body가 있으면 body 우선, 없으면 쿼리 파라미터 사용
    rtsp_url = payload.rtsp_url if payload and payload.rtsp_url else input_url
    validated_url = _validate_rtsp_url(rtsp_url)
    
    if payload and payload.output_url:
        _validate_rtsp_url(payload.output_url)
    
    # 기존 스트림 조회
    state = manager.get_stream_by_url(validated_url)
    
    if state:
        # 기존 스트림이 있으면 해당 stream_id로 시작
        stream_id = state.stream_id
        
        # 설정 업데이트가 필요한 경우 (body에 설정이 있으면)
        if payload:
            try:
                final_state = manager.start_stream(
                    stream_id=stream_id,
                    rtsp_url=validated_url,
                    max_fps=payload.max_fps,
                    downscale=payload.downscale,
                    output_url=payload.output_url,
                    wait=True,
                )
                # 출력 URL을 명시적으로 반환
                output_url = final_state.config.output_url or StreamConfig.generate_output_url(validated_url)
                result = _state_to_dict(final_state)
                result["output_url"] = output_url
                return StreamControlResponse(data=result)
            except (StreamError, ValueError) as e:
                raise StreamError(
                    e.code if isinstance(e, StreamError) else ErrorCode.CONFIG_INVALID,
                    str(e),
                    stream_id=stream_id,
                ) from e
        else:
            # 설정 없이 재시작만
            final_state = manager.restart_stream(stream_id, wait=True)
            # 출력 URL을 명시적으로 반환
            output_url = final_state.config.output_url or StreamConfig.generate_output_url(validated_url)
            result = _state_to_dict(final_state)
            result["output_url"] = output_url
            return StreamControlResponse(data=result)
    else:
        # 스트림이 없으면 새로 생성
        new_stream_id = f"auto-{str(uuid.uuid4())[:8]}"
        
        try:
            final_state = manager.start_stream(
                stream_id=new_stream_id,
                rtsp_url=validated_url,
                max_fps=payload.max_fps if payload else None,
                downscale=payload.downscale if payload else None,
                output_url=payload.output_url if payload else None,
                wait=True,
            )
            # 출력 URL을 명시적으로 반환
            output_url = final_state.config.output_url or StreamConfig.generate_output_url(validated_url)
            result = _state_to_dict(final_state)
            result["output_url"] = output_url
            return StreamControlResponse(data=result)
        except (StreamError, ValueError) as e:
            # 실패 시 정리 시도
            try:
                manager.stop_stream(new_stream_id, force=False)
                manager.delete_stream(new_stream_id)
            except:
                pass
            raise StreamError(
                e.code if isinstance(e, StreamError) else ErrorCode.CONFIG_INVALID,
                f"입력 URL '{rtsp_url}'로 스트림을 시작할 수 없습니다: {e}",
                stream_id="auto",
            ) from e


@router.post("/by-input/stop", response_model=StreamControlResponse)
async def stop_stream_by_input(
    input_url: str = Query(..., description="입력 RTSP URL"),
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """입력 RTSP URL로 스트림을 찾아 중지합니다."""
    stream_id = _get_stream_id_by_url(manager, input_url)
    manager.stop_stream(stream_id)
    state = manager.get_stream_state(stream_id)
    data = _state_to_dict(state) if state else {"stream_id": stream_id, "status": "STOPPED"}
    return StreamControlResponse(data=data)


@router.post("/by-input/restart", response_model=StreamControlResponse)
async def restart_stream_by_input(
    input_url: str = Query(..., description="입력 RTSP URL"),
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """입력 RTSP URL로 스트림을 찾아 재시작합니다."""
    stream_id = _get_stream_id_by_url(manager, input_url)
    state = manager.restart_stream(stream_id, wait=True)
    return StreamControlResponse(data=_state_to_dict(state))


@router.post("/by-input/disconnect", response_model=StreamControlResponse)
async def disconnect_stream_by_input(
    input_url: str = Query(..., description="입력 RTSP URL"),
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    입력 RTSP URL로 스트림을 찾아 중지하고 모든 리소스를 정리한 후 삭제합니다.
    
    동작 순서:
    1. 입력 URL로 스트림 검색
    2. 스트림 중지 요청 (정상 종료 시도)
    3. 스레드 종료 대기 (최대 5초)
    4. 리소스 정리 (디코더, 퍼블리셔 해제)
    5. 스트림 삭제 (메모리에서 제거)
    """
    stream_id = _get_stream_id_by_url(manager, input_url)
    
    # 중지 및 삭제 (내부 헬퍼 함수 사용)
    _disconnect_stream_internal(manager, stream_id)
    
    return StreamControlResponse(data={
        "stream_id": stream_id,
        "input_url": input_url,
        "status": "DISCONNECTED",
        "message": "연결이 완전히 종료되었습니다."
    })


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
        # 출력 URL을 명시적으로 반환
        output_url = final_state.config.output_url or StreamConfig.generate_output_url(validated_url)
        result = _state_to_dict(final_state)
        result["output_url"] = output_url
        return StreamControlResponse(data=result)
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







@router.post("/{stream_id}/disconnect", response_model=StreamControlResponse)
async def disconnect_stream(
    stream_id: str,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    스트림을 중지하고 모든 리소스를 정리한 후 삭제합니다.
    
    동작 순서:
    1. 스트림 중지 요청 (정상 종료 시도)
    2. 스레드 종료 대기 (최대 5초)
    3. 리소스 정리 (디코더, 퍼블리셔 해제)
    4. 스트림 삭제 (메모리에서 제거)
    
    주의: 네트워크 I/O 블로킹 중이면 5초 이상 걸릴 수 있습니다.
    """
    # 존재 여부 확인
    if not manager.get_stream_state(stream_id):
        raise StreamError(
            ErrorCode.STREAM_NOT_FOUND,
            "스트림이 없습니다.",
            stream_id=stream_id
        )

    # 중지 및 삭제 (내부 헬퍼 함수 사용)
    _disconnect_stream_internal(manager, stream_id)
    
    return StreamControlResponse(data={
        "stream_id": stream_id,
        "status": "DISCONNECTED",
        "message": "연결이 완전히 종료되었습니다."
    })


@router.post("/register", response_model=StreamControlResponse)
async def register_stream(
    payload: StreamRegisterRequest,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """
    입력 RTSP URL로 스트림을 강제로 재등록합니다.
    
    동작:
    1. 입력 RTSP URL로 기존 스트림 검색
    2. 기존 스트림이 있으면: 연결 끊기 → 새로 등록 → RUNNING 대기 → 출력 URL 반환
    3. 스트림이 없으면: 새로 등록 → RUNNING 대기 → 출력 URL 반환
    """
    validated_input = _validate_rtsp_url(payload.rtsp_url)
    
    # 1. 기존 스트림 조회
    existing_state = manager.get_stream_by_url(validated_input)
    
    # 2. 기존 스트림이 있으면 연결 끊기 (6번 API와 동일한 로직 사용)
    # 기존 스트림의 output_url을 보존 (설정에 있으면 사용)
    preserved_output_url = None
    if existing_state:
        existing_stream_id = existing_state.stream_id
        logger.info(f"기존 스트림 발견: {existing_stream_id}, 연결 끊기 후 재등록", stream_id=existing_stream_id)
        
        # 기존 스트림의 output_url 보존 (설정에 있으면 사용)
        preserved_output_url = existing_state.config.output_url
        
        # 6번 API(disconnect)와 동일한 로직 사용
        _disconnect_stream_internal(manager, existing_stream_id)
    
    # 3. 새로 등록
    new_stream_id = f"auto-{str(uuid.uuid4())[:8]}"
    
    # output_url 결정: 기존 스트림의 output_url이 있으면 사용, 없으면 generate_output_url 사용
    final_output_url = preserved_output_url or StreamConfig.generate_output_url(validated_input)
    
    try:
        manager.start_stream(
            stream_id=new_stream_id,
            rtsp_url=validated_input,
            output_url=final_output_url,
            wait=False
        )
        
        # 4. RUNNING 대기 및 URL 획득
        final_output_url = await _wait_for_stream_running(manager, new_stream_id)
        
        return StreamControlResponse(data={
            "output_url": final_output_url
        })
        
    except Exception as e:
        # 실패 시 정리 시도
        try:
            manager.stop_stream(new_stream_id, force=False)
            manager.delete_stream(new_stream_id)
        except:
            pass
        raise StreamError(
            ErrorCode.STREAM_CONNECTION_FAILED if not isinstance(e, StreamError) else e.code,
            f"입력 URL '{payload.rtsp_url}'로 스트림을 재등록할 수 없습니다: {e}",
            stream_id="auto",
        ) from e


