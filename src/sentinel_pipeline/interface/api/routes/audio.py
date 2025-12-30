"""
오디오 스트림 API 라우터"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from sentinel_pipeline.interface.api.dependencies import get_audio_manager
from sentinel_pipeline.application.stream.audio_manager import AudioManager
from sentinel_pipeline.interface.config.schema import AudioStreamConfig
from sentinel_pipeline.domain.models.audio_stream import AudioStreamStatus

router = APIRouter(prefix="/api/audio", tags=["Audio Streams"])


class AudioStreamCreateRequest(BaseModel):
    stream_id: str
    rtsp_url: Optional[str] = None
    use_microphone: bool = False
    mic_device_index: Optional[int] = None
    sample_rate: int = 16000
    scream_threshold: float = 0.8
    stt_enabled: bool = True
    stt_model_size: str = "base"


@router.get("/streams", response_model=List[Dict[str, Any]])
async def list_streams(
    manager: AudioManager = Depends(get_audio_manager)
):
    """생성된 오디오 스트림 목록 조회"""
    streams = manager.get_all_streams()
    return [s.to_dict() for s in streams]


@router.post("/streams", status_code=status.HTTP_201_CREATED)
async def register_stream(
    request: AudioStreamCreateRequest,
    manager: AudioManager = Depends(get_audio_manager)
):
    """오디오 스트림 등록 및 시작"""
    if manager.get_stream_state(request.stream_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stream {request.stream_id} already exists"
        )
        
    config = AudioStreamConfig(
        stream_id=request.stream_id,
        rtsp_url=request.rtsp_url,
        use_microphone=request.use_microphone,
        mic_device_index=request.mic_device_index,
        sample_rate=request.sample_rate,
        scream_threshold=request.scream_threshold,
        stt_enabled=request.stt_enabled,
        stt_model_size=request.stt_model_size
    )
    
    try:
        state = manager.start_stream(config)
        if state.status == AudioStreamStatus.ERROR:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start stream: {state.last_error}"
            )
        return state.to_dict()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.delete("/streams/{stream_id}")
async def delete_stream(
    stream_id: str,
    manager: AudioManager = Depends(get_audio_manager)
):
    """오디오 스트림 삭제 및 중지"""
    if not manager.stop_stream(stream_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found"
        )
    
    # WebSocket으로 삭제 알림 전송
    from sentinel_pipeline.interface.api.ws_bus import publish_stream_update
    await publish_stream_update({
        "type": "stream_update",
        "stream_id": stream_id,
        "action": "deleted"
    })
    
    return {"message": "Stream deleted", "stream_id": stream_id}


@router.get("/streams/{stream_id}/status")
async def get_stream_status(
    stream_id: str,
    manager: AudioManager = Depends(get_audio_manager)
):
    """오디오 스트림 상태 조회"""
    state = manager.get_stream_state(stream_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream {stream_id} not found"
        )
    return state.to_dict()
