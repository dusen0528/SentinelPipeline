"""
GPU 부하 테스트 API 엔드포인트

벤치마크 테스트 실행 및 시스템 상태 조회 API를 제공합니다.
SSE(Server-Sent Events)를 통한 실시간 로그 스트리밍 지원.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.scripts.benchmark_runner import LoadTestSimulator, BenchmarkResult

logger = get_logger(__name__)

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


# === Request/Response Models ===

class BenchmarkRequest(BaseModel):
    """벤치마크 테스트 요청 (실제 오디오 파일 사용)"""
    streams: int = Field(default=10, ge=1, le=100, description="시뮬레이션할 스트림 개수")
    whisper_model: str = Field(default="base", description="Whisper 모델 크기 (tiny, base, small, medium)")
    gpu_enabled: bool = Field(default=True, description="GPU 사용 여부")
    warmup: bool = Field(default=True, description="워밍업 실행 여부")
    continuous: bool = Field(default=False, description="연속 부하 테스트 모드")
    duration: float = Field(default=30.0, ge=5.0, le=300.0, description="연속 테스트 지속 시간 (초)")
    interval: float = Field(default=1.0, ge=0.5, le=5.0, description="오디오 입력 간격 (초)")


class StreamDetail(BaseModel):
    """개별 스트림 처리 결과 (청크별 리소스 포함)"""
    stream_id: int
    step1_latency: float = Field(description="비명 감지 시간 (ms)")
    step2_latency: float = Field(description="STT 변환 시간 (ms)")
    total_latency: float = Field(description="전체 처리 시간 (ms)")
    detected: bool = Field(description="비명 감지 여부")
    scream_prob: float = Field(description="비명 확률")
    audio_file: str = Field(default="", description="사용된 오디오 파일명")
    audio_category: str = Field(default="normal", description="Ground Truth 카테고리 (scream|emergency_keyword|normal)")
    gpu_memory_mb: float = Field(default=0.0, description="청크 처리 시점 GPU 메모리 (MB)")
    cpu_percent: float = Field(default=0.0, description="청크 처리 시점 CPU 사용률 (%)")
    transcript: str = Field(default="", description="STT 결과 텍스트")


class BenchmarkResponse(BaseModel):
    """벤치마크 테스트 응답"""
    success: bool
    streams: int
    avg_latency_ms: float = Field(description="평균 지연 시간 (ms)")
    max_latency_ms: float = Field(description="최대 지연 시간 (ms)")
    min_latency_ms: float = Field(description="최소 지연 시간 (ms)")
    fps: float = Field(description="초당 처리량 (chunks/sec 또는 streams/sec)")
    gpu_memory_mb: float = Field(description="GPU 메모리 사용량 (MB)")
    gpu_memory_peak_mb: float = Field(description="GPU 최대 메모리 사용량 (MB)")
    cpu_percent: float = Field(description="CPU 사용률 (%)")
    device: str = Field(description="사용된 디바이스 (cuda/cpu)")
    scream_count: int = Field(description="비명 감지 횟수 (detected=True)")
    stt_count: int = Field(description="STT 실행 횟수 (detected=False일 때 실행)")
    total_time: float = Field(description="총 테스트 시간 (초)")
    continuous: bool = Field(default=False, description="연속 테스트 모드 여부")
    duration: float = Field(default=0, description="연속 테스트 지속 시간 (초)")
    total_processed: int = Field(default=0, description="총 처리된 오디오 청크 수")
    details: list[StreamDetail] = Field(default_factory=list, description="개별 스트림 결과")
    timestamp: str = Field(default="", description="테스트 시간")


class SystemStatusResponse(BaseModel):
    """시스템 상태 응답"""
    success: bool
    device: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    gpu_name: Optional[str] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_memory_allocated_mb: Optional[float] = None
    gpu_memory_cached_mb: Optional[float] = None


# === Endpoints ===

@router.post("/run", response_model=BenchmarkResponse)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResponse:
    """
    GPU 부하 테스트 실행
    
    지정된 수의 가상 오디오 스트림을 처리하고 성능 메트릭을 반환합니다.
    
    - **streams**: 시뮬레이션할 스트림 개수 (1~100)
    - **whisper_model**: Whisper 모델 크기
    - **gpu_enabled**: GPU 사용 여부
    - **warmup**: 모델 워밍업 실행 여부 (첫 실행 시 권장)
    """
    try:
        mode = "continuous" if request.continuous else "batch"
        logger.info(
            f"벤치마크 시작 ({mode}): streams={request.streams}, "
            f"whisper_model={request.whisper_model}"
        )
        
        # 시뮬레이터 생성 (실제 오디오 파일 사용)
        simulator = LoadTestSimulator(
            num_streams=request.streams,
            gpu_enabled=request.gpu_enabled,
            whisper_model=request.whisper_model,
        )
        
        # 벤치마크 실행
        if request.continuous:
            result: BenchmarkResult = simulator.run_continuous_test(
                duration=request.duration,
                interval=request.interval,
                warmup=request.warmup,
            )
        else:
            result: BenchmarkResult = simulator.run_batch_test(
                warmup=request.warmup,
            )
        
        # 리소스 정리
        simulator.cleanup()
        
        logger.info(
            f"벤치마크 완료: avg_latency={result.avg_latency*1000:.1f}ms, "
            f"fps={result.fps:.1f}, gpu_mem={result.gpu_memory_mb:.0f}MB"
        )
        
        return BenchmarkResponse(
            success=True,
            streams=result.streams,
            avg_latency_ms=round(result.avg_latency * 1000, 2),
            max_latency_ms=round(result.max_latency * 1000, 2),
            min_latency_ms=round(result.min_latency * 1000, 2),
            fps=result.fps,
            gpu_memory_mb=result.gpu_memory_mb,
            gpu_memory_peak_mb=result.gpu_memory_peak_mb,
            cpu_percent=result.cpu_percent,
            device=result.device,
            scream_count=result.scream_count,
            stt_count=result.stt_count,
            total_time=result.total_time,
            continuous=request.continuous,
            duration=result.duration,
            total_processed=result.total_processed,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            details=[
                StreamDetail(
                    stream_id=d["stream_id"],
                    step1_latency=d["step1_latency"],
                    step2_latency=d["step2_latency"],
                    total_latency=d["total_latency"],
                    detected=d["detected"],
                    scream_prob=d["scream_prob"],
                    audio_file=d.get("audio_file", ""),
                    audio_category=d.get("audio_category", "normal"),
                    gpu_memory_mb=d.get("gpu_memory_mb", 0.0),
                    cpu_percent=d.get("cpu_percent", 0.0),
                    transcript=d.get("transcript", ""),
                )
                for d in result.details
            ],
        )
        
    except Exception as e:
        logger.error(f"벤치마크 실행 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"벤치마크 실행 실패: {str(e)}")


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status() -> SystemStatusResponse:
    """
    현재 시스템 상태 조회
    
    CPU, 메모리, GPU 사용량 등 시스템 리소스 상태를 반환합니다.
    """
    try:
        simulator = LoadTestSimulator(num_streams=1, gpu_enabled=True)
        status = simulator.get_system_status()
        
        return SystemStatusResponse(
            success=True,
            device=status["device"],
            cpu_percent=status["cpu_percent"],
            memory_percent=status["memory_percent"],
            memory_available_gb=status["memory_available_gb"],
            gpu_name=status.get("gpu_name"),
            gpu_memory_total_mb=status.get("gpu_memory_total_mb"),
            gpu_memory_allocated_mb=status.get("gpu_memory_allocated_mb"),
            gpu_memory_cached_mb=status.get("gpu_memory_cached_mb"),
        )
        
    except Exception as e:
        logger.error(f"시스템 상태 조회 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"시스템 상태 조회 실패: {str(e)}")


@router.get("/presets")
async def get_presets() -> dict:
    """
    테스트 시나리오 프리셋 목록 반환 (스트림 개수 기준)
    """
    return {
        "success": True,
        "presets": [
            {
                "name": "Single",
                "description": "단일 스트림 기준선 측정",
                "streams": 1,
            },
            {
                "name": "Light",
                "description": "가벼운 부하 (5 스트림)",
                "streams": 5,
            },
            {
                "name": "Normal",
                "description": "일반 부하 (10 스트림)",
                "streams": 10,
            },
            {
                "name": "Medium",
                "description": "중간 부하 (20 스트림)",
                "streams": 20,
            },
            {
                "name": "Heavy",
                "description": "높은 부하 (30 스트림)",
                "streams": 30,
            },
            {
                "name": "Panic",
                "description": "최악의 시나리오 (50 스트림)",
                "streams": 50,
            },
        ],
    }


# === SSE (Server-Sent Events) 실시간 스트리밍 ===

@router.get("/stream")
async def stream_benchmark(
    streams: int = 10,
    whisper_model: str = "base",
    gpu_enabled: bool = True,
    warmup: bool = True,
    continuous: bool = False,
    duration: float = 30.0,
    interval: float = 1.0,
):
    """
    SSE를 통한 실시간 벤치마크 스트리밍
    
    프론트엔드에서 EventSource로 연결하면 각 스트림 처리 결과를 실시간으로 수신할 수 있습니다.
    
    사용법:
        const sse = new EventSource('/api/benchmark/stream?streams=10');
        sse.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    # 메시지 큐 (thread-safe)
    message_queue: queue.Queue = queue.Queue()
    
    def progress_callback(data: dict):
        """벤치마크 진행 상황을 큐에 추가"""
        message_queue.put(data)
    
    def run_benchmark_sync():
        """동기 벤치마크 실행 (별도 스레드)"""
        try:
            logger.info(f"SSE 벤치마크 시작: streams={streams}, whisper={whisper_model}, continuous={continuous}")
            
            simulator = LoadTestSimulator(
                num_streams=streams,
                gpu_enabled=gpu_enabled,
                whisper_model=whisper_model,
            )
            
            if continuous:
                logger.info(f"연속 테스트 모드: duration={duration}s, interval={interval}s")
                result = simulator.run_continuous_test(
                    duration=duration,
                    interval=interval,
                    warmup=warmup,
                    progress_callback=progress_callback,
                )
            else:
                logger.info("배치 테스트 모드")
                result = simulator.run_batch_test(
                    warmup=warmup,
                    progress_callback=progress_callback,
                )
            
            logger.info(f"벤치마크 완료: avg_latency={result.avg_latency*1000:.1f}ms, fps={result.fps:.1f}")
            
            # 최종 결과 전송
            message_queue.put({
                "type": "complete",
                "success": True,
                "streams": result.streams,
                "avg_latency_ms": round(result.avg_latency * 1000, 2),
                "max_latency_ms": round(result.max_latency * 1000, 2),
                "min_latency_ms": round(result.min_latency * 1000, 2),
                "fps": result.fps,
                "gpu_memory_mb": result.gpu_memory_mb,
                "gpu_memory_peak_mb": result.gpu_memory_peak_mb,
                "cpu_percent": result.cpu_percent,
                "device": result.device,
                "scream_count": result.scream_count,
                "stt_count": result.stt_count,
                "total_time": result.total_time,
                "continuous": continuous,
                "duration": result.duration,
                "total_processed": result.total_processed,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "details": result.details,
            })
            
            simulator.cleanup()
            logger.info("벤치마크 리소스 정리 완료")
            
        except Exception as e:
            logger.error(f"SSE 벤치마크 오류: {e}", exc_info=True)
            message_queue.put({
                "type": "error",
                "message": str(e)
            })
        finally:
            # 종료 신호
            message_queue.put(None)
            logger.info("SSE 벤치마크 스레드 종료")
    
    async def event_generator():
        """SSE 이벤트 제너레이터"""
        # 벤치마크를 별도 스레드에서 실행
        thread = threading.Thread(target=run_benchmark_sync, daemon=True)
        thread.start()
        
        while True:
            try:
                # 비동기로 큐에서 메시지 가져오기 (타임아웃으로 non-blocking)
                try:
                    data = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: message_queue.get(timeout=0.1)
                    )
                except queue.Empty:
                    # 하트비트 전송 (연결 유지)
                    yield f": heartbeat\n\n"
                    continue
                
                if data is None:
                    # 종료 신호
                    yield f"event: close\ndata: {{}}\n\n"
                    break
                
                # JSON으로 직렬화하여 전송
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.error(f"SSE 스트리밍 오류: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                break
        
        # 스레드 종료 대기
        thread.join(timeout=5.0)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 버퍼링 비활성화
        }
    )
