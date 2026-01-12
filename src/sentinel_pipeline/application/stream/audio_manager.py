"""
오디오 매니저

오디오 스트림의 생명주기와 분석 파이프라인을 관리합니다.
"""

from __future__ import annotations

import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, Callable
import logging
from pathlib import Path

from sentinel_pipeline.interface.config.schema import AudioStreamConfig
from sentinel_pipeline.domain.models.audio_stream import (
    AudioStreamState,
    AudioStreamStatus,
    AudioStreamStats
)
from sentinel_pipeline.domain.models.event import Event, EventType, EventStage
from sentinel_pipeline.infrastructure.audio.readers.mic_reader import MicAudioReader
from sentinel_pipeline.infrastructure.audio.readers.rtsp_reader import RtspAudioReader
from sentinel_pipeline.infrastructure.audio.processors.batch_scream_detector import BatchScreamDetector
from sentinel_pipeline.infrastructure.audio.processors.risk_analyzer import RiskAnalyzer
from sentinel_pipeline.infrastructure.audio.processors.vad_filter import create_vad_filter
from sentinel_pipeline.interface.api.ws_bus import publish_stream_update, broadcast

logger = logging.getLogger(__name__)


class AudioStreamContext:
    """오디오 스트림 실행 컨텍스트"""
    def __init__(self, config: AudioStreamConfig):
        self.stream_id = config.stream_id
        self.state = AudioStreamState(config=config)
        self.reader = None
        self.risk_analyzer = None
        self.vad_filter = None 
        
        # 상태 추적
        self.last_scream_time = 0.0
        self.scream_cooldown = 2.0
        
        # Latest-Win 전략을 위한 버퍼 및 상태
        self.latest_chunk = None  # 가장 최신 오디오 청크 (대기열 크기 1 효과)
        self.is_processing = False # 분석 루프 실행 중 여부
        
        # 성능 측정용
        self.chunk_counter = 0  # 청크 카운터 (Sampling 로깅용)
        self.chunk_duration_ms = int(config.chunk_duration * 1000)  # 청크 크기 (ms)


class AudioManager:
    """오디오 스트림 및 분석 관리자"""
    
    def __init__(self):
        self._streams: Dict[str, AudioStreamContext] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="AudioAnalysis")
        self._event_emitter: Optional[EventEmitter] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 공유 배치 엔진 (Lazy Loading)
        self._batch_scream_detector: Optional[BatchScreamDetector] = None
        
        # 기본 모델 경로
        self.model_dir = Path("models/audio")
        self.scream_model_path = self.model_dir / "resnet18_scream_detector_v2.pth"
        
    def set_dependencies(self, event_emitter: EventEmitter):
        self._event_emitter = event_emitter
        
    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        # 루프가 설정되면 배치 엔진도 시작
        if self._batch_scream_detector:
            self._batch_scream_detector.start(loop)

    def start_stream(self, config: AudioStreamConfig) -> AudioStreamState:
        with self._lock:
            # 1. 공유 배치 탐지기 초기화 (최초 1회)
            if self._batch_scream_detector is None:
                model_path = config.scream_model_path if config.scream_model_path else str(self.scream_model_path)
                logger.info(f"🚀 Initializing Shared BatchScreamDetector: {model_path}")
                self._batch_scream_detector = BatchScreamDetector(
                    model_path=model_path,
                    threshold=config.scream_threshold,
                    batch_size=16  # TODO: 설정에서 가져오게 수정 가능
                )
                # 이미 루프가 설정되어 있다면 즉시 시작
                if self._loop:
                    self._batch_scream_detector.start(self._loop)
                ctx = self._streams[config.stream_id]
                if ctx.state.is_active:
                    logger.warning(f"Audio stream already active: {config.stream_id}")
                    return ctx.state
                
            ctx = AudioStreamContext(config)
            self._streams[config.stream_id] = ctx
            
            try:
                # 2. 프로세서 초기화
                
                if config.stt_enabled:
                    ctx.risk_analyzer = RiskAnalyzer(
                        stream_id=config.stream_id,
                        model_size=config.stt_model_size,
                        enable_medium_path=config.enable_medium_path,
                        enable_heavy_path=config.enable_heavy_path,
                        heavy_path_async=config.heavy_path_async,
                        semantic_threshold=config.semantic_threshold,
                        use_korean_model=config.use_korean_model
                    )
                    
                    try:
                        ctx.vad_filter = create_vad_filter(
                            sample_rate=config.sample_rate,
                            threshold=0.5,
                            use_highpass=True
                        )
                        logger.info(f"🛡️ VAD Filter initialized for stream {config.stream_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to initialize VAD filter for {config.stream_id}: {e}")
                        ctx.vad_filter = None
                
                # 3. 리더 초기화 (스레드 리더 -> 비동기 브리지)
                on_chunk_cb = lambda chunk: self._bridge_on_chunk(ctx, chunk)
                
                if config.use_microphone:
                    ctx.reader = MicAudioReader(
                        sample_rate=config.sample_rate,
                        chunk_duration=config.chunk_duration,
                        device_index=config.mic_device_index,
                        on_chunk=on_chunk_cb
                    )
                else:
                    ctx.reader = RtspAudioReader(
                        rtsp_url=config.rtsp_url,
                        sample_rate=config.sample_rate,
                        chunk_duration=config.chunk_duration,
                        on_chunk=on_chunk_cb
                    )
                
                # 3. 시작
                ctx.state.set_status(AudioStreamStatus.STARTING)
                try:
                    ctx.reader.start()
                except Exception as e:
                    # 리더 시작 실패 시 즉시 예외 발생
                    logger.error(f"Failed to start audio reader for {config.stream_id}: {e}")
                    ctx.state.set_status(AudioStreamStatus.ERROR, str(e))
                    self._notify_status_change(ctx.stream_id, AudioStreamStatus.ERROR)
                    raise  # 예외를 다시 발생시켜서 API에서 처리하도록
                
                ctx.state.set_status(AudioStreamStatus.RUNNING)
                self._notify_status_change(ctx.stream_id, AudioStreamStatus.RUNNING)
                logger.info(f"Audio stream started: {config.stream_id}")
                
            except Exception as e:
                # 다른 초기화 단계에서 발생한 예외
                logger.error(f"Failed to start audio stream {config.stream_id}: {e}")
                ctx.state.set_status(AudioStreamStatus.ERROR, str(e))
                self._notify_status_change(ctx.stream_id, AudioStreamStatus.ERROR)
                raise  # 예외를 다시 발생시켜서 API에서 처리하도록
                
            return ctx.state

    def stop_stream(self, stream_id: str) -> bool:
        with self._lock:
            if stream_id not in self._streams:
                return False
                
            ctx = self._streams[stream_id]
            
            if ctx.reader:
                ctx.reader.stop()
            
            ctx.state.set_status(AudioStreamStatus.STOPPED)
            
            # 리소스 정리
            del self._streams[stream_id]
            logger.info(f"Audio stream stopped: {stream_id}")
            
            # 삭제 알림은 API 레이어에서 처리 (스트림이 이미 삭제된 후에는 상태 변경 알림이 의미 없음)
            return True

    def get_stream_state(self, stream_id: str) -> Optional[AudioStreamState]:
        with self._lock:
            if stream_id in self._streams:
                return self._streams[stream_id].state
            return None

    def get_all_streams(self) -> list[AudioStreamState]:
        with self._lock:
            return [ctx.state for ctx in self._streams.values()]

    def _bridge_on_chunk(self, ctx: AudioStreamContext, chunk: Any):
        """스레드 리더 -> 비동기 이벤트 루프 연결 브리지"""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._async_on_audio_chunk(ctx, chunk),
                self._loop
            )

    async def _async_on_audio_chunk(self, ctx: AudioStreamContext, chunk: Any):
        """[Async] 오디오 청크 처리 핵심 로직"""
        chunk_received_time = time.monotonic()
        
        try:
            ctx.state.stats.record_chunk()
            ctx.chunk_counter += 1
            current_time = time.monotonic()
            
            # 1. 비명 감지 (Batch Inference 요청)
            inference_start_time = time.monotonic()
            
            # 공유 배치 탐지기 호출 (await로 결과를 기다리지만, 다른 스트림은 그 사이 큐에 쌓임)
            if self._batch_scream_detector:
                scream_result = await self._batch_scream_detector.predict(chunk)
            else:
                return # 엔진 미초기화 시 스킵
                
            inference_done_time = time.monotonic()
            scream_detected = scream_result['is_scream']
            
            # 추론 시간 및 레이턴시
            inference_ms = (inference_done_time - inference_start_time) * 1000
            total_latency_ms = (inference_done_time - chunk_received_time) * 1000
            
            # WebSocket 상태 전송
            self._send_pipeline_status(
                stream_id=ctx.stream_id,
                node_id=2, 
                scream_detected=scream_detected,
                confidence=scream_result.get('prob', 0.0)
            )
            
            if scream_detected:
                if current_time - ctx.last_scream_time > ctx.scream_cooldown:
                    ctx.last_scream_time = current_time
                    ctx.state.stats.scream_detected_count += 1
                    
                    self._send_scream_detected(ctx.stream_id, scream_result['prob'])
                    self._emit_event(
                        stream_id=ctx.stream_id,
                        event_type=EventType.SCREAM,
                        confidence=scream_result['prob'],
                        details={'batch_mode': True}
                    )
                    
                    logger.info(
                        f"[STREAM={ctx.stream_id}] [BATCH] chunk={ctx.chunk_counter} "
                        f"infer={inference_ms:.1f}ms score={scream_result['prob']:.2f} "
                        f"event=DETECTED latency={total_latency_ms:.1f}ms"
                    )
            else:
                if ctx.chunk_counter % 100 == 0:
                    logger.debug(f"[STREAM={ctx.stream_id}] [BATCH] chunk={ctx.chunk_counter} Normal")

                # 2. STT 및 위험 분석 (VAD 필터 포함)
                if ctx.risk_analyzer:
                    is_speech = True
                    if ctx.vad_filter:
                        try:
                            # VAD는 여전히 CPU 작업 (추후 이것도 Batch화 가능)
                            is_speech = ctx.vad_filter.is_speech(chunk)
                        except Exception:
                            is_speech = True
                    
                    if is_speech:
                        ctx.latest_chunk = chunk
                        if not ctx.is_processing:
                            ctx.is_processing = True
                            # STT 루프는 여전히 ThreadPool에서 실행 (Whisper 가용성 때문)
                            self._executor.submit(self._process_loop, ctx)
                
        except Exception as e:
            logger.error(f"Error in async_on_audio_chunk for {ctx.stream_id}: {e}", exc_info=True)
            ctx.state.stats.record_error()
                
        except Exception as e:
            logger.error(f"Error processing audio chunk for {ctx.stream_id}: {e}")
            ctx.state.stats.record_error()

    def _process_loop(self, ctx: AudioStreamContext):
        """분석 작업 루프 (항상 최신 청크 처리)"""
        try:
            while True:
                # 1. 가장 최신 데이터 가져오기 (Atomic에 가까운 동작)
                # 락을 걸지 않아도 Python의 변수 할당은 Atomic하지만, 
                # 확실하게 가져오고 비우기 위해 임시 변수 사용
                chunk = ctx.latest_chunk
                ctx.latest_chunk = None
                
                # 더 이상 처리할 데이터가 없으면 종료
                if chunk is None:
                    ctx.is_processing = False
                    break
                
                # 2. 무거운 분석 수행 (여기서 시간 소요)
                # 이 동안 들어온 데이터는 ctx.latest_chunk에 계속 덮어씌워짐 (Drop Oldest)
                self._process_risk_analysis(ctx, chunk)
                
                # 3. 루프 반복 -> 그 사이에 새로운 데이터가 들어왔는지 확인하러 감
        except Exception as e:
            logger.error(f"Analysis loop error for {ctx.stream_id}: {e}")
            ctx.is_processing = False

    def _process_risk_analysis(self, ctx: AudioStreamContext, chunk: Any):
        """백그라운드 STT 처리 (비동기 콜백 패턴)"""
        
        # [내부 함수] 결과가 나왔을 때 실행될 콜백 (워커 스레드에서 실행됨)
        def on_inference_complete(result: dict):
            # 워커 스레드에서 실행되므로, 예외 처리 중요
            try:
                # 1. STT 결과 전송
                self._send_stt_result(
                    stream_id=ctx.stream_id,
                    stt_result=result
                )
                
                # 2. 위험 감지 시 이벤트 처리
                if result.get('is_dangerous'):
                    ctx.state.stats.keyword_detected_count += 1
                    event_type = result.get('event_type') or EventType.CUSTOM
                    
                    self._send_event_detected(
                        stream_id=ctx.stream_id,
                        event_type=event_type,
                        data={
                            'keyword': result.get('keyword'),
                            'text': result.get('text'),
                            'confidence': result.get('confidence')
                        }
                    )
                    
                    self._emit_event(
                        stream_id=ctx.stream_id,
                        event_type=event_type,
                        confidence=result.get('confidence', 0.0),
                        details={
                            'keyword': result.get('keyword'),
                            'text': result.get('text'),
                            'original_text': result.get('text') # 호환성
                        }
                    )
                else:
                    # 안전할 때도 상태 업데이트
                    self._send_pipeline_status(
                        stream_id=ctx.stream_id,
                        node_id=5,
                        scream_detected=False,
                        confidence=0.0
                    )
            except Exception as e:
                logger.error(f"Error in inference callback for {ctx.stream_id}: {e}")

        try:
            # RiskAnalyzer에게 일감 던지기 (콜백 함수도 같이 줌)
            ctx.risk_analyzer.process(chunk, callback=on_inference_complete)
                
        except Exception as e:
            logger.error(f"Risk analysis submission error for {ctx.stream_id}: {e}")

    def _emit_event(self, stream_id: str, event_type: EventType, confidence: float, details: dict):
        if self._event_emitter:
            event = Event(
                type=event_type,
                stage=EventStage.DETECTED,
                confidence=confidence,
                stream_id=stream_id,
                module_name="AudioModule",
                details=details
            )
            self._event_emitter.emit(event)

    def _notify_status_change(self, stream_id: str, status: AudioStreamStatus):
        if self._loop:
            payload = {
                "type": "stream_update",  # 프론트엔드가 기대하는 타입
                "stream_id": stream_id,
                "status": status.value
            }
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(publish_stream_update(payload))
            )
    
    def _send_pipeline_status(self, stream_id: str, node_id: int, scream_detected: bool, confidence: float):
        """파이프라인 상태를 WebSocket으로 전송"""
        if self._loop:
            payload = {
                "type": "pipeline_status",
                "stream_id": stream_id,
                "node_id": node_id,
                "scream_detected": scream_detected,
                "confidence": confidence,
                "status": "alert" if scream_detected else "processing"
            }
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast(payload))
            )
    
    def _send_scream_detected(self, stream_id: str, confidence: float):
        """비명 감지 이벤트를 WebSocket으로 전송"""
        if self._loop:
            payload = {
                "type": "scream_detected",
                "stream_id": stream_id,
                "confidence": confidence
            }
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast(payload))
            )
    
    def _send_stt_result(self, stream_id: str, stt_result: dict):
        """STT 결과를 WebSocket으로 전송"""
        if self._loop:
            payload = {
                "type": "stt_result",
                "stream_id": stream_id,
                "stt_result": {
                    "text": stt_result.get('text', ''),
                    "risk_analysis": {
                        "is_dangerous": stt_result.get('is_dangerous', False),
                        "keyword": stt_result.get('keyword', ''),
                        "confidence": stt_result.get('confidence', 0.0),
                        "event_type": stt_result.get('event_type', None),
                        "path": stt_result.get('path', 'none')
                    }
                }
            }
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast(payload))
            )
    
    def _send_event_detected(self, stream_id: str, event_type: EventType, data: dict):
        """위험 키워드 감지 이벤트를 WebSocket으로 전송"""
        if self._loop:
            payload = {
                "type": "event_detected",
                "stream_id": stream_id,
                "event_type": event_type.value if hasattr(event_type, 'value') else str(event_type),
                "data": data
            }
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast(payload))
            )

    def stop_all(self):
        for stream_id in list(self._streams.keys()):
            self.stop_stream(stream_id)
        
        # 공유 배치 탐지기 정리
        if self._batch_scream_detector and self._loop:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self._batch_scream_detector.stop())
            )
            self._batch_scream_detector = None
            
        self._executor.shutdown(wait=False)
