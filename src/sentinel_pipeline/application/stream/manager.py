"""
스트림 관리자

RTSP 스트림의 생명주기(시작, 중지, 재시작)를 관리합니다.
"""

from __future__ import annotations

import time
import threading
import uuid
from typing import TYPE_CHECKING, Any, Callable, Protocol
from enum import Enum

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.common.errors import StreamError, ErrorCode
from sentinel_pipeline.domain.models.stream import (
    StreamConfig,
    StreamState,
    StreamStatus,
)
from sentinel_pipeline.domain.models.event import Event, EventType
from sentinel_pipeline.application.pipeline.pipeline import PipelineEngine
from sentinel_pipeline.application.event.emitter import EventEmitter
import asyncio
from sentinel_pipeline.interface.api.ws_bus import publish_stream_update

logger = get_logger(__name__)

class DecoderProtocol(Protocol):
    """디코더 인터페이스 정의"""
    @property
    def is_connected(self) -> bool: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def fps(self) -> float: ...
    def connect(self, rtsp_url: str) -> bool: ...
    def read_frame(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...

class PublisherProtocol(Protocol):
    """퍼블리셔 인터페이스 정의"""
    def start(self) -> None: ...
    def stop(self, timeout: float = 5.0) -> None: ...
    def write_frame(self, frame: Any) -> bool: ...
    def update_resolution(self, width: int, height: int) -> None: ...
    def reconfigure(self, width: int, height: int) -> bool: ...

class StreamContext:
    """스트림 실행 컨텍스트"""
    def __init__(self, config: StreamConfig) -> None:
        self.stream_id = config.stream_id
        self.state = StreamState(config=config)
        self.decoder: DecoderProtocol | None = None
        self.publisher: PublisherProtocol | None = None
        self.thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        # [정책] 해상도 안정화 및 고정 캔버스 정책
        # 1순위: 개별 스트림 설정, 2순위: 전역 설정
        self.confirmed_width: int | None = config.target_width
        self.confirmed_height: int | None = config.target_height
        
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

class StreamManager:
    """스트림 생명주기 관리자"""
    MAX_CONCURRENT_STREAMS = 10
    JOIN_TIMEOUT_SECONDS = 5.0

    def __init__(self) -> None:
        self._streams: dict[str, StreamContext] = {}
        self._lock = threading.RLock()
        self._decoder_factory: Callable[..., DecoderProtocol] | None = None
        self._publisher_factory: Callable[..., PublisherProtocol] | None = None
        self._frame_callback: Callable[[Any, dict[str, Any]], tuple[Any, list[Any]]] | None = None
        self._on_status_change: Callable[[str, StreamStatus], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # 전역 설정
        self._global_max_fps = 15
        self._global_downscale = 1.0
        self._global_target_width: int | None = None
        self._global_target_height: int | None = None

        # 종속성
        self._pipeline_engine: PipelineEngine | None = None
        self._event_emitter: EventEmitter | None = None

    def set_dependencies(
        self, pipeline_engine: PipelineEngine, event_emitter: EventEmitter
    ) -> None:
        """종속성을 주입하고 이벤트 핸들러를 등록합니다."""
        self._pipeline_engine = pipeline_engine
        self._event_emitter = event_emitter
        self._event_emitter.set_on_events_emitted(self._handle_system_events)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """이벤트 루프를 설정합니다."""
        self._loop = loop
    
    def set_decoder_factory(self, factory: Callable[..., DecoderProtocol]) -> None:
        self._decoder_factory = factory
    
    def set_publisher_factory(self, factory: Callable[..., PublisherProtocol]) -> None:
        self._publisher_factory = factory
    
    def set_frame_callback(self, callback: Callable[[Any, dict[str, Any]], tuple[Any, list[Any]]]) -> None:
        self._frame_callback = callback
    
    def set_on_status_change(self, callback: Callable[[str, StreamStatus], None] | None) -> None:
        self._on_status_change = callback
    
    def start_stream(
        self,
        stream_id: str,
        rtsp_url: str,
        wait: bool = False,
        timeout: float = 10.0,
        **kwargs: Any,
    ) -> StreamState:
        with self._lock:
            active_streams = self.get_active_streams()
            if len(active_streams) >= self.MAX_CONCURRENT_STREAMS:
                raise StreamError(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    f"동시 스트림 한도({self.MAX_CONCURRENT_STREAMS}개)를 초과했습니다",
                    stream_id=stream_id,
                )

            if stream_id in self._streams:
                ctx = self._streams[stream_id]
                if ctx.state.is_active:
                    raise StreamError(
                        ErrorCode.STREAM_ALREADY_RUNNING,
                        f"스트림이 이미 실행 중입니다: {stream_id}",
                        stream_id=stream_id,
                    )
            
            downscale = kwargs.get("downscale", self._global_downscale)
            if downscale is None:
                downscale = self._global_downscale

            target_width = kwargs.get("target_width") or self._global_target_width
            target_height = kwargs.get("target_height") or self._global_target_height

            config = StreamConfig(
                stream_id=stream_id,
                rtsp_url=rtsp_url,
                max_fps=kwargs.get("max_fps", self._global_max_fps),
                downscale=downscale,
                output_url=kwargs.get("output_url"),
                target_width=target_width,
                target_height=target_height,
            )
            
            ctx = StreamContext(config)
            if ctx.confirmed_width is None:
                ctx.confirmed_width = self._global_target_width
            if ctx.confirmed_height is None:
                ctx.confirmed_height = self._global_target_height
                
            ctx.state.set_status(StreamStatus.STARTING)
            self._streams[stream_id] = ctx
            
            logger.info(
                f"스트림 시작: {stream_id}",
                stream_id=stream_id,
                rtsp_url=StreamConfig._mask_url(rtsp_url),
            )
            
            ctx.thread = threading.Thread(
                target=self._stream_loop,
                args=(ctx,),
                name=f"stream_{stream_id}",
                daemon=True,
            )
            ctx.thread.start()
            
            self._notify_status_change(stream_id, StreamStatus.STARTING)

        if wait:
            deadline = time.time() + timeout
            while time.time() < deadline:
                state = self.get_stream_state(stream_id)
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
                time.sleep(0.1)
            raise StreamError(
                ErrorCode.TRANSPORT_TIMEOUT,
                f"{timeout}초 내에 스트림이 RUNNING 상태가 되지 않았습니다",
                stream_id=stream_id,
            )
            
        return ctx.state
    
    def stop_stream(self, stream_id: str, force: bool = False) -> bool:
        with self._lock:
            ctx = self._streams.get(stream_id)
            if not ctx:
                raise StreamError(
                    ErrorCode.STREAM_NOT_FOUND,
                    f"스트림을 찾을 수 없습니다: {stream_id}",
                    stream_id=stream_id,
                )
            
            if not ctx.state.is_active:
                logger.warning(f"스트림이 이미 중지됨: {stream_id}", stream_id=stream_id)
                return True
            
            ctx.state.set_status(StreamStatus.STOPPING)
            ctx.request_stop()
            
            logger.info(f"스트림 중지 요청: {stream_id}", stream_id=stream_id)
        
        if ctx.thread and ctx.thread.is_alive():
            ctx.thread.join(timeout=self.JOIN_TIMEOUT_SECONDS)
            if ctx.thread.is_alive() and force:
                logger.warning(f"스트림 스레드 강제 종료: {stream_id}", stream_id=stream_id)
        
        self._cleanup_stream(ctx)
        
        with self._lock:
            ctx.state.set_status(StreamStatus.STOPPED)
            self._notify_status_change(stream_id, StreamStatus.STOPPED)
        
        return True
    
    def restart_stream(self, stream_id: str, wait: bool = False, timeout: float = 10.0) -> StreamState:
        with self._lock:
            ctx = self._streams.get(stream_id)
            if not ctx:
                raise StreamError(
                    ErrorCode.STREAM_NOT_FOUND,
                    f"스트림을 찾을 수 없습니다: {stream_id}",
                    stream_id=stream_id,
                )
            config = ctx.state.config
        
        try:
            self.stop_stream(stream_id)
        except StreamError:
            pass
        
        return self.start_stream(
            stream_id=config.stream_id,
            rtsp_url=config.rtsp_url,
            max_fps=config.max_fps,
            downscale=config.downscale,
            output_url=config.output_url,
            wait=wait,
            timeout=timeout,
        )

    def get_or_create_by_input_url(self, rtsp_url: str) -> StreamState:
        existing_state = self.get_stream_by_url(rtsp_url)

        if existing_state:
            stream_id = existing_state.stream_id
            status = existing_state.status
            
            if status in (StreamStatus.STOPPED, StreamStatus.ERROR):
                return self.restart_stream(stream_id, wait=True)
            elif status == StreamStatus.RUNNING:
                return existing_state
            else:
                return self.start_stream(stream_id, rtsp_url, wait=True)

        stream_id = f"auto-{str(uuid.uuid4())[:8]}"
        output_url = StreamConfig.generate_output_url(rtsp_url)
        
        return self.start_stream(
            stream_id=stream_id,
            rtsp_url=rtsp_url,
            output_url=output_url,
            wait=True
        )

    def get_stream_state(self, stream_id: str) -> StreamState | None:
        ctx = self._streams.get(stream_id)
        return ctx.state if ctx else None
    
    def get_all_streams(self) -> list[StreamState]:
        with self._lock:
            return [ctx.state for ctx in self._streams.values()]

    def get_stream_by_url(self, rtsp_url: str) -> StreamState | None:
        with self._lock:
            for state in self.get_all_streams():
                if state.config.rtsp_url.strip() == rtsp_url.strip():
                    return state
        return None
    
    def get_active_streams(self) -> list[StreamState]:
        with self._lock:
            return [ctx.state for ctx in self._streams.values() if ctx.state.is_active]
    
    def _stream_loop(self, ctx: StreamContext) -> None:
        stream_id = ctx.stream_id
        config = ctx.state.config
        
        try:
            if self._decoder_factory:
                try:
                    ctx.decoder = self._decoder_factory(stream_id)
                except TypeError:
                    ctx.decoder = self._decoder_factory()
                
                if not ctx.decoder.connect(config.rtsp_url):
                    raise StreamError(
                        ErrorCode.STREAM_CONNECTION_FAILED,
                        f"RTSP 연결 실패: {stream_id}",
                        stream_id=stream_id,
                    )
            else:
                logger.warning(f"디코더 팩토리 미설정: {stream_id}", stream_id=stream_id)
                ctx.state.set_status(StreamStatus.ERROR, "디코더 팩토리 미설정")
                return
            
            if self._publisher_factory and config.output_enabled and config.output_url:
                try:
                    pub_fps = int(config.max_fps or ctx.decoder.fps or 25)
                    
                    # [정책] 송출 해상도를 1280x720으로 강제 고정 (Letterbox 적용)
                    width, height = 1280, 720
                    ctx.confirmed_width, ctx.confirmed_height = width, height

                    logger.info(
                        f"출력 캔버스 고정: {width}x{height} (원본: {ctx.decoder.width}x{ctx.decoder.height})",
                        stream_id=stream_id
                    )

                    try:
                        ctx.publisher = self._publisher_factory(
                            stream_id, config.output_url, width, height, pub_fps
                        )
                    except TypeError:
                        ctx.publisher = self._publisher_factory()

                    ctx.publisher.start()
                    logger.info(
                        f"퍼블리셔 시작: {stream_id} -> {config.output_url} "
                        f"({width}x{height}@{pub_fps})",
                        stream_id=stream_id
                    )
                except Exception as e:
                    logger.error(f"퍼블리셔 시작 실패: {stream_id} - {e}", stream_id=stream_id)
                    raise
            
            ctx.state.set_status(StreamStatus.RUNNING)
            ctx.state.reset_retry()
            self._notify_status_change(stream_id, StreamStatus.RUNNING)
            
            frame_interval = 1.0 / config.max_fps
            last_frame_time = 0.0
            frame_number = 0
            consecutive_failures = 0
            MAX_CONSECUTIVE_FAILURES = 5
            
            while not ctx.should_stop():
                now = time.time()
                elapsed = now - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue
                
                last_frame_time = time.time()
                
                # 프레임 읽기 전 현재 해상도 저장 (해상도 변경 감지용)
                prev_w, prev_h = ctx.decoder.width, ctx.decoder.height
                
                # 프레임 읽기
                success, frame = ctx.decoder.read_frame()
                if not success or frame is None:
                    consecutive_failures += 1
                    ctx.state.stats.record_error()
                    
                    is_actually_connected = ctx.decoder.is_connected
                    if not is_actually_connected or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning(
                            f"프레임 읽기 실패: {stream_id} (연속 {consecutive_failures}회). 재연결 시도.",
                            stream_id=stream_id
                        )
                        if self._should_reconnect(ctx):
                            if self._attempt_reconnect(ctx):
                                consecutive_failures = 0
                                continue
                        else:
                            ctx.state.set_status(StreamStatus.ERROR, "재연결 실패")
                            self._notify_status_change(stream_id, StreamStatus.ERROR)
                            break
                    continue
                
                # [안정성 강화] 프레임 유효성 및 해상도 급변 체크
                h, w = frame.shape[:2]
                
                # 1. 비정상적으로 작은 프레임(데이터 깨짐) 차단
                if w < 100 or h < 100:
                    logger.warning(f"손상된 프레임 감지 ({w}x{h}), 무시합니다.", stream_id=stream_id)
                    continue

                # 2. 디코더 인식 해상도와 실제 프레임 크기 대조 (swscaler 에러 방어 핵심)
                # read_frame() 내부에서 이미 decoder.width가 업데이트되었을 수 있으므로
                # 읽기 전 해상도(prev_w)와 현재 프레임 해상도(w)를 비교합니다.
                if w != prev_w or h != prev_h:
                    # 일시적인 지터일 수 있으므로 일단 버리고, 재연결 유도
                    logger.warning(
                        f"해상도 변경 감지: {prev_w}x{prev_h} -> {w}x{h}. "
                        "고정 송출(1280x720)을 유지하며 디코더를 재설정합니다.",
                        stream_id=stream_id
                    )
                    if self._should_reconnect(ctx):
                        # [고정 캔버스 모드] 송출기(1280x720)는 유지하고 디코더만 리셋
                        if self._attempt_reconnect(ctx, restart_publisher=False):
                            consecutive_failures = 0
                            continue
                    continue

                consecutive_failures = 0
                if ctx.state.status == StreamStatus.RECONNECTING:
                    ctx.state.set_status(StreamStatus.RUNNING)
                    ctx.state.reset_retry()
                    self._notify_status_change(stream_id, StreamStatus.RUNNING)
                
                frame_number += 1
                metadata = {
                    "stream_id": stream_id,
                    "frame_number": frame_number,
                    "timestamp": time.time(),
                    "fps": ctx.state.stats.fps,
                }
                
                # =================================================================
                # 3. 프레임 처리 및 전송
                # =================================================================
                start_time = time.perf_counter()
                if self._frame_callback:
                    try:
                        processed_frame, events = self._frame_callback(frame, metadata)
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        ctx.state.stats.record_frame(latency_ms)
                        ctx.state.stats.record_event(len(events))
                        
                        # [핵심 수정] 퍼블리셔 생존 확인 및 심폐소생술
                        if ctx.publisher and config.output_enabled:
                            # 1. 퍼블리셔가 죽어있으면 재시작
                            if not ctx.publisher.is_running:
                                logger.warning(f"퍼블리셔 프로세스 사망 감지: {stream_id} -> 재시작 시도")
                                try:
                                    ctx.publisher.restart()
                                    logger.info(f"퍼블리셔 부활 성공: {stream_id}")
                                except Exception as e:
                                    logger.error(f"퍼블리셔 부활 실패: {e}")
                            
                            # 2. 프레임 전송 (이제 안전함)
                            ctx.publisher.write_frame(processed_frame)

                    except Exception as e:
                        logger.error(f"프레임 처리/전송 오류: {e}", stream_id=stream_id)
                        ctx.state.stats.record_error()
                else:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    ctx.state.stats.record_frame(latency_ms)
        
        except StreamError as e:
            logger.error(f"스트림 오류: {e}", stream_id=stream_id)
            ctx.state.set_status(StreamStatus.ERROR, str(e))
        except Exception as e:
            logger.exception(f"스트림 루프 예외: {e}", stream_id=stream_id)
            ctx.state.set_status(StreamStatus.ERROR, str(e))
        finally:
            self._cleanup_stream(ctx)
            logger.info(f"스트림 루프 종료: {stream_id}", stream_id=stream_id)
    
    def _should_reconnect(self, ctx: StreamContext) -> bool:
        if ctx.should_stop():
            return False
        if not ctx.state.config.reconnect_enabled:
            return False
        return ctx.state.retry_count < ctx.state.config.reconnect_max_retries
    
    def _attempt_reconnect(self, ctx: StreamContext, restart_publisher: bool = False) -> bool:
        stream_id = ctx.stream_id
        config = ctx.state.config
        
        if ctx.state.status != StreamStatus.RECONNECTING:
            ctx.state.set_status(StreamStatus.RECONNECTING)
            self._notify_status_change(stream_id, StreamStatus.RECONNECTING)
        
        ctx.state.record_retry()
        delay = ctx.state.calculate_next_retry_delay()
        logger.info(f"재연결 시도 ({ctx.state.retry_count}회, {delay:.1f}초 대기)", stream_id=stream_id)
        
        wait_end = time.time() + delay
        while time.time() < wait_end:
            if ctx.should_stop():
                return False
            time.sleep(0.1)
        
        # 1. 디코더 재연결
        if ctx.decoder:
            try: ctx.decoder.release()
            except: pass
            
            try:
                if ctx.decoder.connect(config.rtsp_url):
                    # 2. 해상도 변경 시 퍼블리셔 재시작
                    if restart_publisher and ctx.publisher:
                        logger.info(f"퍼블리셔 재설정: {ctx.decoder.width}x{ctx.decoder.height} -> 1280x720 고정")
                        # 송출 해상도는 무조건 1280x720으로 유지 (Letterbox)
                        ctx.confirmed_width = 1280
                        ctx.confirmed_height = 720
                        ctx.publisher.reconfigure(ctx.confirmed_width, ctx.confirmed_height)

                    ctx.state.set_status(StreamStatus.RUNNING)
                    ctx.state.reset_retry()
                    self._notify_status_change(stream_id, StreamStatus.RUNNING)
                    logger.info(f"스트림 재연결 성공 (해상도: {ctx.decoder.width}x{ctx.decoder.height})", stream_id=stream_id)
                    return True
            except Exception as e:
                logger.warning(f"재연결 시도 중 오류: {e}", stream_id=stream_id)
        
        return False

    # ... (delete_stream, _cleanup_stream, _notify_status_change 등은 그대로) ...
    def delete_stream(self, stream_id: str) -> bool:
        with self._lock:
            ctx = self._streams.get(stream_id)
            if not ctx:
                raise StreamError(ErrorCode.STREAM_NOT_FOUND, f"스트림 없음: {stream_id}", stream_id=stream_id)
        if ctx.state.is_active:
            self.stop_stream(stream_id)
        self._cleanup_stream(ctx)
        with self._lock:
            del self._streams[stream_id]
        return True

    def _cleanup_stream(self, ctx: StreamContext) -> None:
        if ctx.decoder:
            try: ctx.decoder.release()
            except: pass
            ctx.decoder = None
        if ctx.publisher:
            try: ctx.publisher.stop()
            except: pass
            ctx.publisher = None

    def _notify_status_change(self, stream_id: str, status: StreamStatus) -> None:
        if self._on_status_change:
            try: self._on_status_change(stream_id, status)
            except: pass
        state = self._streams.get(stream_id)
        stats = state.state.stats if state else None
        payload = {
            "stream_id": stream_id,
            "status": status.value,
            "fps": stats.fps if stats else None,
            "error_count": stats.error_count if stats else None,
            "last_frame_ts": stats.last_frame_ts if stats else None,
        }
        if self._loop:
            self._loop.call_soon_threadsafe(
                lambda: asyncio.create_task(publish_stream_update(payload))
            )

    def apply_global_config(self, max_fps=None, downscale=None, target_width=None, target_height=None) -> None:
        if max_fps: self._global_max_fps = max_fps
        if downscale: self._global_downscale = downscale
        self._global_target_width = target_width
        self._global_target_height = target_height

    def stop_all_streams(self, timeout=30.0) -> None:
        for stream_id in list(self._streams.keys()):
            try: self.stop_stream(stream_id, force=True)
            except: pass

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_streams": len(self._streams),
                "active_streams": sum(1 for c in self._streams.values() if c.state.is_active),
                "streams": {k: v.state.to_summary() for k, v in self._streams.items()}
            }

    def _handle_system_events(self, events: list[Event]) -> None:
        # [수정] 해상도 변경 이벤트가 와도 송출 해상도는 바꾸지 않음 (로그만 남김)
        for event in events:
            if event.type == EventType.SYSTEM_RESOLUTION_CHANGED:
                logger.info(
                    f"시스템 해상도 변경 감지: {event.details.get('width')}x{event.details.get('height')} "
                    "(송출 해상도는 1280x720으로 고정 유지됩니다)",
                    stream_id=event.stream_id
                )