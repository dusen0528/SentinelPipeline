"""
스트림 관리자

RTSP 스트림의 생명주기(시작, 중지, 재시작)를 관리합니다.
"""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING, Any, Callable, Protocol
from enum import Enum

from sentinel_pipeline.common.logging import get_logger
from sentinel_pipeline.common.errors import StreamError, ErrorCode
from sentinel_pipeline.domain.models.stream import (
    StreamConfig,
    StreamState,
    StreamStatus,
)
import asyncio
from sentinel_pipeline.interface.api.ws_bus import publish_stream_update

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class DecoderProtocol(Protocol):
    """RTSP 디코더 인터페이스 (Infrastructure에서 구현)"""
    
    def connect(self, url: str) -> bool: ...
    def read_frame(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...
    def is_connected(self) -> bool: ...


class PublisherProtocol(Protocol):
    """FFmpeg 퍼블리셔 인터페이스 (Infrastructure에서 구현)"""
    
    def start(self, output_url: str, width: int, height: int, fps: int) -> None: ...
    def write_frame(self, frame: Any) -> bool: ...
    def stop(self) -> None: ...


class StreamContext:
    """
    스트림 컨텍스트
    
    개별 스트림의 실행 상태와 관련 객체를 관리합니다.
    
    Attributes:
        state: 스트림 상태
        decoder: RTSP 디코더 (Infrastructure에서 주입)
        publisher: FFmpeg 퍼블리셔 (Infrastructure에서 주입)
        thread: 처리 스레드
    """
    
    def __init__(self, config: StreamConfig) -> None:
        self.state = StreamState(config=config)
        self.decoder: DecoderProtocol | None = None
        self.publisher: PublisherProtocol | None = None
        self.thread: threading.Thread | None = None
        self._stop_event = threading.Event()
    
    @property
    def stream_id(self) -> str:
        return self.state.stream_id
    
    def request_stop(self) -> None:
        """중지 요청"""
        self._stop_event.set()
    
    def should_stop(self) -> bool:
        """중지 요청 확인"""
        return self._stop_event.is_set()
    
    def clear_stop(self) -> None:
        """중지 요청 초기화"""
        self._stop_event.clear()


class StreamManager:
    """
    스트림 관리자
    
    여러 RTSP 스트림의 생명주기를 관리합니다.
    스레드 안전하게 동시 접근을 처리합니다.
    
    Attributes:
        decoder_factory: 디코더 생성 팩토리 함수
        publisher_factory: 퍼블리셔 생성 팩토리 함수
        frame_callback: 프레임 처리 콜백
    
    Example:
        >>> manager = StreamManager()
        >>> manager.set_decoder_factory(lambda: RTSPDecoder())
        >>> manager.set_frame_callback(pipeline.process_frame)
        >>> 
        >>> manager.start_stream("cam_01", "rtsp://...")
        >>> manager.stop_stream("cam_01")
    """
    
    # 종료 타임아웃 (초)
    STOP_TIMEOUT_SECONDS = 10.0
    JOIN_TIMEOUT_SECONDS = 5.0
    
    def __init__(self) -> None:
        """스트림 관리자 초기화"""
        self._streams: dict[str, StreamContext] = {}
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        
        # 팩토리 함수 (Infrastructure에서 설정)
        self._decoder_factory: Callable[[], DecoderProtocol] | None = None
        self._publisher_factory: Callable[[], PublisherProtocol] | None = None
        
        # 프레임 처리 콜백
        self._frame_callback: Callable[[Any, dict[str, Any]], tuple[Any, list[Any]]] | None = None
        
        # 상태 변경 콜백
        self._on_status_change: Callable[[str, StreamStatus], None] | None = None
        
        # 전역 설정
        self._global_max_fps = 15
        self._global_downscale = 1.0

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """이벤트 루프를 설정합니다."""
        self._loop = loop
    
    
    def set_decoder_factory(
        self,
        factory: Callable[..., DecoderProtocol],
    ) -> None:
        """디코더 팩토리를 설정합니다."""
        self._decoder_factory = factory
    
    def set_publisher_factory(
        self,
        factory: Callable[..., PublisherProtocol],
    ) -> None:
        """퍼블리셔 팩토리를 설정합니다."""
        self._publisher_factory = factory
    
    def set_frame_callback(
        self,
        callback: Callable[[Any, dict[str, Any]], tuple[Any, list[Any]]],
    ) -> None:
        """프레임 처리 콜백을 설정합니다."""
        self._frame_callback = callback
    
    def set_on_status_change(
        self,
        callback: Callable[[str, StreamStatus], None] | None,
    ) -> None:
        """상태 변경 콜백을 설정합니다."""
        self._on_status_change = callback
    
    def start_stream(
        self,
        stream_id: str,
        rtsp_url: str,
        **kwargs: Any,
    ) -> StreamState:
        """
        스트림을 시작합니다.
        
        Args:
            stream_id: 스트림 ID
            rtsp_url: RTSP URL
            **kwargs: 추가 설정 (max_fps, downscale 등)
        
        Returns:
            스트림 상태
        
        Raises:
            StreamError: 이미 실행 중이거나 시작 실패 시
        """
        with self._lock:
            # 이미 존재하는지 확인
            if stream_id in self._streams:
                ctx = self._streams[stream_id]
                if ctx.state.is_active:
                    raise StreamError(
                        ErrorCode.STREAM_ALREADY_RUNNING,
                        f"스트림이 이미 실행 중입니다: {stream_id}",
                        stream_id=stream_id,
                    )
            
            # 설정 생성
            downscale = kwargs.get("downscale", self._global_downscale)
            if downscale is None:
                downscale = self._global_downscale

            config = StreamConfig(
                stream_id=stream_id,
                rtsp_url=rtsp_url,
                max_fps=kwargs.get("max_fps", self._global_max_fps),
                downscale=downscale,
                output_url=kwargs.get("output_url"),
            )
            
            # 컨텍스트 생성
            ctx = StreamContext(config)
            ctx.state.set_status(StreamStatus.STARTING)
            self._streams[stream_id] = ctx
            
            logger.info(
                f"스트림 시작: {stream_id}",
                stream_id=stream_id,
                rtsp_url=StreamConfig._mask_url(rtsp_url),
            )
            
            # 처리 스레드 시작
            ctx.thread = threading.Thread(
                target=self._stream_loop,
                args=(ctx,),
                name=f"stream_{stream_id}",
                daemon=True,
            )
            ctx.thread.start()
            
            self._notify_status_change(stream_id, StreamStatus.STARTING)
            
            return ctx.state
    
    def stop_stream(self, stream_id: str, force: bool = False) -> bool:
        """
        스트림을 중지합니다.
        
        Args:
            stream_id: 스트림 ID
            force: 강제 종료 여부
        
        Returns:
            성공 여부
        
        Raises:
            StreamError: 스트림이 존재하지 않을 때
        """
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
        
        # 락 밖에서 스레드 종료 대기
        if ctx.thread and ctx.thread.is_alive():
            ctx.thread.join(timeout=self.JOIN_TIMEOUT_SECONDS)
            
            if ctx.thread.is_alive() and force:
                logger.warning(
                    f"스트림 스레드 강제 종료: {stream_id}",
                    stream_id=stream_id,
                )
                # Python에서는 스레드를 강제 종료할 수 없음
                # 데몬 스레드이므로 프로세스 종료 시 함께 종료됨
        
        # 리소스 정리
        self._cleanup_stream(ctx)
        
        with self._lock:
            ctx.state.set_status(StreamStatus.STOPPED)
            self._notify_status_change(stream_id, StreamStatus.STOPPED)
        
        return True
    
    def restart_stream(self, stream_id: str) -> StreamState:
        """
        스트림을 재시작합니다.
        
        Args:
            stream_id: 스트림 ID
        
        Returns:
            스트림 상태
        """
        with self._lock:
            ctx = self._streams.get(stream_id)
            if not ctx:
                raise StreamError(
                    ErrorCode.STREAM_NOT_FOUND,
                    f"스트림을 찾을 수 없습니다: {stream_id}",
                    stream_id=stream_id,
                )
            
            config = ctx.state.config
        
        # 중지 후 시작
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
        )
    
    def get_stream_state(self, stream_id: str) -> StreamState | None:
        """스트림 상태를 반환합니다."""
        ctx = self._streams.get(stream_id)
        return ctx.state if ctx else None
    
    def get_all_streams(self) -> list[StreamState]:
        """모든 스트림 상태를 반환합니다."""
        with self._lock:
            return [ctx.state for ctx in self._streams.values()]
    
    def get_active_streams(self) -> list[StreamState]:
        """활성 스트림 상태만 반환합니다."""
        with self._lock:
            return [
                ctx.state for ctx in self._streams.values()
                if ctx.state.is_active
            ]
    
    def _stream_loop(self, ctx: StreamContext) -> None:
        """
        스트림 처리 루프
        
        별도 스레드에서 실행되어 프레임을 읽고 처리합니다.
        """
        stream_id = ctx.stream_id
        config = ctx.state.config
        
        try:
            # 디코더 생성 및 연결
            if self._decoder_factory:
                # 팩토리가 stream_id를 받도록 시도, 아니면 무인자 호출
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
            
            # 퍼블리셔 생성 및 시작 (출력 URL이 설정된 경우)
            if self._publisher_factory and config.output_enabled and config.output_url:
                try:
                    pub_fps = int(config.max_fps or ctx.decoder.fps or 25)
                    width = ctx.decoder.width
                    height = ctx.decoder.height

                    # 팩토리가 인자를 받도록 시도, 아니면 무인자 호출 후 start()는 이미 init에서 처리된다고 가정
                    try:
                        ctx.publisher = self._publisher_factory(
                            stream_id, config.output_url, width, height, pub_fps
                        )  # type: ignore[arg-type]
                    except TypeError:
                        ctx.publisher = self._publisher_factory()  # type: ignore[call-arg]

                    # 퍼블리셔 시작 (FFmpeg 프로세스 기동)
                    ctx.publisher.start()
                    logger.info(
                        f"퍼블리셔 시작: {stream_id} -> {config.output_url} "
                        f"({width}x{height}@{pub_fps})",
                        stream_id=stream_id,
                        output_url=config.output_url,
                    )
                except Exception as e:  # 시작 실패 시 스트림 오류 처리
                    logger.error(
                        f"퍼블리셔 시작 실패: {stream_id} - {e}",
                        stream_id=stream_id,
                        error=str(e),
                    )
                    raise
            
            # 상태 업데이트
            ctx.state.set_status(StreamStatus.RUNNING)
            ctx.state.reset_retry()
            self._notify_status_change(stream_id, StreamStatus.RUNNING)
            
            logger.info(f"스트림 연결 성공: {stream_id}", stream_id=stream_id)
            
            # 프레임 간격 계산
            frame_interval = 1.0 / config.max_fps
            last_frame_time = 0.0
            frame_number = 0
            consecutive_failures = 0  # 연속 실패 횟수
            MAX_CONSECUTIVE_FAILURES = 5  # 5회 연속 실패 시 재연결 시도
            
            while not ctx.should_stop():
                # FPS 제한
                now = time.time()
                elapsed = now - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue
                
                last_frame_time = time.time()
                
                # 프레임 읽기
                success, frame = ctx.decoder.read_frame()
                if not success or frame is None:
                    consecutive_failures += 1
                    ctx.state.stats.record_error()
                    
                    # 연결 상태 확인 (OpenCV가 여전히 열려있다고 해도 실제로는 끊겼을 수 있음)
                    is_actually_connected = ctx.decoder.is_connected
                    
                    # 연속 실패가 일정 횟수 이상이거나, 연결이 명확히 끊긴 경우 재연결 시도
                    if not is_actually_connected or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning(
                            f"프레임 읽기 실패 감지: {stream_id} (연속 {consecutive_failures}회, 연결 상태: {is_actually_connected})",
                            stream_id=stream_id,
                            consecutive_failures=consecutive_failures,
                            is_connected=is_actually_connected,
                        )
                        # 재연결 시도 (최대 4회: 1, 2, 4, 8초)
                        if self._should_reconnect(ctx):
                            reconnect_success = self._attempt_reconnect(ctx)
                            if reconnect_success:
                                # 재연결 성공 시에만 카운터 리셋
                                consecutive_failures = 0
                            # 재연결 실패 시 consecutive_failures는 유지되어 다음 루프에서 다시 재연결 시도
                        else:
                            # 최대 재시도 횟수 초과 - 스트림 종료
                            logger.error(
                                f"스트림 재연결 실패 (최대 재시도 횟수 초과): {stream_id}",
                                stream_id=stream_id,
                                retry_count=ctx.state.retry_count,
                            )
                            ctx.state.set_status(
                                StreamStatus.ERROR,
                                f"재연결 실패: 최대 재시도 횟수(4회) 초과"
                            )
                            self._notify_status_change(stream_id, StreamStatus.ERROR)
                            break  # 스트림 루프 종료
                    continue
                
                # 프레임 읽기 성공 - 연속 실패 카운터 리셋
                consecutive_failures = 0
                
                # 프레임 읽기 성공 시 재연결 중이었다면 RUNNING으로 복구
                if ctx.state.status == StreamStatus.RECONNECTING:
                    ctx.state.set_status(StreamStatus.RUNNING)
                    ctx.state.reset_retry()
                    self._notify_status_change(stream_id, StreamStatus.RUNNING)
                    logger.info(f"스트림 재연결 성공 (프레임 수신 재개): {stream_id}", stream_id=stream_id)
                    
                    # 퍼블리셔 재시작 (VLC 등 클라이언트가 재연결할 수 있도록)
                    if ctx.publisher and config.output_enabled:
                        try:
                            logger.info(f"퍼블리셔 재시작: {stream_id}", stream_id=stream_id)
                            ctx.publisher.stop(timeout=2.0)
                            # 프레임 큐 비우기
                            time.sleep(0.5)  # FFmpeg 프로세스 완전 종료 대기
                            ctx.publisher.start()
                            logger.info(f"퍼블리셔 재시작 완료: {stream_id}", stream_id=stream_id)
                        except Exception as e:
                            logger.error(
                                f"퍼블리셔 재시작 실패: {stream_id} - {e}",
                                stream_id=stream_id,
                                error=str(e),
                            )
                            # 퍼블리셔 재시작 실패해도 스트림은 계속 진행
                
                frame_number += 1
                
                # 메타데이터 구성
                metadata = {
                    "stream_id": stream_id,
                    "frame_number": frame_number,
                    "timestamp": time.time(),
                    "fps": ctx.state.stats.fps,
                }
                
                # 프레임 처리 콜백 호출
                start_time = time.perf_counter()
                
                if self._frame_callback:
                    try:
                        processed_frame, events = self._frame_callback(frame, metadata)
                        
                        # 통계 업데이트
                        latency_ms = (time.perf_counter() - start_time) * 1000
                        ctx.state.stats.record_frame(latency_ms)
                        ctx.state.stats.record_event(len(events))
                        
                        # 퍼블리셔로 출력
                        if ctx.publisher and config.output_enabled:
                            ctx.publisher.write_frame(processed_frame)
                            
                    except Exception as e:
                        logger.error(
                            f"프레임 처리 오류: {stream_id} - {e}",
                            stream_id=stream_id,
                            error=str(e),
                        )
                        ctx.state.stats.record_error()
                else:
                    # 콜백 없으면 통계만 기록
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    ctx.state.stats.record_frame(latency_ms)
        
        except StreamError as e:
            logger.error(f"스트림 오류: {e}", stream_id=stream_id)
            ctx.state.set_status(StreamStatus.ERROR, str(e))
        
        except Exception as e:
            logger.exception(f"스트림 루프 예외: {stream_id}", stream_id=stream_id)
            ctx.state.set_status(StreamStatus.ERROR, str(e))
        
        finally:
            self._cleanup_stream(ctx)
            logger.info(f"스트림 루프 종료: {stream_id}", stream_id=stream_id)
    
    def _should_reconnect(self, ctx: StreamContext) -> bool:
        """
        재연결을 시도해야 하는지 확인합니다.
        
        재연결이 활성화되어 있고, 최대 재시도 횟수를 넘지 않았으면 재연결을 시도합니다.
        """
        if ctx.should_stop():
            return False
        if not ctx.state.config.reconnect_enabled:
            return False
        # 최대 재시도 횟수 확인
        return ctx.state.retry_count < ctx.state.config.reconnect_max_retries
    
    def _attempt_reconnect(self, ctx: StreamContext) -> bool:
        """
        재연결을 시도합니다.
        
        지수 백오프 방식으로 재연결을 시도합니다 (1, 2, 4, 8초).
        재연결 성공 시 RUNNING 상태로 복구하고, 실패 시에도 계속 재시도합니다.
        
        Returns:
            재연결 성공 여부
        """
        stream_id = ctx.stream_id
        config = ctx.state.config
        
        # RECONNECTING 상태로 변경 (아직 변경되지 않은 경우만)
        if ctx.state.status != StreamStatus.RECONNECTING:
            ctx.state.set_status(StreamStatus.RECONNECTING)
            self._notify_status_change(stream_id, StreamStatus.RECONNECTING)
        
        # 재시도 카운터 증가 (최대 4회까지만)
        ctx.state.record_retry()
        
        delay = ctx.state.calculate_next_retry_delay()
        logger.info(
            f"스트림 재연결 시도: {stream_id} (시도 {ctx.state.retry_count}, 대기 {delay:.1f}초)",
            stream_id=stream_id,
            retry_count=ctx.state.retry_count,
            delay=delay,
        )
        
        # 대기 (중지 요청 확인하면서)
        wait_end = time.time() + delay
        while time.time() < wait_end:
            if ctx.should_stop():
                return False
            time.sleep(0.1)
        
        # 디코더 재연결 시도
        if ctx.decoder:
            try:
                ctx.decoder.release()
            except Exception as e:
                logger.warning(f"디코더 해제 중 오류: {stream_id} - {e}", stream_id=stream_id)
            
            try:
                if ctx.decoder.connect(config.rtsp_url):
                    # 재연결 성공
                    ctx.state.set_status(StreamStatus.RUNNING)
                    ctx.state.reset_retry()
                    self._notify_status_change(stream_id, StreamStatus.RUNNING)
                    logger.info(f"스트림 재연결 성공: {stream_id}", stream_id=stream_id)
                    return True
                else:
                    # 재연결 실패 - 다음 루프에서 다시 시도
                    logger.warning(
                        f"스트림 재연결 실패: {stream_id} (다음 시도 대기 중)",
                        stream_id=stream_id,
                        retry_count=ctx.state.retry_count,
                    )
                    return False
            except Exception as e:
                # 연결 시도 중 예외 발생
                logger.warning(
                    f"스트림 재연결 시도 중 오류: {stream_id} - {e}",
                    stream_id=stream_id,
                    error=str(e),
                )
                return False
        
        return False
    
    def delete_stream(self, stream_id: str) -> bool:
        """
        스트림을 삭제합니다.
        
        스트림이 실행 중이면 먼저 중지한 후 삭제합니다.
        
        Args:
            stream_id: 스트림 ID
            
        Returns:
            성공 여부
            
        Raises:
            StreamError: 스트림이 존재하지 않을 때
        """
        with self._lock:
            ctx = self._streams.get(stream_id)
            if not ctx:
                raise StreamError(
                    ErrorCode.STREAM_NOT_FOUND,
                    f"스트림을 찾을 수 없습니다: {stream_id}",
                    stream_id=stream_id,
                )
        
        # 실행 중이면 먼저 중지
        if ctx.state.is_active:
            logger.info(f"스트림 삭제 전 중지: {stream_id}", stream_id=stream_id)
            self.stop_stream(stream_id)
        
        # 리소스 정리
        self._cleanup_stream(ctx)
        
        # 스트림 제거
        with self._lock:
            del self._streams[stream_id]
            logger.info(f"스트림 삭제 완료: {stream_id}", stream_id=stream_id)
        
        return True
    
    def _cleanup_stream(self, ctx: StreamContext) -> None:
        """스트림 리소스를 정리합니다."""
        stream_id = ctx.stream_id
        
        if ctx.decoder:
            try:
                ctx.decoder.release()
            except Exception as e:
                logger.warning(f"디코더 해제 오류: {stream_id} - {e}", stream_id=stream_id)
            ctx.decoder = None
        
        if ctx.publisher:
            try:
                ctx.publisher.stop()
            except Exception as e:
                logger.warning(f"퍼블리셔 종료 오류: {stream_id} - {e}", stream_id=stream_id)
            ctx.publisher = None
    
    def _notify_status_change(self, stream_id: str, status: StreamStatus) -> None:
        """상태 변경을 알립니다."""
        if self._on_status_change:
            try:
                self._on_status_change(stream_id, status)
            except Exception as e:
                logger.error(f"상태 변경 콜백 오류: {e}")

        # WebSocket 브로드캐스트 (non-blocking)
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
            try:
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(publish_stream_update(payload))
                )
            except RuntimeError as e:
                logger.error(f"WS publish_stream_update failed: {e}")
        else:
            logger.debug("No event loop; skip stream_update")
    
    def apply_global_config(
        self,
        max_fps: int | None = None,
        downscale: float | None = None,
    ) -> None:
        """
        전역 설정을 적용합니다.
        
        Args:
            max_fps: 최대 FPS
            downscale: 다운스케일 비율
        """
        if max_fps is not None:
            self._global_max_fps = max_fps
        if downscale is not None:
            self._global_downscale = downscale
        
        logger.info(
            f"전역 설정 변경: max_fps={self._global_max_fps}, downscale={self._global_downscale}"
        )
    
    def stop_all_streams(self, timeout: float = 30.0) -> None:
        """
        모든 스트림을 중지합니다.
        
        Args:
            timeout: 전체 타임아웃 (초)
        """
        stream_ids = list(self._streams.keys())
        
        logger.info(f"모든 스트림 중지: {len(stream_ids)}개")
        
        for stream_id in stream_ids:
            try:
                self.stop_stream(stream_id, force=True)
            except Exception as e:
                logger.error(f"스트림 중지 오류: {stream_id} - {e}")
        
        logger.info("모든 스트림 중지 완료")
    
    def get_stats(self) -> dict[str, Any]:
        """관리자 통계를 반환합니다."""
        with self._lock:
            active_count = sum(
                1 for ctx in self._streams.values()
                if ctx.state.is_active
            )
            
            return {
                "total_streams": len(self._streams),
                "active_streams": active_count,
                "streams": {
                    stream_id: ctx.state.to_summary()
                    for stream_id, ctx in self._streams.items()
                },
            }
