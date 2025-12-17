# -*- coding: utf-8 -*-
"""
FFmpeg 스트림 퍼블리셔.

FFmpeg 프로세스를 사용하여 처리된 프레임을 RTSP/RTMP로 출력합니다.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from queue import Queue, Full, Empty
from logging.handlers import RotatingFileHandler

import numpy as np
from numpy.typing import NDArray

from sentinel_pipeline.common.errors import ErrorCode, TransportError
from sentinel_pipeline.common.logging import get_logger


# 프레임 타입 정의
FrameType = NDArray[np.uint8]


class FFmpegPublisher:
    """
    FFmpeg 스트림 퍼블리셔.
    
    FFmpeg subprocess를 통해 영상을 RTSP/RTMP 서버로 출력합니다.
    """
    
    def __init__(
        self,
        stream_id: str,
        output_url: str,
        width: int,
        height: int,
        fps: int = 15,
        codec: str = "libx264",
        preset: str = "ultrafast",
        bitrate: str = "2M",
        pix_fmt: str = "yuv420p",
        log_stderr: bool = False,
        stderr_log_path: str | Path | None = None,
        max_stderr_log_size_mb: int = 10,
        stderr_log_backup_count: int = 3,
        frame_queue_size: int = 30,
        drop_policy: str = "drop_oldest",
    ):
        """
        FFmpegPublisher 초기화.
        
        Args:
            stream_id: 스트림 식별자 (로깅용)
            output_url: 출력 URL (rtsp:// 또는 rtmp://)
            width: 출력 프레임 너비
            height: 출력 프레임 높이
            fps: 출력 FPS
            codec: 비디오 코덱 (기본 libx264)
            preset: 인코딩 프리셋 (기본 ultrafast)
            bitrate: 출력 비트레이트 (기본 2M)
            pix_fmt: 픽셀 포맷 (기본 yuv420p)
            log_stderr: stderr를 로그 파일로 저장할지 여부 (기본 False)
            stderr_log_path: stderr 로그 파일 경로 (None이면 logs/ffmpeg_{stream_id}.log)
            max_stderr_log_size_mb: stderr 로그 파일 최대 크기 (MB, 기본 10MB)
            stderr_log_backup_count: stderr 로그 백업 파일 수 (기본 3)
            frame_queue_size: 프레임 큐 크기 (기본 30)
            drop_policy: 큐 가득 참 시 드롭 정책 ("drop_oldest" 또는 "drop_newest", 기본 drop_oldest)
        """
        self._stream_id = stream_id
        self._output_url = output_url
        self._width = width
        self._height = height
        self._fps = fps
        self._codec = codec
        self._preset = preset
        self._bitrate = bitrate
        self._pix_fmt = pix_fmt
        self._log_stderr = log_stderr
        self._stderr_log_path = Path(stderr_log_path) if stderr_log_path else None
        self._max_stderr_log_size_mb = max_stderr_log_size_mb
        self._stderr_log_backup_count = stderr_log_backup_count
        self._frame_queue_size = frame_queue_size
        self._drop_policy = drop_policy
        
        self._process: subprocess.Popen | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._error_count = 0
        self._dropped_frames = 0
        self._stderr_log_file = None
        self._stderr_handler: RotatingFileHandler | None = None
        
        # 프레임 큐 (인코더 지연 시 파이프라인 블로킹 방지)
        self._frame_queue: Queue[FrameType] = Queue(maxsize=frame_queue_size)
        self._queue_thread: threading.Thread | None = None
        self._queue_stop_event = threading.Event()
        
        self._logger = get_logger(__name__, stream_id=stream_id)
    
    @property
    def is_running(self) -> bool:
        """실행 상태 반환."""
        return self._running and self._process is not None and self._process.poll() is None
    
    @property
    def frame_count(self) -> int:
        """전송된 프레임 수."""
        return self._frame_count
    
    @property
    def error_count(self) -> int:
        """오류 수."""
        return self._error_count
    
    def start(self) -> bool:
        """
        FFmpeg 프로세스를 시작합니다.
        
        Returns:
            시작 성공 여부
            
        Raises:
            TransportError: 시작 실패 시
        """
        self._logger.info(
            "FFmpeg 퍼블리셔 시작",
            output_url=self._output_url,
            width=self._width,
            height=self._height,
            fps=self._fps,
        )
        
        with self._lock:
            if self._running:
                self._logger.warning("FFmpeg 퍼블리셔가 이미 실행 중")
                return True
            
            try:
                # FFmpeg 명령 구성
                cmd = self._build_ffmpeg_command()
                self._logger.debug("FFmpeg 명령", cmd=" ".join(cmd))
                
                # stderr 처리 결정 (순환 로그 사용)
                if self._log_stderr:
                    if self._stderr_log_path:
                        log_path = self._stderr_log_path
                    else:
                        # 기본 경로: logs/ffmpeg_{stream_id}.log
                        log_dir = Path("logs")
                        log_dir.mkdir(exist_ok=True)
                        log_path = log_dir / f"ffmpeg_{self._stream_id}.log"
                    
                    # 순환 로그 핸들러 생성
                    self._stderr_handler = RotatingFileHandler(
                        str(log_path),
                        maxBytes=self._max_stderr_log_size_mb * 1024 * 1024,
                        backupCount=self._stderr_log_backup_count,
                        encoding="utf-8",
                    )
                    stderr_handle = self._stderr_handler.stream
                    log_file = None
                else:
                    # 로그 없음 (데드락 방지)
                    stderr_handle = subprocess.DEVNULL
                    log_file = None
                    self._stderr_handler = None
                
                # FFmpeg 프로세스 시작
                # stdout는 항상 DEVNULL (데드락 방지)
                # stderr는 옵션에 따라 로그 파일 또는 DEVNULL
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_handle,
                    bufsize=10**8,  # 큰 버퍼 사용
                )
                
                # 로그 파일 핸들 저장 (나중에 닫기 위해)
                self._stderr_log_file = log_file
                
                self._running = True
                self._frame_count = 0
                self._error_count = 0
                self._dropped_frames = 0
                
                # 프레임 큐 처리 스레드 시작
                self._queue_stop_event.clear()
                self._queue_thread = threading.Thread(
                    target=self._frame_queue_worker,
                    name=f"ffmpeg_queue_{self._stream_id}",
                    daemon=True,
                )
                self._queue_thread.start()
                
                self._logger.info("FFmpeg 퍼블리셔 시작 완료")
                return True
                
            except FileNotFoundError:
                raise TransportError(
                    ErrorCode.TRANSPORT_FAILED,
                    "FFmpeg가 설치되어 있지 않습니다",
                    details={"stream_id": self._stream_id},
                )
            except Exception as e:
                self._logger.error("FFmpeg 시작 실패", error=str(e))
                raise TransportError(
                    ErrorCode.TRANSPORT_FAILED,
                    f"FFmpeg 시작 실패: {e}",
                    details={"stream_id": self._stream_id, "error": str(e)},
                )
    
    def write_frame(self, frame: FrameType) -> bool:
        """
        프레임을 큐에 추가합니다 (비동기 처리).
        
        Args:
            frame: BGR 형식의 numpy 배열
            
        Returns:
            성공 여부 (큐에 추가됨)
        """
        if not self.is_running:
            return False
        
        try:
            # 프레임 크기 확인 및 리사이즈
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                import cv2
                frame = cv2.resize(frame, (self._width, self._height))
            
            # 큐에 프레임 추가
            try:
                self._frame_queue.put_nowait(frame)
                return True
            except Full:
                # 큐가 가득 찬 경우 드롭 정책 적용
                if self._drop_policy == "drop_oldest":
                    try:
                        self._frame_queue.get_nowait()  # 오래된 프레임 제거
                        self._frame_queue.put_nowait(frame)
                        self._dropped_frames += 1
                        self._logger.debug("프레임 큐 가득 참, 오래된 프레임 드롭")
                        return True
                    except Empty:
                        pass
                else:
                    # drop_newest: 새 프레임 거부
                    self._dropped_frames += 1
                    self._logger.debug("프레임 큐 가득 참, 새 프레임 드롭")
                    return False
            
        except Exception as e:
            self._error_count += 1
            self._logger.error("프레임 큐 추가 실패", error=str(e))
            return False
    
    def _frame_queue_worker(self) -> None:
        """프레임 큐에서 프레임을 읽어 FFmpeg에 전송하는 워커 스레드."""
        while self._running and not self._queue_stop_event.is_set():
            try:
                # 큐에서 프레임 가져오기 (타임아웃으로 종료 확인 가능)
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # FFmpeg stdin에 프레임 쓰기
                if self._process and self._process.stdin:
                    try:
                        self._process.stdin.write(frame.tobytes())
                        self._frame_count += 1
                    except BrokenPipeError:
                        self._error_count += 1
                        self._logger.warning("FFmpeg 파이프 끊김")
                        self._running = False
                        break
                    except Exception as e:
                        self._error_count += 1
                        self._logger.error("프레임 쓰기 실패", error=str(e))
                        
            except Exception as e:
                self._logger.error("프레임 큐 워커 오류", error=str(e))
    
    def stop(self, timeout: float = 5.0) -> None:
        """
        FFmpeg 프로세스를 중지합니다.
        
        Args:
            timeout: 종료 대기 시간 (초)
        """
        self._logger.info("FFmpeg 퍼블리셔 중지")
        
        with self._lock:
            self._running = False
            
            if self._process is None:
                return
            
            try:
                # stdin 닫기 (FFmpeg에 종료 신호)
                if self._process.stdin:
                    self._process.stdin.close()
                
                # Graceful 종료 대기
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._logger.warning("FFmpeg 종료 타임아웃, 강제 종료")
                    self._process.kill()
                    self._process.wait(timeout=1.0)
                
            except Exception as e:
                self._logger.error("FFmpeg 중지 중 오류", error=str(e))
            finally:
                # 프레임 큐 워커 스레드 종료
                self._queue_stop_event.set()
                if self._queue_thread and self._queue_thread.is_alive():
                    self._queue_thread.join(timeout=2.0)
                
                # 큐에 남은 프레임 버리기
                while not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        break
                
                # 로그 파일/핸들러 닫기
                if self._stderr_log_file:
                    self._stderr_log_file.close()
                    self._stderr_log_file = None
                
                if self._stderr_handler:
                    self._stderr_handler.close()
                    self._stderr_handler = None
                
                self._process = None
                self._logger.info("FFmpeg 퍼블리셔 중지 완료")
    
    def restart(self) -> bool:
        """
        FFmpeg 프로세스를 재시작합니다.
        
        Returns:
            재시작 성공 여부
        """
        self._logger.info("FFmpeg 퍼블리셔 재시작")
        self.stop()
        time.sleep(0.5)
        return self.start()
    
    def _build_ffmpeg_command(self) -> list[str]:
        """FFmpeg 명령 구성."""
        cmd = [
            "ffmpeg",
            "-loglevel", "error",  # H.264 디코딩 경고 메시지 필터링 (error 레벨만 표시)
            "-y",  # 덮어쓰기
            "-f", "rawvideo",  # 입력 형식: raw video
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",  # OpenCV BGR 형식
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "-",  # stdin에서 입력
            "-c:v", self._codec,
            "-pix_fmt", self._pix_fmt,
            "-preset", self._preset,
            "-b:v", self._bitrate,
            "-maxrate", self._bitrate,
            "-bufsize", self._bitrate,
            "-g", str(self._fps * 2),  # GOP 크기 (2초)
            "-f", "rtsp",  # 출력 형식
            "-rtsp_transport", "tcp",
            self._output_url,
        ]
        return cmd
    
    def get_stats(self) -> dict:
        """통계 정보 반환."""
        return {
            "stream_id": self._stream_id,
            "output_url": self._output_url,
            "running": self.is_running,
            "frame_count": self._frame_count,
            "error_count": self._error_count,
            "dropped_frames": self._dropped_frames,
            "queue_size": self._frame_queue.qsize(),
            "queue_max_size": self._frame_queue_size,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
        }
    
    def __enter__(self) -> FFmpegPublisher:
        """컨텍스트 매니저 진입."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료."""
        self.stop()

