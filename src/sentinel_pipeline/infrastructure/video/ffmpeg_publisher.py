# -*- coding: utf-8 -*-
"""
FFmpeg 스트림 퍼블리셔.

FFmpeg 프로세스를 사용하여 처리된 프레임을 RTSP/RTMP로 출력합니다.
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from sentinel_pipeline.common.errors import ErrorCode, TransportError
from sentinel_pipeline.common.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
        
        self._process: subprocess.Popen | None = None
        self._running = False
        self._lock = threading.Lock()
        self._frame_count = 0
        self._error_count = 0
        
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
                
                # FFmpeg 프로세스 시작
                # stdout/stderr를 DEVNULL로 설정하여 파이프 데드락 방지
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8,  # 큰 버퍼 사용
                )
                
                self._running = True
                self._frame_count = 0
                self._error_count = 0
                
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
        프레임을 출력 스트림에 씁니다.
        
        Args:
            frame: BGR 형식의 numpy 배열
            
        Returns:
            성공 여부
        """
        if not self.is_running:
            return False
        
        try:
            # 프레임 크기 확인 및 리사이즈
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                import cv2
                frame = cv2.resize(frame, (self._width, self._height))
            
            # stdin에 프레임 쓰기
            self._process.stdin.write(frame.tobytes())  # type: ignore
            self._frame_count += 1
            return True
            
        except BrokenPipeError:
            self._error_count += 1
            self._logger.warning("FFmpeg 파이프 끊김")
            self._running = False
            return False
        except Exception as e:
            self._error_count += 1
            self._logger.error("프레임 쓰기 실패", error=str(e))
            return False
    
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

