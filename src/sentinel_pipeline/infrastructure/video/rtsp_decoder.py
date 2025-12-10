# -*- coding: utf-8 -*-
"""
RTSP 스트림 디코더.

OpenCV를 사용하여 RTSP 스트림을 디코딩합니다.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

from sentinel_pipeline.common.errors import ErrorCode, StreamError
from sentinel_pipeline.common.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


# 프레임 타입 정의
FrameType = NDArray[np.uint8]


class RTSPDecoder:
    """
    RTSP 스트림 디코더.
    
    OpenCV의 VideoCapture를 사용하여 RTSP 스트림을 읽습니다.
    스레드 안전하지 않으므로 단일 스레드에서만 사용해야 합니다.
    """
    
    def __init__(
        self,
        stream_id: str,
        buffer_size: int = 2,
        connection_timeout_ms: int = 10000,
        read_timeout_ms: int = 5000,
    ):
        """
        RTSPDecoder 초기화.
        
        Args:
            stream_id: 스트림 식별자 (로깅용)
            buffer_size: OpenCV 버퍼 크기 (기본 2)
            connection_timeout_ms: 연결 타임아웃 (밀리초)
            read_timeout_ms: 프레임 읽기 타임아웃 (밀리초)
        """
        self._stream_id = stream_id
        self._buffer_size = buffer_size
        self._connection_timeout_ms = connection_timeout_ms
        self._read_timeout_ms = read_timeout_ms
        
        self._cap: cv2.VideoCapture | None = None
        self._rtsp_url: str | None = None
        self._connected = False
        self._lock = threading.Lock()
        
        # 스트림 정보
        self._width: int = 0
        self._height: int = 0
        self._fps: float = 0.0
        
        self._logger = get_logger(__name__, stream_id=stream_id)
    
    @property
    def is_connected(self) -> bool:
        """연결 상태 반환."""
        return self._connected and self._cap is not None and self._cap.isOpened()
    
    @property
    def width(self) -> int:
        """프레임 너비."""
        return self._width
    
    @property
    def height(self) -> int:
        """프레임 높이."""
        return self._height
    
    @property
    def fps(self) -> float:
        """스트림 FPS."""
        return self._fps
    
    def connect(self, rtsp_url: str) -> bool:
        """
        RTSP 스트림에 연결합니다.
        
        Args:
            rtsp_url: RTSP URL
            
        Returns:
            연결 성공 여부
            
        Raises:
            StreamError: 연결 실패 시
        """
        self._logger.info("RTSP 스트림 연결 시도", rtsp_url=self._mask_password(rtsp_url))
        
        with self._lock:
            # 기존 연결 해제
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            
            self._rtsp_url = rtsp_url
            self._connected = False
            
            try:
                # OpenCV VideoCapture 생성
                self._cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                
                # 버퍼 크기 설정
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
                
                # 타임아웃 설정 (밀리초) - OpenCV 빌드에 따라 미지원될 수 있음
                try:
                    self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._connection_timeout_ms)
                    self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_timeout_ms)
                except AttributeError:
                    self._logger.warning(
                        "OpenCV 타임아웃 설정 미지원, 기본값 사용"
                    )
                
                # 연결 확인
                if not self._cap.isOpened():
                    raise StreamError(
                        ErrorCode.STREAM_CONNECTION_FAILED,
                        f"RTSP 스트림 연결 실패: {self._mask_password(rtsp_url)}",
                        stream_id=self._stream_id,
                    )
                
                # 스트림 정보 읽기
                self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._fps = self._cap.get(cv2.CAP_PROP_FPS)
                
                # FPS가 0이면 기본값 설정
                if self._fps <= 0:
                    self._fps = 25.0
                
                self._connected = True
                self._logger.info(
                    "RTSP 스트림 연결 성공",
                    width=self._width,
                    height=self._height,
                    fps=self._fps,
                )
                return True
                
            except StreamError:
                raise
            except Exception as e:
                self._logger.error("RTSP 스트림 연결 중 오류 발생", error=str(e))
                raise StreamError(
                    ErrorCode.STREAM_CONNECTION_FAILED,
                    f"RTSP 스트림 연결 오류: {e}",
                    stream_id=self._stream_id,
                    details={"error": str(e)},
                )
    
    def read_frame(self) -> tuple[bool, FrameType | None]:
        """
        프레임을 읽습니다.
        
        Returns:
            (성공 여부, 프레임) 튜플.
            실패 시 (False, None) 반환.
        """
        if not self.is_connected:
            return False, None
        
        try:
            ret, frame = self._cap.read()  # type: ignore
            
            if not ret or frame is None:
                self._logger.warning("프레임 읽기 실패")
                return False, None
            
            return True, frame
            
        except Exception as e:
            self._logger.error("프레임 읽기 중 오류 발생", error=str(e))
            return False, None
    
    def grab_frame(self) -> bool:
        """
        프레임을 가져옵니다 (디코딩 없이).
        
        버퍼를 비우거나 프레임을 건너뛸 때 사용합니다.
        
        Returns:
            성공 여부
        """
        if not self.is_connected:
            return False
        
        try:
            return self._cap.grab()  # type: ignore
        except Exception:
            return False
    
    def release(self) -> None:
        """연결을 해제합니다."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            
            self._connected = False
            self._logger.info("RTSP 스트림 연결 해제")
    
    def reconnect(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> bool:
        """
        지수 백오프 방식으로 재연결을 시도합니다.
        
        Args:
            max_retries: 최대 재시도 횟수 (기본 5)
            base_delay: 초기 대기 시간 (초, 기본 1.0)
            max_delay: 최대 대기 시간 (초, 기본 30.0)
        
        Returns:
            연결 성공 여부
        """
        if self._rtsp_url is None:
            self._logger.error("재연결 실패: RTSP URL이 설정되지 않음")
            return False
        
        # 기존 연결 해제
        self.release()
        
        for attempt in range(max_retries):
            # 지수 백오프 계산
            delay = min(base_delay * (2 ** attempt), max_delay)
            
            self._logger.info(
                "RTSP 스트림 재연결 시도",
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=f"{delay:.1f}s",
            )
            
            time.sleep(delay)
            
            try:
                if self.connect(self._rtsp_url):
                    return True
            except StreamError as e:
                self._logger.warning(
                    "재연결 시도 실패",
                    attempt=attempt + 1,
                    error=str(e),
                )
        
        self._logger.error("RTSP 재연결 최대 횟수 초과", max_retries=max_retries)
        return False
    
    def _mask_password(self, url: str) -> str:
        """URL에서 비밀번호를 마스킹합니다."""
        import re
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)
    
    def __enter__(self) -> RTSPDecoder:
        """컨텍스트 매니저 진입."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료."""
        self.release()

