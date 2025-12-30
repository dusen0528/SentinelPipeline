"""
RTSP 오디오 리더

FFmpeg를 사용하여 RTSP 스트림에서 오디오를 추출합니다.
"""

import threading
import time
import subprocess
import numpy as np
from scipy import signal
from typing import Callable, Optional
import logging
import shlex

from sentinel_pipeline.domain.interfaces.audio_processor import AudioReader

logger = logging.getLogger(__name__)


class RtspAudioReader(AudioReader):
    """
    RTSP 스트림에서 오디오를 추출하고 전처리(HPF)하여 청크 단위로 제공하는 클래스
    """
    
    def __init__(
        self,
        rtsp_url: str,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        hop_duration: float = 0.5,
        hpf_cutoff: float = 120.0,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None
    ):
        self.rtsp_url = rtsp_url
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.on_chunk = on_chunk
        
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.process: Optional[subprocess.Popen] = None
        
        # HPF 필터 설계
        nyquist = sample_rate / 2
        normal_cutoff = hpf_cutoff / nyquist
        self.b, self.a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        
        # 청크 크기 계산
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.hop_samples = int(sample_rate * hop_duration)
        
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def start(self) -> None:
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logger.info(f"RtspAudioReader started for {self.rtsp_url}")

    def stop(self) -> None:
        self.is_running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        if self.thread:
            self.thread.join(timeout=2.0)
            
        logger.info("RtspAudioReader stopped")

    def is_active(self) -> bool:
        return self.is_running and (self.thread is not None and self.thread.is_alive())

    def _process_audio_chunk(self, audio: np.ndarray):
        """오디오 데이터 전처리 및 버퍼링"""
        if len(audio) == 0:
            return
            
        # HPF 적용
        try:
            filtered_audio = signal.filtfilt(self.b, self.a, audio)
        except Exception:
            filtered_audio = audio
            
        self.audio_buffer = np.concatenate([self.audio_buffer, filtered_audio])
        
        # Chunk 단위 처리
        while len(self.audio_buffer) >= self.chunk_samples:
            chunk = self.audio_buffer[:self.chunk_samples]
            self.audio_buffer = self.audio_buffer[self.hop_samples:]
            
            if self.on_chunk:
                self.on_chunk(chunk)

    def _process_loop(self):
        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-vn',                    # 비디오 제외
            '-acodec', 'pcm_s16le',   # PCM 16-bit
            '-ar', str(self.sample_rate),
            '-ac', '1',               # Mono
            '-f', 's16le',            # Raw format
            '-'                       # Stdout
        ]
        
        # Windows 환경에서 ffmpeg 명령어 처리
        # 만약 ffmpeg가 PATH에 없다면 절대 경로가 필요할 수 있음
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, # 에러 로그 무시 (필요시 PIPE로 변경)
                bufsize=0
            )
            
            chunk_size = self.hop_samples * 2  # 16-bit = 2 bytes per sample
            
            while self.is_running:
                if self.process.poll() is not None:
                    logger.warning(f"FFmpeg process exited with code {self.process.returncode}")
                    break
                    
                raw_data = self.process.stdout.read(chunk_size)
                
                if not raw_data:
                    time.sleep(0.1)
                    continue
                    
                # int16 -> float32
                audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                
                self._process_audio_chunk(audio_float32)
                
        except Exception as e:
            logger.error(f"RTSP processing loop error: {e}")
        finally:
            if self.process:
                self.process.kill()
            self.is_running = False
