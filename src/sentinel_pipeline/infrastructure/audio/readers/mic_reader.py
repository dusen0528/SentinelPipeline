"""
마이크 오디오 리더

PyAudio를 사용하여 마이크 입력을 캡처합니다.
"""

import threading
import time
import numpy as np
from scipy import signal
from typing import Callable, Optional
import logging

try:
    import pyaudio
except ImportError:
    pyaudio = None

from sentinel_pipeline.domain.interfaces.audio_processor import AudioReader

logger = logging.getLogger(__name__)


class MicAudioReader(AudioReader):
    """
    마이크에서 오디오를 읽고 전처리(HPF)하여 청크 단위로 제공하는 클래스
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        hop_duration: float = 0.5,
        hpf_cutoff: float = 120.0,
        device_index: Optional[int] = None,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None
    ):
        if pyaudio is None:
            logger.warning("PyAudio not installed. Microphone input will not work.")
            self.pyaudio_available = False
        else:
            self.pyaudio_available = True
            
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.device_index = device_index
        self.on_chunk = on_chunk
        
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.audio = None
        self.stream = None
        self.init_error: Optional[Exception] = None
        self.init_complete = threading.Event()
        
        # HPF 필터 설계
        nyquist = sample_rate / 2
        normal_cutoff = hpf_cutoff / nyquist
        self.b, self.a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        
        # 청크 크기 계산
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.hop_samples = int(sample_rate * hop_duration)
        
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def start(self) -> None:
        if not self.pyaudio_available:
            error_msg = "Cannot start MicAudioReader: PyAudio is missing."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        if self.is_running:
            return
        
        # 초기화 상태 리셋
        self.init_error = None
        self.init_complete.clear()
            
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        # 초기화 완료 대기 (최대 2초)
        if self.init_complete.wait(timeout=2.0):
            if self.init_error:
                self.is_running = False
                raise self.init_error
            logger.info(f"MicAudioReader started (device_index={self.device_index})")
        else:
            # 타임아웃 - 초기화가 완료되지 않음
            logger.warning("MicAudioReader initialization timeout")
            # 스레드는 계속 실행 중일 수 있으므로 에러로 간주하지 않음

    def stop(self) -> None:
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        self._cleanup()
        logger.info("MicAudioReader stopped")

    def is_active(self) -> bool:
        return self.is_running and (self.thread is not None and self.thread.is_alive())

    def _cleanup(self):
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.debug(f"Error closing stream: {e}")
            self.stream = None
            
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.debug(f"Error terminating PyAudio: {e}")
            self.audio = None

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
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.hop_samples,
                stream_callback=None
            )
            self.stream.start_stream()
            
            # 초기화 성공 신호
            self.init_complete.set()
            
            while self.is_running and self.stream.is_active():
                try:
                    data = self.stream.read(self.hop_samples, exception_on_overflow=False)
                    # int16 -> float32
                    audio_int16 = np.frombuffer(data, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    
                    self._process_audio_chunk(audio_float32)
                except Exception as e:
                    logger.error(f"Error reading from mic: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Mic processing loop error: {e}")
            self.init_error = e
            self.init_complete.set()  # 에러가 발생해도 신호를 보내서 대기 중인 start()가 깨어나도록
        finally:
            self._cleanup()
