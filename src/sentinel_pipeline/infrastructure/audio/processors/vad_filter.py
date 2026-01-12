"""
Voice Activity Detection (VAD) 필터

Silero VAD를 사용하여 사람 목소리가 포함된 오디오만 통과시킵니다.
조용한 구간이나 잡음만 있는 오디오는 GPU 처리 전에 필터링하여 성능을 향상시킵니다.

전처리 필터:
- High-Pass Filter: 300Hz 이하 저주파 제거 (바람 소리, 배경 노이즈 필터링)
  → 비명은 고주파에 집중되어 있으므로 저주파를 제거해도 영향 없음

Silero VAD 특징:
- 설치 간단: pip install silero-vad
- 정확도 높음: 6000개 이상 언어로 훈련
- 빠름: 30ms 청크를 1ms 미만에 처리
- 가벼움: 약 2MB
"""

import numpy as np
from typing import Optional

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None

try:
    from silero_vad import load_silero_vad, get_speech_timestamps
    SILERO_VAD_AVAILABLE = True
except ImportError:
    SILERO_VAD_AVAILABLE = False
    load_silero_vad = None
    get_speech_timestamps = None


class HighPassFilter:
    """
    High-Pass Filter (고주파 통과 필터)
    
    바람 소리나 배경 노이즈는 주로 저주파(300Hz 이하)에 있고,
    비명은 고주파에 집중되어 있으므로 저주파를 제거해도 비명 감지에는 영향 없음.
    """
    
    def __init__(self, cutoff_freq: float = 300.0, sample_rate: int = 16000):
        """
        Args:
            cutoff_freq: 차단 주파수 (Hz) - 이 주파수 이하는 제거
            sample_rate: 샘플링 레이트
        """
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is not installed. Install it with: pip install scipy"
            )
        
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        
        # Butterworth High-Pass Filter 설계
        # Nyquist frequency
        nyquist = sample_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist
        
        # 4차 Butterworth 필터 (충분히 가파른 차단)
        self.b, self.a = signal.butter(4, normal_cutoff, btype='high', analog=False)
    
    def apply(self, audio: np.ndarray) -> np.ndarray:
        """
        High-Pass Filter 적용
        
        각 오디오 청크는 독립적으로 필터링됩니다 (상태 유지 없음).
        
        Args:
            audio: 오디오 데이터 (numpy array, float32)
            
        Returns:
            필터링된 오디오 데이터
        """
        if len(audio) == 0:
            return audio
        
        # 필터 적용 (각 청크 독립적으로 처리)
        filtered = signal.lfilter(self.b, self.a, audio)
        
        return filtered.astype(np.float32)


class VADFilter:
    """
    Silero VAD 기반 음성 활동 감지 필터
    
    GPU에게 잡동사니를 던져주지 않기 위해 CPU에서 가볍게 동작하는 VAD로
    "사람 목소리가 아예 없는 구간"을 쳐냅니다.
    
    전처리: High-Pass Filter로 저주파 노이즈 제거 후 VAD 적용
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        use_highpass: bool = True,
        highpass_cutoff: float = 300.0,
    ):
        """
        Args:
            sample_rate: 오디오 샘플링 레이트 (8000 또는 16000 지원)
            threshold: 음성 판단 임계값 (0.0 ~ 1.0)
                - 높을수록 엄격 (잡음 많이 거름)
                - 낮을수록 관대 (비명도 통과)
                - 추천: 0.5 (중간)
            use_highpass: High-Pass Filter 사용 여부 (300Hz 이하 제거)
            highpass_cutoff: High-Pass Filter 차단 주파수 (Hz)
        """
        if not SILERO_VAD_AVAILABLE:
            raise ImportError(
                "silero-vad is not installed. Install it with: pip install silero-vad"
            )
        
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.use_highpass = use_highpass
        
        # High-Pass Filter 초기화
        self._highpass_filter = None
        if use_highpass:
            try:
                self._highpass_filter = HighPassFilter(
                    cutoff_freq=highpass_cutoff,
                    sample_rate=sample_rate
                )
            except ImportError:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning("scipy not available, High-Pass Filter disabled")
                self.use_highpass = False
        
        # Silero VAD 모델 로드 (lazy loading)
        self._model = None
        
        # 샘플링 레이트 검증
        if sample_rate not in [8000, 16000]:
            raise ValueError(
                f"Silero VAD supports only 8000 or 16000 Hz, got {sample_rate}"
            )

    def _load_model(self):
        """Silero VAD 모델 로드 (최초 1회)"""
        if self._model is None:
            try:
                self._model = load_silero_vad()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to load Silero VAD model: {e}")
                raise

    def is_speech(self, float_audio: np.ndarray) -> bool:
        """
        Float32 오디오를 받아서 사람 목소리 또는 비명이 포함되어 있는지 판단합니다.
        
        전처리: High-Pass Filter로 저주파 노이즈 제거 후 VAD 적용
        ⚠️ 주의: High-Pass Filter는 비명의 저주파 성분도 제거할 수 있어서 비명 감지에 방해될 수 있음
        
        Args:
            float_audio: 오디오 데이터 (numpy array, float32, -1.0 ~ 1.0)
            
        Returns:
            True: 음성 또는 비명이 포함되어 있음 (통과)
            False: 조용하거나 잡음만 있음 (차단)
        """
        if len(float_audio) == 0:
            return False
        
        # Step 0: 에너지 기반 사전 체크 (High-Pass Filter 전에)
        # 비명은 일반적으로 큰 소리이므로, 에너지가 높으면 일단 통과
        rms = np.sqrt(np.mean(float_audio ** 2))
        max_amplitude = np.max(np.abs(float_audio))
        
        # 에너지가 충분히 높으면 (비명일 가능성) High-Pass Filter 없이 바로 VAD 체크
        # 또는 High-Pass Filter를 건너뛰고 원본 오디오로 VAD 체크
        use_highpass_for_vad = self.use_highpass
        
        # Step 1: High-Pass Filter 적용 (저주파 노이즈 제거)
        # 단, 에너지가 높은 경우(비명 가능성)는 High-Pass Filter를 건너뛰거나 원본 사용
        if use_highpass_for_vad and self._highpass_filter is not None:
            # 에너지가 충분히 높으면 (비명 가능성) High-Pass Filter 건너뛰기
            if rms > 0.05 or max_amplitude > 0.2:
                # 비명일 가능성이 높으므로 High-Pass Filter 건너뛰기
                use_highpass_for_vad = False
        
        if use_highpass_for_vad and self._highpass_filter is not None:
            try:
                float_audio = self._highpass_filter.apply(float_audio)
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"High-Pass Filter error, using original audio: {e}")
        
        # Step 2: Silero VAD로 음성 감지
        self._load_model()
        
        try:
            # Silero VAD는 torch.Tensor를 입력으로 받음
            import torch
            
            # numpy array를 torch tensor로 변환
            if not isinstance(float_audio, torch.Tensor):
                audio_tensor = torch.from_numpy(float_audio).float()
            else:
                audio_tensor = float_audio.float()
            
            # get_speech_timestamps로 음성 구간 감지
            # threshold를 낮춰서 더 관대하게 판단 (비명도 통과시키기 위해)
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=max(0.3, self.threshold - 0.1),  # threshold를 낮춰서 더 관대하게
                sampling_rate=self.sample_rate,
                return_seconds=False,  # 샘플 인덱스 반환
            )
            
            # 음성 구간이 하나라도 있으면 통과
            if len(speech_timestamps) > 0:
                return True
            
            # VAD가 음성을 못 찾았지만, 에너지가 충분히 높으면 통과 (비명일 가능성)
            if rms > 0.05 or max_amplitude > 0.2:
                return True
            
            return False
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Silero VAD error, allowing audio through: {e}")
            # 오류 시 보수적으로 통과 (VAD 실패해도 처리 계속)
            return True

    def filter_audio(self, float_audio: np.ndarray) -> Optional[np.ndarray]:
        """
        오디오를 필터링합니다. 음성이 없으면 None을 반환합니다.
        
        Args:
            float_audio: 오디오 데이터
            
        Returns:
            np.ndarray: 음성이 있으면 원본 오디오 반환
            None: 음성이 없으면 None 반환 (차단)
        """
        if self.is_speech(float_audio):
            return float_audio
        return None


def create_vad_filter(
    sample_rate: int = 16000,
    threshold: float = 0.5,
    use_highpass: bool = True,
    highpass_cutoff: float = 300.0,
) -> Optional[VADFilter]:
    """
    Silero VAD 필터를 생성합니다.
    
    Args:
        sample_rate: 샘플링 레이트 (8000 또는 16000)
        threshold: 음성 판단 임계값 (0.0 ~ 1.0)
        use_highpass: High-Pass Filter 사용 여부 (300Hz 이하 제거)
        highpass_cutoff: High-Pass Filter 차단 주파수 (Hz)
        
    Returns:
        VADFilter 인스턴스 또는 None (실패 시)
    """
    if not SILERO_VAD_AVAILABLE:
        return None
    
    try:
        return VADFilter(
            sample_rate=sample_rate,
            threshold=threshold,
            use_highpass=use_highpass,
            highpass_cutoff=highpass_cutoff,
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to create Silero VAD filter: {e}")
        return None
