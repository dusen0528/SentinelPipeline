"""
위험 키워드 분석 프로세서

Faster-Whisper를 사용하여 음성을 텍스트로 변환하고 위험 키워드를 분석합니다.
"""

import threading
import numpy as np
import logging
from typing import Any, Optional
from difflib import SequenceMatcher

from sentinel_pipeline.domain.interfaces.audio_processor import AudioProcessor
from sentinel_pipeline.domain.models.event import EventType

logger = logging.getLogger(__name__)


class RiskKeywordMatcher:
    """
    텍스트에서 위험 키워드를 감지하고 이벤트 타입을 분류합니다.
    """
    
    def __init__(self):
        self.SIMILARITY_THRESHOLD = 0.6
        self.SINGLE_CHAR_KEYWORDS = {'불', '풀', '뿔', '꿀'}
        
        # 위험 키워드 매핑
        self.DANGER_MAPPING = {
            # 화재 (Fire)
            '불이야': EventType.FIRE, '뿌리야': EventType.FIRE, '풀이야': EventType.FIRE,
            '불났어요': EventType.FIRE, '불': EventType.FIRE,
            
            # 구조 요청 (Help) - KEYWORD_DETECTED로 매핑할 수도 있으나, EventType에 맞게
            '살려주세요': EventType.CUSTOM, '살려줘': EventType.CUSTOM, 
            '도와주세요': EventType.CUSTOM, '구해주세요': EventType.CUSTOM,
            
            # 범죄/위협 (Threat)
            '강도': EventType.INTRUSION, '칼': EventType.INTRUSION, '총': EventType.INTRUSION,
            '신고해': EventType.SYSTEM_ALERT,
            
            # 응급 의료 (Emergency)
            '119': EventType.SYSTEM_ALERT, '아파요': EventType.SYSTEM_ALERT, '구급차': EventType.SYSTEM_ALERT
        }
        
        # EventType에 없는 것은 CUSTOM 또는 가장 유사한 것으로 매핑
        # 기존 MODULE-SOUND의 'HELP', 'THREAT' 등은 EventType에 직접적으로 없으므로
        # 적절히 매핑하거나 EventType을 확장해야 함.
        # 여기서는 EventType.SYSTEM_ALERT 등을 활용하거나 CUSTOM 사용.

    def analyze(self, text: str) -> dict[str, Any]:
        if not text:
            return {
                'is_dangerous': False,
                'event_type': None,
                'keyword': None,
                'confidence': 0.0,
                'original_text': text
            }

        clean_text = text.replace(" ", "").replace(".", "").replace(",", "").replace("!", "")
        
        for wrong_pattern, event_type in self.DANGER_MAPPING.items():
            if wrong_pattern in clean_text:
                if len(wrong_pattern) == 1 and wrong_pattern in self.SINGLE_CHAR_KEYWORDS:
                    if clean_text.count(wrong_pattern) < 2:
                        continue

                return {
                    'is_dangerous': True,
                    'event_type': event_type,
                    'keyword': wrong_pattern,
                    'confidence': 1.0,
                    'original_text': text
                }

        return {
            'is_dangerous': False,
            'event_type': None,
            'keyword': None,
            'confidence': 0.0,
            'original_text': text
        }


class RiskAnalyzer(AudioProcessor):
    """
    STT 및 위험 분석 통합 프로세서
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default"
    ):
        try:
            from faster_whisper import WhisperModel
            import torch
        except ImportError:
            logger.error("faster_whisper not installed.")
            raise

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.compute_type = compute_type
        if self.compute_type == "default":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
            
        self.model_size = model_size
        self.model = None
        self._model_ready = False
        
        # 모델 비동기 로드
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()
        
        self.matcher = RiskKeywordMatcher()
        
    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Whisper model ({self.model_size}) on {self.device}...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            self._model_ready = True
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")

    def process(self, audio_data: np.ndarray) -> dict[str, Any]:
        if not self._model_ready or self.model is None:
            return {
                'text': '',
                'is_dangerous': False,
                'event_type': None
            }
            
        try:
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            audio_data = np.ascontiguousarray(audio_data)
            
            if np.abs(audio_data).max() > 1.0:
                audio_data = np.clip(audio_data, -1.0, 1.0)
                
            if len(audio_data) == 0:
                return {'text': '', 'is_dangerous': False, 'event_type': None}
                
            segments, _ = self.model.transcribe(
                audio_data,
                beam_size=5,
                language="ko",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=200, threshold=0.3),
                task="transcribe"
            )
            
            full_text_parts = [s.text for s in segments]
            full_text = " ".join(full_text_parts).strip()
            
            if not full_text:
                return {'text': '', 'is_dangerous': False, 'event_type': None}
                
            # 위험 분석
            risk_result = self.matcher.analyze(full_text)
            
            return {
                'text': full_text,
                'is_dangerous': risk_result['is_dangerous'],
                'event_type': risk_result['event_type'],
                'keyword': risk_result['keyword'],
                'confidence': risk_result['confidence']
            }
            
        except Exception as e:
            logger.error(f"STT Processing error: {e}")
            return {'text': '', 'is_dangerous': False, 'event_type': None}
