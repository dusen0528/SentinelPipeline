"""
위험 키워드 분석 프로세서

Faster-Whisper를 사용하여 음성을 텍스트로 변환하고 위험 키워드를 분석합니다.
3단계 하이브리드 아키텍처를 사용하여 STT 오인식과 한국어 교착어 특성을 처리합니다.
"""

import threading
from pathlib import Path

import numpy as np
import logging
from typing import Any, Optional

from sentinel_pipeline.domain.interfaces.audio_processor import AudioProcessor
from sentinel_pipeline.infrastructure.audio.processors.hybrid_keyword_detector import (
    HybridKeywordDetector,
)

logger = logging.getLogger(__name__)


# 기존 RiskKeywordMatcher는 하위 호환성을 위해 주석 처리
# 필요시 참조용으로 유지
"""
class RiskKeywordMatcher:
    # 기존 단순 문자열 매칭 구현 (하위 호환성 유지용)
    # HybridKeywordDetector로 대체됨
    pass
"""


class RiskAnalyzer(AudioProcessor):
    """
    STT 및 위험 분석 통합 프로세서
    
    3단계 하이브리드 키워드 감지 시스템을 사용합니다:
    - Fast Path: 해시맵 조회 (O(1))
    - Medium Path: 형태소 분석 + 편집 거리
    - Heavy Path: 의미 유사도 (비동기)
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        keyword_dict_path: str | Path | None = None,
        enable_medium_path: bool = True,
        enable_heavy_path: bool = True,
        heavy_path_async: bool = True,
        semantic_threshold: float = 0.7,
        use_korean_model: bool = False,
    ):
        """
        Args:
            model_size: Whisper 모델 크기
            device: 디바이스 ('auto', 'cuda', 'cpu')
            compute_type: 계산 타입 ('default', 'float16', 'int8')
            keyword_dict_path: 키워드 사전 JSON 파일 경로
            enable_medium_path: Medium Path 활성화 여부
            enable_heavy_path: Heavy Path 활성화 여부
            heavy_path_async: Heavy Path 비동기 처리 여부
            semantic_threshold: 의미 유사도 임계값
            use_korean_model: 한국어 특화 모델 사용 여부
        """
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
        
        # 하이브리드 키워드 감지기 초기화
        try:
            self.detector = HybridKeywordDetector(
                keyword_dict_path=keyword_dict_path,
                enable_medium_path=enable_medium_path,
                enable_heavy_path=enable_heavy_path,
                heavy_path_async=heavy_path_async,
                semantic_threshold=semantic_threshold,
                use_korean_model=use_korean_model,
            )
            logger.info("HybridKeywordDetector 초기화 완료")
        except Exception as e:
            logger.error(f"HybridKeywordDetector 초기화 실패: {e}")
            raise
        
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
                
            # 하이브리드 키워드 감지 (3단계 아키텍처)
            risk_result = self.detector.analyze(full_text)
            
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
