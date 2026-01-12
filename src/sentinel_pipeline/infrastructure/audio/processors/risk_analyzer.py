"""
위험 키워드 분석 프로세서 (Producer-Consumer Pattern 적용)

- Producer: RiskAnalyzer (오디오 청크를 큐에 제출)
- Consumer: GlobalInferenceEngine (백그라운드에서 Whisper 추론 수행)
- Queue: Python native queue (가장 빠름)
"""

import threading
import queue
import logging
import time
from pathlib import Path
from typing import Any, Optional, Callable, Dict
from dataclasses import dataclass

import numpy as np

from sentinel_pipeline.domain.interfaces.audio_processor import AudioProcessor
from sentinel_pipeline.infrastructure.audio.processors.hybrid_keyword_detector import (
    HybridKeywordDetector,
)

logger = logging.getLogger(__name__)

# --- [Data Structure] ---
@dataclass
class InferenceRequest:
    """큐에 들어갈 작업 단위"""
    stream_id: str
    audio_data: np.ndarray
    callback: Optional[Callable[[Dict[str, Any]], None]]
    timestamp: float

# --- [Consumer: The GPU Worker] ---
class GlobalInferenceEngine:
    """
    [Singleton] 중앙 추론 엔진
    모델 로딩, 큐 관리, 백그라운드 추론을 전담합니다.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GlobalInferenceEngine, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # 1. 설정 (사령관님의 RTX 2000을 위해 Small 고정 추천)
        self.model_size = "small"
        self.device = "cuda"
        self.compute_type = "float16"
        
        # 2. 큐 생성 (Backpressure 조절용 maxsize 설정)
        # 너무 많이 쌓이면(100개 이상) 최신 데이터를 위해 오래된건 버리거나 입력을 막아야 함
        self.queue = queue.Queue(maxsize=100)
        self.running = True
        
        # 3. 모델 및 감지기 초기화 (Lazy Loading)
        self.model = None
        self.detector = None
        self._load_resources()

        # 4. 워커 스레드 시작 (소비자)
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="GPU-Inference-Worker")
        self.worker_thread.start()
        
        self._initialized = True
        logger.info(f"🚀 [GlobalInferenceEngine] 엔진 시동 완료 (Queue Size: 100)")

    def _load_resources(self):
        """모델과 키워드 감지기를 로딩합니다."""
        try:
            from faster_whisper import WhisperModel
            import torch
            
            # device 재확인
            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
                self.compute_type = "int8"

            logger.info(f"📥 [Engine] Whisper 모델 로딩 시작 ({self.model_size} / {self.device})...")
            self.model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type=self.compute_type
            )
            logger.info("✅ [Engine] Whisper 모델 로딩 완료")

            # 키워드 감지기는 RiskAnalyzer에서 설정을 받아야 하지만, 
            # Singleton 구조상 여기서 기본값으로 초기화하거나, 
            # 요청 시 detector를 인자로 받을 수도 있습니다. 
            # 여기서는 편의상 기본값으로 초기화합니다. (필요시 config 주입 구조로 변경 가능)
            self.detector = HybridKeywordDetector(
                enable_medium_path=True,
                enable_heavy_path=True,
                heavy_path_async=True,
                semantic_threshold=0.7,
                use_korean_model=False
            )
            logger.info("✅ [Engine] 키워드 감지기 준비 완료")

        except Exception as e:
            logger.error(f"❌ [Engine] 리소스 로딩 실패: {e}")
            raise e

    def submit(self, request: InferenceRequest):
        """[Producer Interface] 작업을 큐에 넣습니다. (Non-blocking)"""
        try:
            # 큐가 꽉 찼으면 즉시 에러 발생 (오래된 요청 대기시키지 않고 버림 -> 실시간성 유지)
            self.queue.put_nowait(request)
        except queue.Full:
            # 로깅은 너무 많이 찍힐 수 있으므로 샘플링하거나 debug 레벨로
            # logger.warning(f"⚠️ [Engine] 큐가 가득 찼습니다. 요청 드랍: {request.stream_id}")
            pass

    def _worker_loop(self):
        """[Consumer Loop] 큐에서 하나씩 꺼내 처리"""
        logger.info("🔧 [Worker] 추론 루프 시작")
        
        while self.running:
            try:
                # 큐에서 작업 가져오기 (대기)
                req = self.queue.get()
                
                # 처리 시작 시간
                start_time = time.time()
                
                # 1. 오디오 전처리 (여기서 수행하여 Main Thread 부하 감소)
                if req.audio_data.dtype != np.float32:
                    audio = req.audio_data.astype(np.float32)
                else:
                    audio = req.audio_data
                
                # 2. Whisper 추론 (GPU 사용)
                segments, _ = self.model.transcribe(
                    audio,
                    beam_size=1, # 속도 최적화
                    language="ko",
                    vad_filter=True, # Whisper 내부 VAD도 켜둠 (이중 안전장치)
                    vad_parameters=dict(min_silence_duration_ms=200, threshold=0.3)
                )
                
                full_text = " ".join([s.text for s in segments]).strip()
                
                # 3. 키워드 분석
                result_data = {
                    "text": full_text,
                    "is_dangerous": False,
                    "event_type": None,
                    "keyword": None,
                    "confidence": 0.0,
                    "stream_id": req.stream_id,
                    "latency": time.time() - req.timestamp
                }

                if full_text and self.detector:
                    analysis = self.detector.analyze(full_text)
                    result_data.update(analysis) # 결과 병합

                # 4. 콜백 실행 (결과 통보)
                if req.callback:
                    try:
                        req.callback(result_data)
                    except Exception as cb_err:
                        logger.error(f"❌ [Worker] 콜백 실행 중 에러: {cb_err}")

                # 작업 완료 표시
                self.queue.task_done()
                
                # (선택) 처리 속도 로깅
                # logger.debug(f"⚡ 처리완료: {req.stream_id} (len={len(full_text)}) time={time.time()-start_time:.3f}s")

            except Exception as e:
                logger.error(f"❌ [Worker] 치명적 오류: {e}")
                # 에러 발생 시에도 루프는 계속 돌아야 함

# --- [Producer: The Client] ---
class RiskAnalyzer(AudioProcessor):
    """
    이제 RiskAnalyzer는 직접 무거운 일을 하지 않습니다.
    엔진(GlobalInferenceEngine)에 작업을 던져주는 역할만 합니다.
    """
    
    def __init__(self, stream_id: str = "unknown", **kwargs):
        # **kwargs는 호환성을 위해 받아주지만, 실제로는 엔진이 관리합니다.
        self.stream_id = stream_id
        self.engine = GlobalInferenceEngine() # Singleton 인스턴스 획득

    def process(self, audio_data: np.ndarray, callback: Optional[Callable] = None) -> dict[str, Any]:
        """
        [비동기 처리 변경]
        이제 결과를 바로 반환하지 않습니다 (None 반환).
        대신 callback 함수를 통해 나중에 결과를 받습니다.
        """
        if len(audio_data) == 0:
            return {}

        # 1. 요청 객체 생성
        request = InferenceRequest(
            stream_id=self.stream_id,
            audio_data=np.copy(audio_data), # 데이터 복사 중요 (원본이 덮어써질 수 있음)
            callback=callback,
            timestamp=time.time()
        )
        
        # 2. 엔진에 제출 (Non-blocking)
        self.engine.submit(request)
        
        # 3. 즉시 리턴 (메인 스레드 해방!)
        # 기존 코드와의 호환성을 위해 빈 딕셔너리 반환하지만,
        # 호출하는 쪽(AudioManager)에서 리턴값을 기다리면 안 됩니다.
        return {
            "status": "queued",
            "is_dangerous": False # 임시 값
        }