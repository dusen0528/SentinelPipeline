"""
GPU 부하 테스트 시뮬레이터

실제 오디오 파일을 사용하여 GPU 처리 용량을 측정합니다.

파이프라인 흐름:
1. Input: N개의 오디오 스트림 (sample_data/ 폴더의 실제 오디오 파일)
2. Step 1 (Scream Detector): ResNet18 모델로 비명 감지 (GPU 상시 부하)
3. Step 2 (Logic): 비명이 아닌 경우만 STT 실행
4. Step 3 (STT Pipeline): Whisper 모델로 텍스트 변환 → 키워드 분석
5. Output: 지연 시간(Latency)과 GPU/CPU 리소스 사용량 측정 (매 청크별 기록)

사용법:
    python -m sentinel_pipeline.scripts.benchmark_runner --streams 10
    python -m sentinel_pipeline.scripts.benchmark_runner --streams 10 --continuous --duration 30
"""

import csv
import gc
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import librosa
import numpy as np
import psutil
import torch

# 로깅 설정
from sentinel_pipeline.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamMetrics:
    """단일 스트림의 처리 결과 메트릭 (매 청크별 리소스 포함)"""
    stream_id: int
    step1_latency: float  # 비명 감지 시간 (초)
    step2_latency: float  # STT 변환 시간 (초, 실행 안 했으면 0)
    total_latency: float  # 전체 처리 시간 (초)
    detected: bool        # 비명 감지 여부
    scream_prob: float = 0.0  # 비명 확률
    transcript: str = ""  # STT 결과 텍스트
    audio_file: str = ""  # 사용된 오디오 파일명
    audio_category: str = "normal"  # Ground Truth 카테고리 (scream | emergency_keyword | normal)
    # 청크 처리 시점의 리소스 사용량 (시계열 분석용)
    gpu_memory_mb: float = 0.0  # 해당 청크 처리 시점의 GPU 메모리 (MB)
    cpu_percent: float = 0.0    # 해당 청크 처리 시점의 CPU 사용률 (%)
    system_memory_mb: float = 0.0  # 해당 청크 처리 시점의 시스템 메모리 (RAM, MB)


@dataclass
class BenchmarkResult:
    """벤치마크 테스트 결과"""
    streams: int
    avg_latency: float
    max_latency: float
    min_latency: float
    fps: float
    gpu_memory_mb: float
    gpu_memory_peak_mb: float
    cpu_percent: float
    device: str
    scream_count: int
    stt_count: int
    total_time: float
    duration: float = 0.0  # 테스트 지속 시간 (초)
    total_processed: int = 0  # 총 처리된 오디오 청크 수
    details: list = field(default_factory=list)
    
    def save_to_csv(self, filepath: Optional[str] = None) -> str:
        """
        결과를 CSV 파일로 저장
        
        Args:
            filepath: 저장 경로 (None이면 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"benchmark_result_{timestamp}.csv"
        
        filepath = Path(filepath)
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # 헤더 작성
            writer.writerow([
                "stream_id",
                "chunk_id",             # Continuous 모드에서 몇 번째 청크인지
                "timestamp",            # 처리 시점 (Unix timestamp)
                "audio_file",
                "audio_category",       # Ground Truth (GT): scream | emergency_keyword | normal
                "detected",             # 모델 예측 (비명 감지 여부)
                "scream_prob",          # 비명 확률 (0~1)
                "step1_latency_ms",     # 비명 감지 시간 (ResNet)
                "step2_latency_ms",     # STT 변환 시간 (Whisper)
                "total_latency_ms",     # 전체 처리 시간
                "gpu_memory_mb",        # 청크 처리 시점 GPU 메모리 (시계열)
                "cpu_percent",          # 청크 처리 시점 CPU 사용률 (시계열)
                "system_memory_mb",     # 청크 처리 시점 시스템 메모리 (RAM, 시계열)
                "transcript",           # STT 추출 텍스트
            ])
            
            # 데이터 작성
            for d in self.details:
                writer.writerow([
                    d.get("stream_id", ""),
                    d.get("chunk_id", 0),
                    d.get("timestamp", ""),
                    d.get("audio_file", ""),
                    d.get("audio_category", "normal"),
                    d.get("detected", False),
                    d.get("scream_prob", 0),
                    d.get("step1_latency", 0),
                    d.get("step2_latency", 0),
                    d.get("total_latency", 0),
                    d.get("gpu_memory_mb", 0),
                    d.get("cpu_percent", 0),
                    d.get("system_memory_mb", 0),
                    d.get("transcript", ""),
                ])
        
        return str(filepath)


class LoadTestSimulator:
    """
    GPU 부하 테스트 시뮬레이터
    
    실제 오디오 파일(sample_data/)을 사용하여 ScreamDetector + Whisper STT 파이프라인의
    처리 용량을 측정합니다. 매 청크 처리 시점의 GPU/CPU 리소스도 기록합니다.
    """
    
    # 오디오 설정
    SAMPLE_RATE = 16000
    WINDOW_SEC = 2.0  # 2초 윈도우
    
    def __init__(
        self,
        num_streams: int = 1,
        gpu_enabled: bool = True,
        scream_threshold: float = 0.7,
        whisper_model: str = "base",
        model_path: Optional[str] = None,
        sample_data_path: Optional[str] = None,
    ):
        """
        Args:
            num_streams: 시뮬레이션할 스트림 개수
            gpu_enabled: GPU 사용 여부
            scream_threshold: 비명 판정 임계값 (이 값 초과 시 비명으로 판정)
            whisper_model: Whisper 모델 크기 (tiny, base, small, medium, large)
            model_path: ScreamDetector 모델 경로 (None이면 기본 경로 사용)
            sample_data_path: 샘플 오디오 파일 디렉토리 경로
        """
        self.num_streams = num_streams
        self.scream_threshold = scream_threshold
        self.whisper_model_name = whisper_model
        
        # 디바이스 설정
        if gpu_enabled and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        logger.info(f"LoadTestSimulator 초기화: device={self.device}, streams={num_streams}")
        
        # 모델 경로 설정
        if model_path is None:
            # 기본 모델 경로
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            model_path = str(project_root / "models" / "audio" / "resnet18_scream_detector_v2.pth")
        
        self.model_path = model_path
        
        # 샘플 데이터 경로 설정
        if sample_data_path is None:
            self.sample_data_path = Path(__file__).resolve().parent / "sample_data"
        else:
            self.sample_data_path = Path(sample_data_path)
        
        # 모델 로드 (lazy loading)
        self._scream_model = None
        self._stt_model = None
        
        # VAD 필터 (Silero VAD 사용)
        self._vad_filter = None
        try:
            from sentinel_pipeline.infrastructure.audio.processors.vad_filter import create_vad_filter
            self._vad_filter = create_vad_filter(
                sample_rate=self.SAMPLE_RATE,
                threshold=0.5,  # 중간 임계값 (비명도 통과시키면서 잡음은 거름)
            )
            if self._vad_filter:
                logger.info("🛡️ Silero VAD Filter Initialized (Gatekeeper)")
        except ImportError:
            logger.warning("⚠️ silero-vad not installed. VAD filtering disabled. Install with: pip install silero-vad")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize VAD filter: {e}")
        
        # 실제 오디오 파일 캐시 - 카테고리별 분류
        # 비명: scream_*.mp3 (예: scream_1.mp3, scream_2.mp3, scream_3.mp3)
        # 긴급 키워드: 경찰.m4a, 긴급.m4a, 도와주세요.m4a, 사람살려.m4a, 살려주세요.m4a
        # 일반: non_scream_*.wav, 마이크.m4a, 음성파일.m4a, 처리용량.m4a, 테스트.m4a
        self._all_audio_files: list[tuple[str, np.ndarray, str]] = []  # [(filename, audio_data, category), ...]
        
        # 긴급 키워드 파일 목록 (하드코딩)
        self.EMERGENCY_KEYWORD_FILES = {"경찰.m4a", "긴급.m4a", "도와주세요.m4a", "사람살려.m4a", "살려주세요.m4a"}
        self.NORMAL_M4A_FILES = {"마이크.m4a", "음성파일.m4a", "처리용량.m4a", "테스트.m4a"}
        
        # 실제 오디오 파일 로드 (필수)
        self._load_sample_audio_files()
        
        if not self._all_audio_files:
            raise ValueError(f"샘플 오디오 파일이 없습니다: {self.sample_data_path}")
        
    def _load_models(self):
        """모델 로드 (최초 1회)"""
        if self._scream_model is None:
            logger.info("ScreamDetector 모델 로딩 중...")
            from sentinel_pipeline.infrastructure.audio.processors.scream_detector import ScreamDetector
            
            self._scream_model = ScreamDetector(
                model_path=self.model_path,
                threshold=self.scream_threshold,
                device=self.device,
                enable_filtering=True,  # ResNet-ScreamDetect와 동일하게 필터링 활성화
            )
            logger.info(f"ScreamDetector 로드 완료: {self.device}")
            
        if self._stt_model is None:
            # large 모델 선택 시 large-v3-turbo 사용 (ResNet 프로젝트와 동일)
            model_name = self.whisper_model_name
            if model_name == "large":
                model_name = "large-v3-turbo"
            
            logger.info(f"Whisper STT 모델 로딩 중 ({model_name})...")
            from faster_whisper import WhisperModel
            
            # GPU 사용 시 float16, CPU는 int8
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            self._stt_model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=compute_type,
            )
            
            # 모델이 실제로 GPU에 로드되었는지 확인
            if self.device == "cuda":
                import torch
                # 더 상세한 GPU 메모리 정보 로깅
                allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                logger.info(f"   GPU Memory Allocated: {allocated_mb:.2f} MB, Reserved: {reserved_mb:.2f} MB")
                if allocated_mb == 0:
                    logger.warning("   GPU에 메모리가 할당되지 않았습니다. CPU 모드로 실행 중일 수 있습니다.")
            
            logger.info(f"Whisper 로드 완료: {model_name} on {self.device}")
    
    def _load_sample_audio_files(self):
        """sample_data 폴더에서 실제 오디오 파일들을 로드하고 카테고리별 분류"""
        if not self.sample_data_path.exists():
            logger.warning(f"샘플 데이터 폴더가 없습니다: {self.sample_data_path}")
            return
        
        logger.info(f"샘플 오디오 파일 로딩 중: {self.sample_data_path}")
        
        # 카테고리 카운터
        category_counts = {"scream": 0, "emergency_keyword": 0, "normal": 0}
        
        # 모든 오디오 파일 로드 (wav, m4a, mp3)
        all_files = (
            list(self.sample_data_path.glob("*.wav")) + 
            list(self.sample_data_path.glob("*.m4a")) + 
            list(self.sample_data_path.glob("*.mp3"))
        )
        
        for file_path in all_files:
            try:
                audio, sr = librosa.load(str(file_path), sr=self.SAMPLE_RATE)
                
                # 파일명 기반 카테고리 분류
                filename = file_path.name
                category = self._classify_audio_file(filename)
                
                self._all_audio_files.append((filename, audio, category))
                category_counts[category] += 1
                logger.debug(f"  로드: {filename} ({len(audio)/self.SAMPLE_RATE:.1f}초) [{category}]")
            except Exception as e:
                logger.warning(f"  로드 실패: {file_path.name} - {e}")
        
        logger.info(f"샘플 오디오 로드 완료: 비명 {category_counts['scream']}개, "
                   f"긴급키워드 {category_counts['emergency_keyword']}개, "
                   f"일반 {category_counts['normal']}개")
    
    def _classify_audio_file(self, filename: str) -> str:
        """
        파일명 기반으로 오디오 카테고리 분류
        
        Returns:
            "scream" | "emergency_keyword" | "normal"
        """
        # 비명: scream_*.mp3 (예: scream_1.mp3, scream_2.mp3, scream_3.mp3)
        if filename.startswith("scream_"):
            return "scream"
        # 긴급 키워드: 특정 한국어 파일들
        elif filename in self.EMERGENCY_KEYWORD_FILES:
            return "emergency_keyword"
        # 나머지는 일반
        else:
            return "normal"
    
    def _get_random_audio(self) -> tuple[str, np.ndarray, str]:
        """
        실제 오디오 파일에서 완전 랜덤 선택
        
        Returns:
            (파일명, 오디오 데이터, 카테고리) 튜플
            카테고리: "scream" | "emergency_keyword" | "normal"
        """
        return random.choice(self._all_audio_files)
    
    def _prepare_audio_chunk(self, audio: np.ndarray, prefer_start: bool = False) -> np.ndarray:
        """
        오디오를 2초 윈도우로 자르거나 패딩
        
        Args:
            audio: 원본 오디오 데이터
            prefer_start: True면 처음부터 시작 (비명 파일의 경우 유용)
            
        Returns:
            2초 분량의 오디오 청크
        """
        target_len = int(self.SAMPLE_RATE * self.WINDOW_SEC)
        
        if len(audio) < target_len:
            # 짧으면 패딩
            return np.pad(audio, (0, target_len - len(audio)), mode='constant')
        elif len(audio) > target_len:
            if prefer_start:
                # 처음부터 시작 (비명 파일의 경우 처음에 비명이 있을 가능성 높음)
                return audio[:target_len]
            else:
                # 길면 랜덤 위치에서 자르기 (일반적인 경우)
                start = random.randint(0, len(audio) - target_len)
                return audio[start:start + target_len]
        else:
            return audio
    
    
    def warmup(self):
        """
        모델 워밍업 (첫 추론은 느리므로 측정 전 실행)
        
        멘토 조언: 첫 번째 실행(Inference)은 항상 느립니다.
        테스트 전에 한 번 데이터로 모델을 돌려주는 Warm-up 로직이
        코드에 포함되어야 정확한 측정이 됩니다.
        """
        logger.info("모델 워밍업 시작...")
        
        self._load_models()
        
        # 실제 오디오로 워밍업
        _, warmup_audio, _ = self._get_random_audio()
        warmup_chunk = self._prepare_audio_chunk(warmup_audio)
        
        # ScreamDetector 워밍업
        for _ in range(3):
            self._scream_model.predict(warmup_chunk)
        
        # Whisper STT 워밍업
        for _ in range(2):
            segments, _ = self._stt_model.transcribe(
                warmup_chunk,
                beam_size=1,
                language="ko",
            )
            # 제너레이터 소비
            list(segments)
        
        # GPU 메모리 정리
        if self.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
        
        logger.info("워밍업 완료!")
    
    def _measure_resources(self) -> tuple[float, float, float]:
        """현재 시점의 GPU 메모리, CPU 사용률, 시스템 메모리 측정"""
        gpu_mem = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        cpu_pct = psutil.cpu_percent(interval=None)
        
        # 시스템 메모리 사용량 (MB)
        mem_info = psutil.virtual_memory()
        system_mem_mb = mem_info.used / (1024 * 1024)  # MB
        
        return gpu_mem, cpu_pct, system_mem_mb
    
    def simulate_stream(
        self,
        stream_id: int,
    ) -> StreamMetrics:
        """
        단일 스트림의 처리 과정을 시뮬레이션하고 시간을 측정합니다.
        
        Args:
            stream_id: 스트림 식별자
            
        Returns:
            StreamMetrics: 처리 결과 메트릭 (리소스 사용량 포함)
        """
        start_time = time.perf_counter()
        
        # 오디오 선택 (실제 파일에서 랜덤)
        audio_filename, raw_audio, audio_category = self._get_random_audio()
        # 비명 파일의 경우 처음부터 시작 (비명이 처음에 있을 가능성 높음)
        prefer_start = (audio_category == "scream")
        audio = self._prepare_audio_chunk(raw_audio, prefer_start=prefer_start)
        
        # --- Step 0: VAD 필터 (CPU 처리 - 매우 빠름) ---
        # 목소리나 강한 소리가 없으면 아예 Drop (GPU 사용 0)
        if self._vad_filter and not self._vad_filter.is_speech(audio):
            # 조용한 구간이나 잡음만 있음 -> 처리 종료
            # 로그 레벨을 Debug로 낮춰서 도배 방지
            logger.debug(f"Stream {stream_id}: 🔇 Silence/Noise dropped by VAD")
            
            # VAD로 차단된 경우에도 메트릭 반환 (처리 시간은 매우 짧음)
            total_time = time.perf_counter() - start_time
            gpu_mem, cpu_pct, system_mem = self._measure_resources()
            
            return StreamMetrics(
                stream_id=stream_id,
                step1_latency=0.0,  # VAD에서 차단되어 처리 안 함
                step2_latency=0.0,
                total_latency=total_time,
                detected=False,
                scream_prob=0.0,
                transcript="",
                audio_file=audio_filename,
                audio_category=audio_category,
                gpu_memory_mb=gpu_mem,
                cpu_percent=cpu_pct,
                system_memory_mb=system_mem,
            )
        
        # --- Step 1: 비명 감지 (ResNet18) ---
        t1_start = time.perf_counter()
        result = self._scream_model.predict(audio)
        
        if self.device == "cuda":
            torch.cuda.synchronize()  # GPU 작업 완료 대기
            
        t1_end = time.perf_counter()
        
        scream_prob = result.get("prob", 0.0)
        status = result.get("status", "UNKNOWN")
        reason = result.get("reason", "")
        
        # 디버깅: 비명 파일인데 prob가 0이면 로그 출력
        if audio_category == "scream" and scream_prob < 0.01:
            logger.warning(
                f"Stream {stream_id}: 비명 파일({audio_filename})인데 prob={scream_prob:.4f}, "
                f"status={status}, reason={reason}"
            )
        
        is_scream = result.get("is_scream", False) or scream_prob > self.scream_threshold
        
        # --- Step 2: STT (Whisper) - 비명이 아닐 때만 실행 ---
        # 비명 감지 시: 즉각 알림 (별도 API) → STT 불필요
        # 비명 아닐 시: 음성 내용 분석 → STT로 키워드 검출
        t2_latency = 0.0
        transcript = ""
        
        if not is_scream:
            t2_start = time.perf_counter()
            
            segments, _ = self._stt_model.transcribe(
                audio,
                beam_size=5,
                language="ko",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=200, threshold=0.3),
            )
            
            # 제너레이터에서 텍스트 추출
            text_parts = [s.text for s in segments]
            transcript = " ".join(text_parts).strip()
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                
            t2_end = time.perf_counter()
            t2_latency = t2_end - t2_start
        
        total_time = time.perf_counter() - start_time
        
        # 청크 처리 완료 시점의 리소스 측정
        gpu_mem, cpu_pct, system_mem = self._measure_resources()
        
        return StreamMetrics(
            stream_id=stream_id,
            step1_latency=t1_end - t1_start,
            step2_latency=t2_latency,
            total_latency=total_time,
            detected=is_scream,
            scream_prob=scream_prob,
            transcript=transcript,
            audio_file=audio_filename,
            audio_category=audio_category,
            gpu_memory_mb=gpu_mem,
            cpu_percent=cpu_pct,
            system_memory_mb=system_mem,
        )
    
    def run_batch_test(
        self,
        warmup: bool = True,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> BenchmarkResult:
        """
        N개의 스트림을 동시에 처리하는 상황을 시뮬레이션 (Batch Processing)
        
        Args:
            warmup: 워밍업 실행 여부
            progress_callback: 각 스트림 처리 후 호출되는 콜백 (실시간 로그용)
            
        Returns:
            BenchmarkResult: 벤치마크 결과
        """
        # 모델 로드 단계 알림
        if progress_callback:
            progress_callback({
                "type": "status",
                "message": "모델 로딩 중...",
                "phase": "loading"
            })
        
        # 모델 로드 및 워밍업
        self._load_models()
        
        if warmup:
            if progress_callback:
                progress_callback({
                    "type": "status", 
                    "message": "워밍업 실행 중...",
                    "phase": "warmup"
                })
            self.warmup()
        
        if progress_callback:
            progress_callback({
                "type": "status",
                "message": f"테스트 시작: {self.num_streams} 스트림",
                "phase": "running"
            })
        
        # CPU 사용량 측정 시작
        psutil.cpu_percent(interval=None)  # 초기화
        
        # GPU 메모리 측정 시작
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
        
        results: list[StreamMetrics] = []
        scream_count = 0
        stt_count = 0
        
        loop_start = time.perf_counter()
        
        # 시뮬레이션: N개의 스트림을 순차 처리 (실제 오디오 랜덤 선택)
        for i in range(self.num_streams):
            metrics = self.simulate_stream(i)
            results.append(metrics)
            
            if metrics.detected:
                scream_count += 1
            else:
                # STT는 비명이 아닐 때 실행됨
                stt_count += 1
            
            # 실시간 진행 상황 콜백 (매 스트림마다)
            if progress_callback:
                progress_callback({
                    "type": "stream_result",
                    "stream_id": i,
                    "total_streams": self.num_streams,
                    "audio_file": metrics.audio_file,
                    "audio_category": metrics.audio_category,
                    "detected": metrics.detected,
                    "scream_prob": round(metrics.scream_prob, 3),
                    "step1_latency": round(metrics.step1_latency * 1000, 2),
                    "step2_latency": round(metrics.step2_latency * 1000, 2),
                    "total_latency": round(metrics.total_latency * 1000, 2),
                    "transcript": metrics.transcript,
                    "gpu_memory_mb": round(metrics.gpu_memory_mb, 2),
                    "cpu_percent": round(metrics.cpu_percent, 1),
                    "system_memory_mb": round(metrics.system_memory_mb, 2),
                })
            
            # 진행 상황 로깅 (10개마다)
            if (i + 1) % 10 == 0 or i == self.num_streams - 1:
                logger.debug(f"진행: {i + 1}/{self.num_streams} 스트림 처리 완료")
        
        loop_end = time.perf_counter()
        total_time = loop_end - loop_start
        
        # CPU 사용량 측정 종료
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # GPU 메모리 측정 종료
        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        else:
            gpu_mem = 0
            gpu_mem_peak = 0
        
        # 통계 집계
        latencies = [r.total_latency for r in results]
        avg_latency = float(np.mean(latencies))
        max_latency = float(np.max(latencies))
        min_latency = float(np.min(latencies))
        fps = self.num_streams / total_time if total_time > 0 else 0
        
        return BenchmarkResult(
            streams=self.num_streams,
            avg_latency=round(avg_latency, 4),
            max_latency=round(max_latency, 4),
            min_latency=round(min_latency, 4),
            fps=round(fps, 2),
            gpu_memory_mb=round(gpu_mem, 2),
            gpu_memory_peak_mb=round(gpu_mem_peak, 2),
            cpu_percent=round(cpu_percent, 1),
            device=self.device,
            scream_count=scream_count,
            stt_count=stt_count,
            total_time=round(total_time, 3),
            details=[
                {
                    "stream_id": r.stream_id,
                    "chunk_id": 0,  # Batch 모드에서는 각 스트림당 1개 청크
                    "timestamp": loop_start + r.total_latency,  # 처리 완료 시점
                    "step1_latency": round(r.step1_latency * 1000, 2),  # ms로 변환
                    "step2_latency": round(r.step2_latency * 1000, 2),
                    "total_latency": round(r.total_latency * 1000, 2),
                    "detected": r.detected,
                    "scream_prob": round(r.scream_prob, 3),
                    "audio_file": r.audio_file,
                    "audio_category": r.audio_category,
                    "gpu_memory_mb": round(r.gpu_memory_mb, 2),
                    "cpu_percent": round(r.cpu_percent, 1),
                    "system_memory_mb": round(r.system_memory_mb, 2),
                    "transcript": r.transcript,
                }
                for r in results
            ],
        )
    
    def run_continuous_test(
        self,
        duration: float = 30.0,
        interval: float = 1.0,
        warmup: bool = True,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> BenchmarkResult:
        """
        실제 스트림처럼 지속적인 부하 테스트
        
        각 스트림이 interval 간격으로 계속 새로운 오디오를 처리합니다.
        마치 N개의 마이크가 동시에 실시간 오디오를 보내는 것처럼 시뮬레이션합니다.
        
        Args:
            duration: 테스트 지속 시간 (초)
            interval: 각 스트림의 오디오 입력 간격 (초)
            warmup: 워밍업 실행 여부
            progress_callback: 각 청크 처리 후 호출되는 콜백 (실시간 로그용)
            
        Returns:
            BenchmarkResult: 벤치마크 결과
        """
        import threading
        import queue
        
        # 모델 로드 단계 알림
        if progress_callback:
            progress_callback({
                "type": "status",
                "message": "모델 로딩 중...",
                "phase": "loading"
            })
        
        # 모델 로드 및 워밍업
        self._load_models()
        
        if warmup:
            if progress_callback:
                progress_callback({
                    "type": "status",
                    "message": "워밍업 실행 중...",
                    "phase": "warmup"
                })
            self.warmup()
        
        if progress_callback:
            progress_callback({
                "type": "status",
                "message": f"연속 테스트 시작: {self.num_streams} 스트림 x {duration}초",
                "phase": "running"
            })
        
        logger.info(f"연속 부하 테스트 시작: {self.num_streams} streams x {duration}초, 간격={interval}초")
        
        # 결과 저장용
        results_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        
        # CPU/GPU 메모리 측정 시작
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
        
        cpu_samples = []
        
        # 콜백용 lock (thread-safe)
        callback_lock = threading.Lock()
        
        def stream_worker(stream_id: int):
            """개별 스트림 워커 - 지속적으로 오디오 처리 (실제 파일 랜덤 선택)"""
            chunk_count = 0
            first_chunk = True
            worker_start_time = time.perf_counter()
            
            if progress_callback:
                with callback_lock:
                    progress_callback({
                        "type": "status",
                        "message": f"Stream {stream_id} 워커 활성화됨. 첫 청크 처리 시작...",
                        "phase": "running"
                    })
            
            while not stop_event.is_set():
                # 오디오 처리
                try:
                    chunk_start = time.perf_counter()
                    
                    if first_chunk and progress_callback:
                        with callback_lock:
                            progress_callback({
                                "type": "status",
                                "message": f"Stream {stream_id} 첫 청크 처리 중... (예상 시간: CPU 모드에서 30-60초)",
                                "phase": "running"
                            })
                    
                    metrics = self.simulate_stream(stream_id)
                    chunk_end = time.perf_counter()
                    chunk_duration = chunk_end - chunk_start
                    
                    metrics_dict = {
                        "stream_id": stream_id,
                        "chunk_id": chunk_count,
                        "step1_latency": round(metrics.step1_latency * 1000, 2),
                        "step2_latency": round(metrics.step2_latency * 1000, 2),
                        "total_latency": round(metrics.total_latency * 1000, 2),
                        "detected": metrics.detected,
                        "scream_prob": round(metrics.scream_prob, 3),
                        "audio_file": metrics.audio_file,
                        "audio_category": metrics.audio_category,
                        "gpu_memory_mb": round(metrics.gpu_memory_mb, 2),
                        "cpu_percent": round(metrics.cpu_percent, 1),
                        "system_memory_mb": round(metrics.system_memory_mb, 2),
                        "transcript": metrics.transcript,
                        "timestamp": time.time(),
                    }
                    results_queue.put(metrics_dict)
                    
                    # 첫 청크 처리 완료 알림
                    if first_chunk and progress_callback:
                        with callback_lock:
                            progress_callback({
                                "type": "status",
                                "message": f"✅ Stream {stream_id} 첫 청크 처리 완료! (소요 시간: {chunk_duration:.1f}초)",
                                "phase": "running"
                            })
                        first_chunk = False
                    
                    # 실시간 콜백 (thread-safe)
                    if progress_callback:
                        with callback_lock:
                            progress_callback({
                                "type": "stream_result",
                                "stream_id": stream_id,
                                "chunk_id": chunk_count,
                                "total_streams": self.num_streams,
                                **metrics_dict
                            })
                    
                    chunk_count += 1
                except Exception as e:
                    logger.error(f"Stream {stream_id} 오류: {e}", exc_info=True)
                    if progress_callback:
                        with callback_lock:
                            progress_callback({
                                "type": "error",
                                "message": f"Stream {stream_id} 오류: {str(e)}"
                            })
                
                # 다음 오디오까지 대기
                time.sleep(interval)
        
        # 스트림 워커 스레드 시작
        threads = []
        test_start = time.perf_counter()
        workers_started = 0
        
        if progress_callback:
            progress_callback({
                "type": "status",
                "message": f"{self.num_streams}개 스트림 워커 시작 중...",
                "phase": "running"
            })
        
        for i in range(self.num_streams):
            t = threading.Thread(target=stream_worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)
            workers_started += 1
            
            if progress_callback:
                progress_callback({
                    "type": "status",
                    "message": f"Stream {i} 워커 시작됨 ({workers_started}/{self.num_streams})",
                    "phase": "running"
                })
            
            # 스트림 시작 시간 분산 (동시 시작 방지)
            time.sleep(interval / self.num_streams)
        
        logger.info(f"모든 스트림 시작됨. {duration}초 동안 실행...")
        
        if progress_callback:
            progress_callback({
                "type": "status",
                "message": f"✅ 모든 스트림 워커 시작 완료 ({self.num_streams}개). 첫 청크 처리 대기 중...",
                "phase": "running"
            })
        
        # duration 동안 대기하면서 CPU 샘플링
        elapsed = 0
        last_processed = 0
        last_log_time = test_start
        
        while elapsed < duration:
            time.sleep(1.0)
            elapsed = time.perf_counter() - test_start
            cpu_samples.append(psutil.cpu_percent(interval=None))
            
            # 진행 상황 로깅
            processed = results_queue.qsize()
            logger.debug(f"진행: {elapsed:.0f}/{duration:.0f}초, 처리된 청크: {processed}")
            
            # 처리 중인 스트림 수 추정 (처리 속도 기반)
            if processed > last_processed:
                # 처리 속도 계산
                time_diff = elapsed - (last_log_time - test_start) if last_log_time > test_start else 1.0
                rate = (processed - last_processed) / max(time_diff, 0.1)  # 초당 처리량
            else:
                rate = 0
            
            last_processed = processed
            last_log_time = time.perf_counter()
            
            # 진행률 콜백 (처리된 청크가 없어도 시간 경과는 표시)
            if progress_callback:
                # 처리 중인 워커 수 추정 (활성 스레드 수)
                active_threads = sum(1 for t in threads if t.is_alive())
                
                progress_callback({
                    "type": "progress",
                    "elapsed": round(elapsed, 1),
                    "duration": duration,
                    "processed": processed,
                    "processing_rate": round(rate, 1),
                    "active_workers": active_threads,
                    "total_workers": self.num_streams,
                    "percent": round(elapsed / duration * 100, 1),
                    "note": "처리 중..." if processed == 0 and elapsed < duration else None
                })
        
        # 테스트 종료
        stop_event.set()
        
        # 스레드 종료 대기
        for t in threads:
            t.join(timeout=2.0)
        
        test_end = time.perf_counter()
        total_time = test_end - test_start
        
        # 결과 수집
        all_results = []
        while not results_queue.empty():
            try:
                all_results.append(results_queue.get_nowait())
            except queue.Empty:
                break
        
        # GPU 메모리 측정
        if self.device == "cuda":
            torch.cuda.synchronize()
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2
        else:
            gpu_mem = 0
            gpu_mem_peak = 0
        
        # 통계 집계
        if all_results:
            latencies = [r["total_latency"] for r in all_results]
            avg_latency = float(np.mean(latencies)) / 1000  # 초 단위로 변환
            max_latency = float(np.max(latencies)) / 1000
            min_latency = float(np.min(latencies)) / 1000
            
            scream_count = sum(1 for r in all_results if r["detected"])
            stt_count = sum(1 for r in all_results if not r["detected"])
        else:
            avg_latency = max_latency = min_latency = 0
            scream_count = stt_count = 0
        
        fps = len(all_results) / total_time if total_time > 0 else 0
        avg_cpu = float(np.mean(cpu_samples)) if cpu_samples else 0
        
        logger.info(f"연속 테스트 완료: {len(all_results)} 청크 처리, {fps:.1f} chunks/sec")
        
        return BenchmarkResult(
            streams=self.num_streams,
            avg_latency=round(avg_latency, 4),
            max_latency=round(max_latency, 4),
            min_latency=round(min_latency, 4),
            fps=round(fps, 2),
            gpu_memory_mb=round(gpu_mem, 2),
            gpu_memory_peak_mb=round(gpu_mem_peak, 2),
            cpu_percent=round(avg_cpu, 1),
            device=self.device,
            scream_count=scream_count,
            stt_count=stt_count,
            total_time=round(total_time, 3),
            duration=duration,
            total_processed=len(all_results),
            details=all_results,
        )
    
    def get_system_status(self) -> dict[str, Any]:
        """현재 시스템 상태 조회"""
        status = {
            "device": self.device,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_available_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            status.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 0),
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
                "gpu_memory_cached_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
            })
        
        return status
    
    def cleanup(self):
        """리소스 정리 (메모리 누수 방지)"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        self._scream_model = None
        self._stt_model = None
        self._all_audio_files = []
        
        logger.info("리소스 정리 완료")


def main():
    """CLI 실행"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU 부하 테스트 시뮬레이터 (실제 오디오 파일 사용)")
    parser.add_argument("--streams", "-n", type=int, default=10, help="스트림 개수 (기본: 10)")
    parser.add_argument("--whisper-model", "-m", type=str, default="base", help="Whisper 모델 (기본: base)")
    parser.add_argument("--cpu-only", action="store_true", help="CPU만 사용")
    parser.add_argument("--no-warmup", action="store_true", help="워밍업 건너뛰기")
    parser.add_argument("--continuous", "-c", action="store_true", help="연속 부하 테스트 모드 (실제 스트림처럼)")
    parser.add_argument("--duration", "-t", type=float, default=30.0, help="연속 테스트 지속 시간 (초, 기본: 30)")
    parser.add_argument("--interval", "-i", type=float, default=1.0, help="오디오 입력 간격 (초, 기본: 1.0)")
    parser.add_argument("--output", "-o", type=str, default=None, help="결과 CSV 파일 경로 (기본: benchmark_result_YYYYMMDD_HHMMSS.csv)")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")
    
    args = parser.parse_args()
    
    # 로깅 설정
    from sentinel_pipeline.common.logging import configure_logging
    log_level = "DEBUG" if args.verbose else "INFO"
    configure_logging(level=log_level)
    
    print()
    print("=" * 60)
    print("  GPU Load Test Simulator")
    print("=" * 60)
    print(f"  Mode: {'Continuous (Real Stream Simulation)' if args.continuous else 'Batch (One-shot)'}")
    print(f"  Streams: {args.streams}")
    print(f"  Audio Source: Real Files (sample_data/)")
    if args.continuous:
        print(f"  Duration: {args.duration}초")
        print(f"  Interval: {args.interval}초 (각 스트림)")
    print(f"  Whisper Model: {args.whisper_model}")
    print(f"  Device: {'CPU' if args.cpu_only else 'GPU (if available)'}")
    print("=" * 60)
    print()
    
    # 시뮬레이터 생성 및 실행
    sim = LoadTestSimulator(
        num_streams=args.streams,
        gpu_enabled=not args.cpu_only,
        whisper_model=args.whisper_model,
    )
    
    # 시스템 상태 출력
    status = sim.get_system_status()
    print(f"[System] Device: {status['device']}")
    print(f"[System] CPU: {status['cpu_percent']}%")
    print(f"[System] Memory: {status['memory_percent']}% used")
    if "gpu_name" in status:
        print(f"[System] GPU: {status['gpu_name']}")
        print(f"[System] VRAM: {status['gpu_memory_allocated_mb']:.0f} / {status['gpu_memory_total_mb']:.0f} MB")
    print()
    
    # 벤치마크 실행
    if args.continuous:
        print(f"[Test] 연속 부하 테스트 시작 ({args.duration}초)...")
        result = sim.run_continuous_test(
            duration=args.duration,
            interval=args.interval,
            warmup=not args.no_warmup,
        )
    else:
        print("[Test] 배치 테스트 시작...")
        result = sim.run_batch_test(
            warmup=not args.no_warmup,
        )
    
    # 결과 출력
    print()
    print("=" * 60)
    print("  Benchmark Results")
    print("=" * 60)
    print(f"  Mode: {'Continuous' if args.continuous else 'Batch'}")
    print(f"  Total Streams: {result.streams}")
    if args.continuous:
        print(f"  Duration: {result.duration:.0f} sec")
        print(f"  Total Chunks Processed: {result.total_processed}")
    print(f"  Scream Detected: {result.scream_count} | STT Executed: {result.stt_count}")
    print("-" * 60)
    print(f"  Avg Latency: {result.avg_latency * 1000:.1f} ms")
    print(f"  Max Latency: {result.max_latency * 1000:.1f} ms")
    print(f"  Min Latency: {result.min_latency * 1000:.1f} ms")
    print(f"  Throughput: {result.fps:.1f} {'chunks' if args.continuous else 'streams'}/sec")
    print("-" * 60)
    print(f"  GPU Memory: {result.gpu_memory_mb:.0f} MB")
    print(f"  GPU Peak Memory: {result.gpu_memory_peak_mb:.0f} MB")
    print(f"  CPU Usage: {result.cpu_percent:.1f}%")
    print(f"  Total Time: {result.total_time:.2f} sec")
    print("=" * 60)
    
    # 실제 오디오 사용 시 상세 결과 출력
    if args.use_real_audio and args.verbose:
        print()
        print("  Stream Details:")
        print("-" * 60)
        for d in result.details:
            status = "🚨 SCREAM" if d["detected"] else "✅ SAFE"
            category = d.get("audio_category", "normal")
            category_emoji = {"scream": "🔴비명", "emergency_keyword": "🟠긴급", "normal": "🟢일반"}.get(category, "⚪")
            # 비명 카테고리인 경우에만 정답 여부 체크
            is_scream_gt = category == "scream"
            correct = "✓" if d["detected"] == is_scream_gt else "✗"
            print(f"  [{d['stream_id']:2d}] {d['audio_file']:25s} | GT:{category_emoji} | {status} (prob:{d['scream_prob']:.2f}) {correct}")
            if d["transcript"]:
                print(f"       └─ STT: \"{d['transcript']}\"")
        print("=" * 60)
        
        # 정확도 계산 (비명 감지 정확도: scream 카테고리만 detected=True여야 함)
        correct_count = sum(1 for d in result.details if d["detected"] == (d.get("audio_category") == "scream"))
        accuracy = correct_count / len(result.details) * 100 if result.details else 0
        print(f"  Detection Accuracy: {correct_count}/{len(result.details)} ({accuracy:.1f}%)")
        print("=" * 60)
    
    # CSV 저장
    csv_path = result.save_to_csv(args.output)
    print()
    print(f"[Save] 결과가 CSV 파일로 저장되었습니다: {csv_path}")
    
    # 정리
    sim.cleanup()


if __name__ == "__main__":
    main()
