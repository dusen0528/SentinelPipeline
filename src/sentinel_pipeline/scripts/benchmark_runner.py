"""
GPU 부하 테스트 시뮬레이터 (Batch & Async 지원 버전)

- BatchScreamDetector (GPU) 기반 비동기 처리
- GlobalInferenceEngine (GPU) 기반 Whisper STT 연동
- SSE 연동을 위한 Progress Callback 지원
"""

import asyncio
import time
import psutil
import torch
import numpy as np
import logging
import threading
from typing import List, Dict, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field

from sentinel_pipeline.infrastructure.audio.processors.batch_scream_detector import BatchScreamDetector
from sentinel_pipeline.infrastructure.audio.processors.risk_analyzer import GlobalInferenceEngine
from sentinel_pipeline.scripts.data_loader import AudioDataLoader

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """API 호환 결과를 위한 데이터 클래스"""
    streams: int
    avg_latency: float  # 초 단위 (API에서 ms로 변환)
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
    duration: float = 0.0
    total_processed: int = 0
    details: List[Dict] = field(default_factory=list)

class LoadTestSimulator:
    """
    [GPU-Optimized Load Test Simulator]
    API(benchmark.py)와 scripts/benchmark/runner.py를 연결하는 가교 역할을 수행합니다.
    """

    def __init__(
        self, 
        num_streams: int = 10, 
        gpu_enabled: bool = True, 
        whisper_model: str = "base",
        batch_size: int = 16
    ):
        self.num_streams = num_streams
        self.device = "cuda" if gpu_enabled and torch.cuda.is_available() else "cpu"
        self.whisper_model_name = whisper_model
        self.batch_size = batch_size
        
        # 모델 경로
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.scream_model_path = str(project_root / "models" / "audio" / "resnet18_scream_detector_v2.pth")
        
        # 엔진 (Lazy Loading)
        self.detector: Optional[BatchScreamDetector] = None
        self.stt_engine: Optional[GlobalInferenceEngine] = None
        self.data_loader = AudioDataLoader()
        
        # 결과 저장용
        self.results = []
        self._lock = threading.Lock()
        
        # 비동기 루프 (스레드에서 실행될 때 필요)
        self._loop = None

    def _ensure_engines(self):
        """엔진 초기화 (동기 호출용)"""
        if self.detector is None:
            # BatchScreamDetector는 내부적으로 loop가 필요하므로 
            # 현재 스레드에 루프가 없으면 생성하거나 get_event_loop 사용
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            
            self.detector = BatchScreamDetector(
                model_path=self.scream_model_path,
                device=self.device,
                batch_size=self.batch_size
            )
            
            # start()가 async이므로 run_until_complete로 실행
            if self._loop.is_running():
                # 이미 루프가 실행 중이면 create_task 사용
                asyncio.run_coroutine_threadsafe(self.detector.start(), self._loop)
            else:
                # 루프가 없으면 run_until_complete 사용
                self._loop.run_until_complete(self.detector.start())
            
            # Whisper 엔진은 Singleton
            self.stt_engine = GlobalInferenceEngine()
            # 벤치마크 설정에 맞춰 모델 크기 조정 가능하지만, 
            # 여기서는 이미 초기화된 엔진을 사용함

    async def _process_chunk_async(self, stream_id: int, chunk: np.ndarray, file_info: dict) -> dict:
        """한 개의 청크를 처리하는 비동기 로직"""
        start_t = time.perf_counter()
        
        # 1. Scream Detection (Batch)
        s1_start = time.perf_counter()
        scream_res = await self.detector.predict(chunk)
        s1_latency = (time.perf_counter() - s1_start) * 1000.0
        
        # 2. STT (비명이 아닐 때만 수행하는 시나리오)
        s2_latency = 0.0
        transcript = ""
        
        if not scream_res['is_scream']:
            s2_start = time.perf_counter()
            # STT는 동기 큐 방식이므로 래핑
            # 벤치마크이므로 결과를 기다려야 함 (Callback 사용)
            stt_future = self._loop.create_future()
            
            def stt_callback(res):
                if not stt_future.done():
                    self._loop.call_soon_threadsafe(stt_future.set_result, res)
            
            # RiskAnalyzer 구조와 유사하게 요청
            from sentinel_pipeline.infrastructure.audio.processors.risk_analyzer import InferenceRequest
            req = InferenceRequest(
                stream_id=f"sim_{stream_id}",
                audio_data=chunk,
                callback=stt_callback,
                timestamp=time.time()
            )
            self.stt_engine.submit(req)
            
            # 결과 대기
            stt_res = await stt_future
            s2_latency = (time.perf_counter() - s2_start) * 1000.0
            transcript = stt_res.get('text', '')

        total_latency = (time.perf_counter() - start_t) * 1000.0
        
        # 시스템 리소스 (청크 처리 시점)
        gpu_mem = 0
        if "cuda" in self.device:
            gpu_mem = torch.cuda.memory_allocated() / 1024**2
            
        return {
            "stream_id": stream_id,
            "step1_latency": s1_latency,
            "step2_latency": s2_latency,
            "total_latency": total_latency,
            "detected": scream_res['is_scream'],
            "scream_prob": scream_res['prob'],
            "audio_file": file_info.get("filename", "unknown"),
            "audio_category": file_info.get("category", "normal"),
            "gpu_memory_mb": gpu_mem,
            "cpu_percent": psutil.cpu_percent(),
            "transcript": transcript
        }

    def run_batch_test(self, warmup: bool = True, progress_callback: Callable = None) -> BenchmarkResult:
        """[API 동기 인터페이스] 배치 테스트 실행"""
        self._ensure_engines()
        return self._loop.run_until_complete(self._run_batch_async(warmup, progress_callback))

    async def _run_batch_async(self, warmup: bool, progress_callback: Callable) -> BenchmarkResult:
        if warmup:
            if progress_callback: progress_callback({"type": "status", "message": "Warming up GPU...", "phase": "warmup"})
            dummy = np.zeros(32000, dtype=np.float32)
            await asyncio.gather(*[self.detector.predict(dummy) for _ in range(self.batch_size)])

        if progress_callback: progress_callback({"type": "status", "message": "Starting batch test...", "phase": "running"})
        
        start_t = time.time()
        tasks = []
        for i in range(self.num_streams):
            file_path, chunk, info = self.data_loader.get_prepared_chunk()
            tasks.append(self._process_chunk_async(i, chunk, info))
        
        # 실행 및 결과 수집 (Progress는 개별적으로 쏴야 함)
        details = []
        for coro in asyncio.as_completed(tasks):
            res = await coro
            details.append(res)
            if progress_callback:
                progress_callback({
                    "type": "stream_result",
                    **res
                })
        
        total_time = time.time() - start_t
        return self._build_result(details, total_time)

    def run_continuous_test(self, duration: float, interval: float, warmup: bool = True, progress_callback: Callable = None) -> BenchmarkResult:
        """[API 동기 인터페이스] 연속 부하 테스트 실행"""
        self._ensure_engines()
        return self._loop.run_until_complete(self._run_continuous_async(duration, interval, warmup, progress_callback))

    async def _run_continuous_async(self, duration: float, interval: float, warmup: bool, progress_callback: Callable) -> BenchmarkResult:
        if warmup:
            dummy = np.zeros(32000, dtype=np.float32)
            await self.detector.predict(dummy)

        if progress_callback: progress_callback({"type": "status", "message": f"Starting continuous test ({duration}s)...", "phase": "running"})
        
        start_time = time.time()
        end_time = start_time + duration
        details = []
        processed_count = 0
        
        async def stream_loop(sid):
            nonlocal processed_count
            while time.time() < end_time:
                filename, chunk, info = self.data_loader.get_prepared_chunk()
                res = await self._process_chunk_async(sid, chunk, info)
                
                with self._lock:
                    details.append(res)
                    processed_count += 1
                
                if progress_callback:
                    progress_callback({"type": "stream_result", **res})
                
                # 다음 청크까지 대기
                await asyncio.sleep(interval)

        # 모든 스트림 동시 실행
        tasks = [asyncio.create_task(stream_loop(i)) for i in range(self.num_streams)]
        
        # 진행률 주기적 보고
        while time.time() < end_time:
            await asyncio.sleep(1.0)
            if progress_callback:
                elapsed = time.time() - start_time
                progress_callback({
                    "type": "progress",
                    "percent": (elapsed / duration) * 100,
                    "processed": processed_count,
                    "processing_rate": processed_count / elapsed if elapsed > 0 else 0
                })
        
        await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        return self._build_result(details, total_time, continuous=True, duration=duration)

    def _build_result(self, details, total_time, continuous=False, duration=0.0) -> BenchmarkResult:
        if not details:
            return BenchmarkResult(self.num_streams, 0, 0, 0, 0, 0, 0, 0, self.device, 0, 0, total_time)
            
        latencies = [d['total_latency'] for d in details]
        gpu_mems = [d['gpu_memory_mb'] for d in details]
        scream_count = sum(1 for d in details if d['detected'])
        
        return BenchmarkResult(
            streams=self.num_streams,
            avg_latency=np.mean(latencies) / 1000.0,
            max_latency=max(latencies) / 1000.0,
            min_latency=min(latencies) / 1000.0,
            fps=len(details) / total_time,
            gpu_memory_mb=np.mean(gpu_mems),
            gpu_memory_peak_mb=max(gpu_mems),
            cpu_percent=psutil.cpu_percent(),
            device=self.device,
            scream_count=scream_count,
            stt_count=len(details) - scream_count,
            total_time=total_time,
            duration=duration,
            total_processed=len(details),
            details=details
        )

    def get_system_status(self) -> dict:
        """시스템 리소스 상태 정보 반환"""
        mem = psutil.virtual_memory()
        status = {
            "device": self.device,
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": mem.percent,
            "memory_available_gb": mem.available / (1024**3),
            "gpu_name": None,
            "gpu_memory_total_mb": None,
            "gpu_memory_allocated_mb": None,
            "gpu_memory_cached_mb": None
        }
        
        if "cuda" in self.device:
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
            status["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024**2
            status["gpu_memory_cached_mb"] = torch.cuda.memory_reserved(0) / 1024**2
            
        return status

    def cleanup(self):
        """리소스 정리"""
        if self.detector:
            asyncio.run_coroutine_threadsafe(self.detector.stop(), self._loop)
            self.detector = None
