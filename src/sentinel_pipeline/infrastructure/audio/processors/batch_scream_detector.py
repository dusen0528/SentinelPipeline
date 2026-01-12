import asyncio
import numpy as np
import torch
import torchaudio
import logging
import time
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class BatchScreamDetector:
    """
    [GPU-Optimized Async Scream Detector]
    - Async IO: Non-blocking predict interface
    - Dynamic Batching: Collects requests into batches for GPU processing
    - GPU Preprocessing: torchaudio based MelSpectrogram & Normalization
    """

    def __init__(
        self, 
        model_path: str,
        threshold: float = 0.7,
        device: str = None, 
        batch_size: int = 16, 
        latency_limit: float = 0.05
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.latency_limit = latency_limit 
        self.model_path = model_path
        self.threshold = threshold
        
        # 상태 관리
        self.queue: Optional[asyncio.Queue] = None
        self.worker_task: Optional[asyncio.Task] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 모델 및 전처리 (Lazy Loading)
        self.mel_transform = None
        self.model = None
        
        # 오디오 설정
        self.sample_rate = 16000
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        self.target_length = 32000 # 2초

        # 리소스 로드
        self._load_resources()

    def _load_resources(self):
        """모델 및 전처리 리소스 로드 (동기 실행)"""
        gpu_info = f" (Device: {torch.cuda.get_device_name(0)})" if "cuda" in self.device else ""
        logger.info(f"🔥 [BatchScreamDetector] Loading resources on {self.device}{gpu_info}...")
        
        try:
            # 1. 전처리 모듈 (GPU)
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0,
                norm='slaney',     # librosa 호환
                mel_scale='slaney' # librosa 호환
            ).to(self.device)
            
            # 2. 모델 로드 (resnet18 기반)
            from torchvision.models import resnet18
            import torch.nn as nn
            
            model = resnet18(weights=None) # Weights=None for clean load
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 2)
            )
            
            if Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
                logger.info(f"✅ Model weights loaded from {self.model_path}")
            else:
                logger.warning(f"⚠️ Model path not found: {self.model_path}. Using random weights.")
            
            model.to(self.device)
            model.eval()
            self.model = model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise e

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """워커 시작 (명시적 루프 지정 가능)"""
        if not self.running:
            self.running = True
            self.loop = loop or asyncio.get_running_loop()
            
            # 큐 생성은 반드시 루프 내에서 (또는 루프 바인딩)
            async def _init_queue():
                self.queue = asyncio.Queue()
                self.worker_task = asyncio.create_task(self._worker_loop())
            
            if self.loop.is_running():
                asyncio.run_coroutine_threadsafe(_init_queue(), self.loop)
            else:
                # 아직 루프가 돌지 않는다면 (예: 테스트 코드)
                # 이 방식은 uvicorn 환경에선 거의 안 쓰임
                pass
                
            logger.info(f"🚀 Batch worker started on loop {id(self.loop)}")

    async def predict(self, audio: np.ndarray) -> dict:
        """[Public API]"""
        if not self.running or self.queue is None:
            # 방어 코드: 시작되지 않았으면 결과 즉시 반환 (또는 에러)
            return {"is_scream": False, "prob": 0.0, "status": "not_ready"}

        future = self.loop.create_future()
        await self.queue.put((audio, future))
        return await future

    async def _worker_loop(self):
        """배치 처리 루프"""
        while self.running:
            batch_items = []
            try:
                # 1. 첫 아이템 대기 (Timeout 없음 = CPU Idle)
                item = await self.queue.get()
                
                # 종료 신호 확인
                if item is None:
                    break
                    
                batch_items.append(item)
                
                # 2. Latency Limit 동안 추가 수집
                start_t = time.monotonic()
                while len(batch_items) < self.batch_size:
                    remaining = self.latency_limit - (time.monotonic() - start_t)
                    if remaining <= 0: break
                    
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                        # 종료 신호 확인
                        if item is None:
                            break
                        batch_items.append(item)
                    except asyncio.TimeoutError:
                        break
                
                # 3. 처리
                await self._process_batch(batch_items)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker Loop Error: {e}", exc_info=True)

    async def _process_batch(self, items: List[Tuple[np.ndarray, asyncio.Future]]):
        if not items: return
        
        futures = [x[1] for x in items]
        audios = [x[0] for x in items]
        
        try:
            # A. Numpy -> Tensor (CPU) -> GPU Stack
            tensors = []
            for a in audios:
                if len(a) < self.target_length:
                    a = np.pad(a, (0, self.target_length - len(a)), 'constant')
                else:
                    a = a[:self.target_length]
                tensors.append(torch.from_numpy(a))
            
            batch_tensor = torch.stack(tensors).float().to(self.device)
            
            # B. Preprocessing (GPU)
            with torch.no_grad():
                melspec = self.mel_transform(batch_tensor)
                
                # PowerToDB & Norm
                melspec_db = 10.0 * torch.log10(melspec + 1e-6)
                
                # Per-sample Max/Min
                flat = melspec_db.view(melspec_db.size(0), -1)
                max_val = flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
                min_val = flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
                
                denom = max_val - min_val
                denom[denom == 0] = 1.0
                
                spec_norm = (melspec_db - min_val) / denom
                input_tensor = spec_norm.unsqueeze(1) # [B, 1, H, W]
                
                # C. Inference
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            # D. Result
            for i, prob in enumerate(probs):
                if not futures[i].done():
                    futures[i].set_result({
                        "is_scream": float(prob) > self.threshold,
                        "prob": float(prob)
                    })
                    
        except Exception as e:
            logger.error(f"Batch Processing Error: {e}", exc_info=True)
            for f in futures:
                if not f.done(): f.set_exception(e)
    
    async def stop(self):
        """워커 종료"""
        if not self.running:
            return
        
        self.running = False
        
        # 큐에 None을 넣어서 워커 루프 종료 신호
        if self.queue:
            await self.queue.put(None)
        
        # 워커 태스크 종료 대기
        if self.worker_task:
            try:
                await asyncio.wait_for(self.worker_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("Worker task did not stop in time, cancelling...")
                self.worker_task.cancel()
                try:
                    await self.worker_task
                except asyncio.CancelledError:
                    pass
        
        logger.info("BatchScreamDetector stopped")