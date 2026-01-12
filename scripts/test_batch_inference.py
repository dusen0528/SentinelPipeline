import asyncio
import numpy as np
import torch
import logging
import sys
import time
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from sentinel_pipeline.infrastructure.audio.processors.batch_scream_detector import BatchScreamDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def test_batch_scream_detector():
    logger.info("=== BatchScreamDetector Integration Test ===")
    
    # 1. 초기화 (GPU 확인)
    model_path = "models/audio/resnet18_scream_detector_v2.pth"
    detector = BatchScreamDetector(
        model_path=model_path,
        batch_size=4,
        latency_limit=0.1
    )
    
    # 2. 루프 시작 (async start)
    await detector.start()
    
    logger.info(f"Using device: {detector.device}")
    if "cuda" in detector.device:
        logger.info(f"GPU Model: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")

    # 3. 더미 오디오 데이터 생성 (4개의 스트림이 동시에 요청하는 시나리오)
    async def mock_stream_request(stream_id, delay):
        await asyncio.sleep(delay)
        dummy_audio = np.random.uniform(-1, 1, 32000).astype(np.float32)
        
        start_t = time.monotonic()
        result = await detector.predict(dummy_audio)
        latency = (time.monotonic() - start_t) * 1000
        
        logger.info(f"[Stream {stream_id}] Result: {result['is_scream']} (prob: {result['prob']:.4f}) latency: {latency:.2f}ms")

    # 4. 동시 요청 실행
    logger.info("Sending 8 concurrent requests...")
    tasks = []
    for i in range(8):
        # 0.01초 간격으로 요청 (배치 수집 테스트)
        tasks.append(mock_stream_request(i, i * 0.01))
    
    await asyncio.gather(*tasks)
    
    # 5. 종료
    await detector.stop()
    logger.info("=== Test Completed ===")

if __name__ == "__main__":
    try:
        asyncio.run(test_batch_scream_detector())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
