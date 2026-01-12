import random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import librosa
from sentinel_pipeline.common.logging import get_logger

logger = get_logger(__name__)

class AudioDataLoader:
    """
    벤치마크용 오디오 데이터 로더
    
    기능:
    1. sample_data 폴더의 모든 WAV 파일 로드
    2. 메모리에 PCM 데이터 캐싱 (Disk I/O 병목 제거)
    3. 카테고리별(Scream, Normal, Emergency) 분류
    """
    
    EMERGENCY_KEYWORDS = {"경찰.wav", "긴급.wav", "도와주세요.wav", "사람살려.wav", "살려주세요.wav"}
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.audio_cache: List[Tuple[str, np.ndarray, str]] = [] # (filename, audio, category)
        self.categories: Dict[str, List[int]] = {
            "scream": [],
            "emergency_keyword": [],
            "normal": []
        }
        
        # 기본 경로 설정: 스크립트와 같은 위치의 sample_data 폴더
        self.base_path = Path(__file__).resolve().parent / "sample_data"
        self._load_all()

    def _classify(self, filename: str) -> str:
        if filename.startswith("scream_"):
            return "scream"
        elif filename in self.EMERGENCY_KEYWORDS:
            return "emergency_keyword"
        return "normal"

    def _load_all(self):
        if not self.base_path.exists():
            logger.warning(f"⚠️ Sample data not found at {self.base_path}")
            return

        wav_files = list(self.base_path.glob("*.wav"))
        logger.info(f"📂 Loading {len(wav_files)} audio files from {self.base_path}...")

        for f in wav_files:
            try:
                # librosa로 로드 (여기서는 전처리가 아니라 단순 로딩이므로 OK)
                audio, _ = librosa.load(str(f), sr=self.sample_rate)
                
                # float32 확인
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                
                category = self._classify(f.name)
                
                # 인덱스 저장
                idx = len(self.audio_cache)
                self.audio_cache.append((f.name, audio, category))
                self.categories[category].append(idx)
                
            except Exception as e:
                logger.warning(f"❌ Failed to load {f.name}: {e}")

        logger.info(f"✅ Data loaded: Scream={len(self.categories['scream'])}, "
                    f"Emergency={len(self.categories['emergency_keyword'])}, "
                    f"Normal={len(self.categories['normal'])}")

    def get_random_sample(self) -> Tuple[str, np.ndarray, str]:
        """무작위 샘플 반환 (가중치 없이 완전 랜덤)"""
        if not self.audio_cache:
            # 데이터 없으면 0으로 채운 더미 반환
            return "dummy.wav", np.zeros(self.sample_rate*2, dtype=np.float32), "normal"
        return random.choice(self.audio_cache)
    
    def get_prepared_chunk(self, window_sec: float = 2.0) -> Tuple[str, np.ndarray, Dict]:
        """모델 입력 길이에 맞게 자르거나 패딩된 청크 반환
        
        Returns:
            (filename, chunk, info_dict) 튜플
            info_dict: {"filename": str, "category": str}
        """
        filename, raw_audio, category = self.get_random_sample()
        
        target_len = int(self.sample_rate * window_sec)
        curr_len = len(raw_audio)
        
        if curr_len < target_len:
            # Padding
            chunk = np.pad(raw_audio, (0, target_len - curr_len), mode='constant')
        elif curr_len > target_len:
            # Cutting (비명은 앞부분 선호, 나머지는 랜덤)
            if category == "scream":
                chunk = raw_audio[:target_len]
            else:
                start = random.randint(0, curr_len - target_len)
                chunk = raw_audio[start:start+target_len]
        else:
            chunk = raw_audio
        
        info = {
            "filename": filename,
            "category": category
        }
            
        return filename, chunk, info
