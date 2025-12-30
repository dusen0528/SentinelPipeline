"""
비명 감지 프로세서

ResNet 기반 모델을 사용하여 오디오에서 비명을 감지합니다.
"""

import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Any, Optional
import logging

from sentinel_pipeline.domain.interfaces.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class ScreamDetector(AudioProcessor):
    """
    ResNet 기반 비명 감지기
    """
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.8,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.threshold = threshold
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 전처리 파라미터
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 64
        
        # 멜스펙트로그램 변환기
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        ).to(self.device)
        
        # 이미지 변환
        self.image_transform = transforms.Compose([
            transforms.Resize((64, 862)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"ScreamDetector initialized on {self.device} (threshold={threshold})")

    def _load_model(self) -> nn.Module:
        try:
            from torchvision.models import resnet34
            model = resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            
            if os.path.exists(self.model_path):
                # weights_only=False for older PyTorch compatibility/safety
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        try:
                            model.load_state_dict(checkpoint)
                        except:
                            if hasattr(checkpoint, 'state_dict'):
                                model.load_state_dict(checkpoint.state_dict())
                else:
                    if hasattr(checkpoint, 'state_dict'):
                        model.load_state_dict(checkpoint.state_dict())
                    elif hasattr(checkpoint, 'fc'):
                        model = checkpoint.to(self.device)
                        return model
                    else:
                        model.load_state_dict(checkpoint)
            else:
                logger.warning(f"Model file not found at {self.model_path}, using random weights")
                
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback
            from torchvision.models import resnet34
            model = resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            return model.to(self.device)

    def _audio_to_melspectrogram(self, audio: np.ndarray) -> torch.Tensor:
        audio_tensor = torch.from_numpy(audio).float()
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        audio_tensor = audio_tensor.to(self.device)
        mel_spec = self.mel_transform(audio_tensor)
        mel_spec = mel_spec.squeeze(0)
        mel_spec = torch.log2(mel_spec + 1e-10)
        return mel_spec

    def _melspectrogram_to_image(self, mel_spec: torch.Tensor) -> Image.Image:
        mel_spec_np = mel_spec.cpu().numpy()
        mel_spec_np = (mel_spec_np - mel_spec_np.min()) / (mel_spec_np.max() - mel_spec_np.min() + 1e-10)
        mel_spec_np = (mel_spec_np * 255).astype(np.uint8)
        
        img = Image.fromarray(mel_spec_np, mode='L')
        img = img.convert('RGB')
        return img

    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        try:
            mel_spec = self._audio_to_melspectrogram(audio)
            img = self._melspectrogram_to_image(mel_spec)
            input_tensor = self.image_transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                scream_prob = probabilities[0][1].item()
                is_scream = scream_prob >= self.threshold
                
            return is_scream, scream_prob
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return False, 0.0

    def process(self, audio_data: np.ndarray) -> dict[str, Any]:
        is_scream, confidence = self.detect(audio_data)
        return {
            "is_scream": is_scream,
            "confidence": confidence,
            "threshold": self.threshold
        }
