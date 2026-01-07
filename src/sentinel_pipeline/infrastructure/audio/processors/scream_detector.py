"""
비명 감지 프로세서

ResNet 기반 모델을 사용하여 오디오에서 비명을 감지합니다.
ResNet18 및 ResNet34를 지원하며, 추가 필터링 로직을 포함합니다.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Tuple, Any, Optional, Dict
import logging

from sentinel_pipeline.domain.interfaces.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class ScreamDetector(AudioProcessor):
    """
    ResNet 기반 비명 감지기
    
    ResNet18 또는 ResNet34 모델을 사용하며, 기계음 필터링 및 비명 강도 분석 기능을 포함합니다.
    """
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.6,
        device: str = "auto",
        model_arch: str = "auto",  # "auto", "resnet18", "resnet34"
        enable_filtering: bool = True,  # 필터링 로직 활성화 여부
    ):
        """
        Args:
            model_path: 모델 가중치 파일 경로 (.pth)
            threshold: 비명 판정 임계값 (0.0 ~ 1.0)
            device: 디바이스 ('auto', 'cuda', 'cpu')
            model_arch: 모델 아키텍처 ('auto', 'resnet18', 'resnet34')
            enable_filtering: 기계음 필터링 및 비명 강도 분석 활성화 여부
        """
        self.model_path = model_path
        self.threshold = threshold
        self.enable_filtering = enable_filtering
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 전처리 파라미터
        self.sample_rate = 16000
        self.window_sec = 2.0
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 64
        
        # 모델 아키텍처 자동 감지 또는 명시적 설정
        self.model_arch = model_arch
        if model_arch == "auto":
            # 파일명이나 모델 구조를 보고 자동 감지
            if "resnet18" in model_path.lower() or "18" in model_path.lower():
                self.model_arch = "resnet18"
            elif "resnet34" in model_path.lower() or "34" in model_path.lower():
                self.model_arch = "resnet34"
            else:
                # 기본값: ResNet34 (기존 호환성)
                self.model_arch = "resnet34"
        
        # 멜스펙트로그램 변환기 (librosa 사용 방식과 호환)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        ).to(self.device)
        
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(
            f"ScreamDetector initialized on {self.device} "
            f"(arch={self.model_arch}, threshold={threshold}, filtering={enable_filtering})"
        )

    def _load_model(self) -> nn.Module:
        """모델 아키텍처 로드 및 가중치 로드"""
        try:
            from torchvision import models
            
            # 모델 아키텍처 선택
            if self.model_arch == "resnet18":
                model = models.resnet18(weights=None)
                # 1채널(Grayscale) 입력 수용
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            else:  # resnet34
                model = models.resnet34(weights=None)
                # ResNet34는 기본적으로 3채널 입력이지만, 1채널로 변경 가능
                # 기존 호환성을 위해 3채널 유지 (이미지 변환 방식 사용)
            
            # 출력 레이어 설정 (2개 클래스: Non-Scream, Scream)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}, using random weights")
                return model.to(self.device)
            
            # 가중치 로드
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # 직접 state_dict인 경우
                        model.load_state_dict(checkpoint)
                else:
                    # 모델 객체 자체인 경우
                    if hasattr(checkpoint, 'state_dict'):
                        model.load_state_dict(checkpoint.state_dict())
                    elif hasattr(checkpoint, 'fc'):
                        # 이미 완전한 모델인 경우
                        model = checkpoint.to(self.device)
                        return model
                    else:
                        model.load_state_dict(checkpoint)
                        
                logger.info(f"Model weights loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                logger.warning("Using random weights instead")
                
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model architecture: {e}")
            # Fallback to ResNet34
            from torchvision.models import resnet34
            model = resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
            return model.to(self.device)

    def _preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        오디오 전처리: 길이 맞춤 -> Mel-Spectrogram -> 정규화 -> 텐서
        
        Args:
            audio: 오디오 데이터 (numpy array)
            
        Returns:
            전처리된 텐서 [1, 1, Height, Width]
        """
        target_len = int(self.sample_rate * self.window_sec)
        
        # 길이 맞추기 (패딩 또는 자르기)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant', constant_values=0.0)
        else:
            audio = audio[:target_len]
        
        # Mel-Spectrogram 변환 (librosa 방식 - ResNet18 모델과 호환)
        if self.model_arch == "resnet18":
            melspec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate,
                n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
            )
            melspec_db = librosa.power_to_db(melspec, ref=np.max)
            
            # 정규화 (Min-Max Scaling)
            min_val, max_val = melspec_db.min(), melspec_db.max()
            if max_val - min_val > 0:
                melspec_norm = (melspec_db - min_val) / (max_val - min_val)
            else:
                melspec_norm = melspec_db
            
            # [Batch, Channel, Height, Width] 형태로 변환
            tensor = torch.tensor(melspec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            # ResNet34 방식 (기존 호환성 유지)
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            audio_tensor = audio_tensor.to(self.device)
            mel_spec = self.mel_transform(audio_tensor)
            mel_spec = mel_spec.squeeze(0)
            mel_spec = torch.log2(mel_spec + 1e-10)
            
            # 이미지 변환을 위한 정규화
            mel_spec_np = mel_spec.cpu().numpy()
            mel_spec_np = (mel_spec_np - mel_spec_np.min()) / (mel_spec_np.max() - mel_spec_np.min() + 1e-10)
            mel_spec_np = (mel_spec_np * 255).astype(np.uint8)
            
            img = Image.fromarray(mel_spec_np, mode='L')
            img = img.convert('RGB')
            
            # 이미지 변환
            image_transform = transforms.Compose([
                transforms.Resize((64, 862)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = image_transform(img).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _is_human_voice(self, segment: np.ndarray) -> Tuple[bool, str]:
        """
        기계음/비프음 필터링 (True면 사람 목소리/환경음 가능성 높음)
        
        Args:
            segment: 오디오 세그먼트
            
        Returns:
            (is_human, reason) 튜플
        """
        if len(segment) < self.sample_rate * 0.1:
            return True, "너무 짧음"
        
        try:
            # Spectral Flatness (낮을수록 톤성 강함 -> 비프음 등)
            flatness = librosa.feature.spectral_flatness(y=segment)[0]
            mean_flatness = np.mean(flatness)
            
            # Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.sample_rate)[0]
            mean_bandwidth = np.mean(bandwidth)
            
            score = 0
            reasons = []
            
            if mean_flatness < 0.01:
                score -= 2
                reasons.append("순수톤")
            elif mean_flatness < 0.05:
                score -= 1
            else:
                score += 1
            
            if mean_bandwidth < 500:
                score -= 2
                reasons.append("좁은대역")
            elif mean_bandwidth < 1000:
                score -= 1
            else:
                score += 1
            
            return score > 0, ", ".join(reasons) if reasons else "정상"
        except Exception as e:
            logger.debug(f"Error in _is_human_voice: {e}")
            return True, "검사 실패"
    
    def _is_scream_intensity(self, segment: np.ndarray) -> Tuple[bool, str]:
        """
        비명 강도 분석 (말소리와 비명 구분)
        
        Args:
            segment: 오디오 세그먼트
            
        Returns:
            (is_intense, reason) 튜플
        """
        if len(segment) < self.sample_rate * 0.1:
            return False, "너무 짧음"
        
        try:
            score = 0
            reasons = []
            
            # 1. RMS (에너지)
            rms = librosa.feature.rms(y=segment)[0]
            mean_rms = np.mean(rms)
            if mean_rms > 0.08:
                score += 2
                reasons.append("매우큼")
            elif mean_rms > 0.05:
                score += 1
                reasons.append("큼")
            
            # 2. 주파수 대역 비율 (비명은 1.6k~4k 대역 에너지 집중)
            stft = np.abs(librosa.stft(segment, n_fft=2048))
            freq_bins = stft.shape[0]
            low = np.mean(stft[:int(freq_bins*0.2), :])
            mid = np.mean(stft[int(freq_bins*0.2):int(freq_bins*0.5), :])  # Scream band
            high = np.mean(stft[int(freq_bins*0.5):, :])
            
            ratio = mid / (low + high + 1e-6)
            if ratio > 1.5:
                score += 2
                reasons.append("비명대역")
            elif ratio > 1.0:
                score += 1
            
            # 3. 피치 (높은음)
            pitches, magnitudes = librosa.piptrack(y=segment, sr=self.sample_rate, fmin=100, fmax=4000)
            pitch_vals = []
            for t in range(pitches.shape[1]):
                idx = magnitudes[:, t].argmax()
                p = pitches[idx, t]
                if p > 100:
                    pitch_vals.append(p)
            
            if len(pitch_vals) > 5:
                mean_pitch = np.mean(pitch_vals)
                if mean_pitch > 600:
                    score += 2
                    reasons.append("초고음")
                elif mean_pitch > 400:
                    score += 1
                    reasons.append("고음")
            
            # 5점 이상이어야 '강렬한 비명'으로 인정
            return score >= 5, ", ".join(reasons) if reasons else "약함"
        except Exception as e:
            logger.debug(f"Error in _is_scream_intensity: {e}")
            return False, "검사 실패"

    def predict(self, audio_data: np.ndarray, sr: Optional[int] = None) -> Dict[str, Any]:
        """
        오디오 데이터를 받아 비명 여부 판정 (상세한 결과 반환)
        
        Args:
            audio_data: 오디오 데이터 (numpy array)
            sr: 샘플링 레이트 (다를 경우 16000으로 리샘플링됨)
            
        Returns:
            dict: {
                "is_scream": bool,  # 최종 비명 판정
                "prob": float,      # 모델 확률 (0.0 ~ 1.0)
                "status": str,      # "SCREAM", "SAFE", "SPEECH", "FILTERED"
                "reason": str       # 판정 이유
            }
        """
        # 리샘플링
        if sr and sr != self.sample_rate:
            try:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
            except Exception as e:
                logger.warning(f"Resampling failed: {e}, using original audio")
        
        # 1. 기계음 필터링 (활성화된 경우)
        if self.enable_filtering:
            is_human, voice_reason = self._is_human_voice(audio_data)
            if not is_human:
                return {
                    "is_scream": False,
                    "prob": 0.0,
                    "status": "FILTERED",
                    "reason": f"기계음 감지 ({voice_reason})",
                    "confidence": 0.0,  # 기존 호환성
                    "threshold": self.threshold
                }
        
        # 2. 모델 추론
        try:
            feature = self._preprocess(audio_data)
            with torch.no_grad():
                outputs = self.model(feature)
                probabilities = F.softmax(outputs, dim=1)
                prob_scream = probabilities[0][1].item()
        except Exception as e:
            logger.error(f"Model inference error: {e}", exc_info=True)
            return {
                "is_scream": False,
                "prob": 0.0,
                "status": "SAFE",
                "reason": f"추론 오류: {str(e)}",
                "confidence": 0.0,
                "threshold": self.threshold
            }
        
        # 3. 비명 강도 필터링 (활성화된 경우, 확률이 높을 때만 체크)
        status = "SAFE"
        final_decision = False
        intensity_reason = ""
        
        if prob_scream > self.threshold:
            if self.enable_filtering:
                is_intense, intensity_reason = self._is_scream_intensity(audio_data)
                if is_intense:
                    final_decision = True
                    status = "SCREAM"
                else:
                    status = "SPEECH"  # 모델은 비명이라 했지만 강도가 약함 (말소리 등)
            else:
                # 필터링 비활성화 시 모델 확률만 사용
                final_decision = True
                status = "SCREAM"
        
        return {
            "is_scream": final_decision,
            "prob": prob_scream,
            "status": status,
            "reason": intensity_reason if intensity_reason else "정상",
            "confidence": prob_scream,  # 기존 호환성
            "threshold": self.threshold
        }
    
    def detect(self, audio: np.ndarray, sr: Optional[int] = None) -> Tuple[bool, float]:
        """
        오디오에서 비명 감지 (기존 호환성 유지)
        
        Args:
            audio: 오디오 데이터 (numpy array, float32, [-1, 1] 범위)
            sr: 샘플링 레이트 (선택사항)
            
        Returns:
            (is_scream, confidence) 튜플
            
        Note:
            최소 오디오 길이: 약 0.1초 (1600 샘플 @ 16kHz)
            권장 오디오 길이: 2.0초 (32000 샘플 @ 16kHz)
        """
        result = self.predict(audio, sr=sr)
        return result["is_scream"], result["prob"]

    def process(self, audio_data: np.ndarray, sr: Optional[int] = None) -> Dict[str, Any]:
        """
        오디오 데이터 처리 (AudioProcessor 인터페이스 구현)
        
        Args:
            audio_data: 오디오 데이터 (numpy array)
            sr: 샘플링 레이트 (선택사항)
            
        Returns:
            처리 결과 딕셔너리
        """
        return self.predict(audio_data, sr=sr)
