"""
비명 감지 프로세서

ResNet18 기반 모델을 사용하여 오디오에서 비명을 감지합니다.
기계음 필터링 및 비명 강도 분석 기능을 포함합니다.
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Tuple, Any, Optional, Dict
import logging

from sentinel_pipeline.domain.interfaces.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class ScreamDetector(AudioProcessor):
    """
    ResNet18 기반 비명 감지기
    
    기계음 필터링 및 비명 강도 분석 기능을 포함합니다.
    """
    
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.6,
        device: str = "auto",
        model_arch: str = "resnet18",  # "resnet18" (ResNet34 레거시 지원 제거)
        enable_filtering: bool = True,  # 필터링 로직 활성화 여부
    ):
        """
        Args:
            model_path: 모델 가중치 파일 경로 (.pth)
            threshold: 비명 판정 임계값 (0.0 ~ 1.0)
            device: 디바이스 ('auto', 'cuda', 'cpu')
            model_arch: 모델 아키텍처 ('resnet18')
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
        
        # 모델 아키텍처 설정 (ResNet18만 지원)
        self.model_arch = "resnet18"
        
        
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(
            f"ScreamDetector initialized on {self.device} "
            f"(arch={self.model_arch}, threshold={threshold}, filtering={enable_filtering})"
        )

    def _load_model(self) -> nn.Module:
        """
        모델 아키텍처 로드 및 가중치 로드
        
        ResNet-ScreamDetect와 동일한 방식으로 로드:
        1. ImageNet 사전 학습 가중치로 초기화
        2. conv1을 1채널로 변경
        3. fc를 2클래스로 변경
        4. 학습된 가중치 로드
        """
        try:
            from torchvision import models
            
            # ResNet18 모델 로드 (ImageNet 사전 학습 가중치 사용 - ResNet-ScreamDetect와 동일)
            # 참고: 학습 시에도 ImageNet 가중치로 시작했을 가능성이 높음
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except (AttributeError, TypeError):
                # 구버전 torchvision 호환성
                try:
                    model = models.resnet18(pretrained=True)
                except:
                    # 최후의 수단: 가중치 없이 초기화
                    logger.warning("Could not load ImageNet weights, using random initialization")
                    model = models.resnet18(weights=None)
            
            # 1채널(Grayscale) 입력 수용 (ResNet-ScreamDetect와 동일)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # 출력 레이어 설정 (2개 클래스: Non-Scream, Scream)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found at {self.model_path}, using ImageNet weights only")
                return model.to(self.device)
            
            # 가중치 로드 (ResNet-ScreamDetect와 정확히 동일한 방식)
            # 원본: model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            try:
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=False)
                model.load_state_dict(state_dict)
                logger.info(f"Model weights loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                logger.warning("Using ImageNet weights only (model file may be corrupted)")
                
            return model.to(self.device)
        except Exception as e:
            logger.error(f"Error loading model architecture: {e}")
            raise

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
        # 참모 의견: 학습 시와 토씨 하나 안 틀리고 똑같이 적용해야 함
        melspec = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate,
            n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        
        # 정규화 (Min-Max Scaling) - 학습 시와 동일한 방식
        # 주의: 마이크 입력 감도(Gain)에 따라 RMS 값이 달라질 수 있음 (필터 임계값 튜닝 필요 가능성)
        min_val, max_val = melspec_db.min(), melspec_db.max()
        if max_val - min_val > 0:
            melspec_norm = (melspec_db - min_val) / (max_val - min_val)
        else:
            melspec_norm = melspec_db
        
        # [Batch, Channel, Height, Width] 형태로 변환
        tensor = torch.tensor(melspec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _is_human_voice(self, segment: np.ndarray) -> Tuple[bool, str]:
        """
        사람 목소리인지 기계음인지 판별 (제공된 코드의 정교한 버전)
        
        기계음(비프음) 특징:
        - Spectral Flatness가 매우 낮음 (단일 주파수)
        - Spectral Bandwidth가 매우 좁음
        - Zero-Crossing Rate가 매우 규칙적
        - Pitch 변화가 거의 없음
        
        Args:
            segment: 오디오 세그먼트
            
        Returns:
            (is_human, reason) 튜플
        """
        if len(segment) < self.sample_rate * 0.1:
            return True, "너무 짧음"
        
        #무음 체크 (0 나누기 에러 방지)
        if np.max(np.abs(segment)) < 0.001:
            return False, "무음"
        
        try:
            # 1. Spectral Flatness (0에 가까우면 톤성, 1에 가까우면 노이즈)
            flatness = librosa.feature.spectral_flatness(y=segment)[0]
            mean_flatness = np.mean(flatness)
            
            # 2. Spectral Bandwidth (주파수 폭)
            bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.sample_rate)[0]
            mean_bandwidth = np.mean(bandwidth)
            
            # 3. Spectral Centroid 변화량 (피치 변화)
            centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)[0]
            centroid_std = np.std(centroid)
            
            # 4. Zero-Crossing Rate 변화량 (규칙성)
            zcr = librosa.feature.zero_crossing_rate(segment)[0]
            zcr_std = np.std(zcr)
            
            # 판정 로직
            reasons = []
            score = 0
            
            # 비프음 특징: 매우 낮은 flatness (순수 톤)
            if mean_flatness < 0.01:
                reasons.append(f"순수톤(flatness={mean_flatness:.4f})")
                score -= 2
            elif mean_flatness < 0.05:
                reasons.append(f"톤성 강함(flatness={mean_flatness:.4f})")
                score -= 1
            else:
                score += 1
            
            # 비프음 특징: 좁은 bandwidth
            if mean_bandwidth < 500:
                reasons.append(f"좁은주파수(bw={mean_bandwidth:.0f}Hz)")
                score -= 2
            elif mean_bandwidth < 1000:
                score -= 1
            else:
                score += 1
            
            # 비프음 특징: centroid 변화 적음 (일정한 피치)
            if centroid_std < 100:
                reasons.append(f"피치불변(std={centroid_std:.0f})")
                score -= 1
            else:
                score += 1
            
            # 비프음 특징: zcr 변화 적음 (규칙적)
            if zcr_std < 0.01:
                reasons.append(f"규칙적(zcr_std={zcr_std:.4f})")
                score -= 1
            else:
                score += 1
            
            # 최종 판정
            is_voice = score > 0
            reason = ", ".join(reasons) if reasons else "정상 음성"
            
            return is_voice, reason
        except Exception as e:
            logger.debug(f"Error in _is_human_voice: {e}")
            return True, "검사 실패"
    
    def _is_scream_intensity(self, segment: np.ndarray) -> Tuple[bool, str]:
        """
        비명 강도 분석 - 일반 말소리와 비명을 구분 (엄격한 기준)
        
        비명의 핵심 특징:
        - 매우 높은 절대 에너지 (크게 소리침)
        - 지속적인 고주파 에너지 (비명은 계속 높은 음)
        - 급격한 onset (갑자기 시작)
        - 높은 피치 (일반 말소리보다 높음)
        
        Args:
            segment: 오디오 세그먼트
            
        Returns:
            (is_intense, reason) 튜플
        """
        if len(segment) < self.sample_rate * 0.1:
            return False, "너무 짧음"
        
        # 무음 체크 (0 나누기 에러 방지)
        if np.max(np.abs(segment)) < 0.001:
            return False, "무음"
        
        try:
            score = 0
            reasons = []
            
            # 1. 절대 에너지 수준 (비명은 RMS > 0.05 이상)
            rms = librosa.feature.rms(y=segment)[0]
            mean_rms = np.mean(rms)
            max_rms = np.max(rms)
            
            if mean_rms > 0.08:  # 매우 큰 소리
                score += 2
                reasons.append(f"큰소리(rms={mean_rms:.3f})")
            elif mean_rms > 0.05:  # 큰 소리
                score += 1
                reasons.append(f"소리큼(rms={mean_rms:.3f})")
            
            # 2. 지속적인 고에너지 (비명은 계속 크게)
            # RMS의 최소값도 높아야 함 (잠깐 큰 소리가 아니라 계속 큰 소리)
            min_rms = np.min(rms[rms > 0.01]) if np.any(rms > 0.01) else 0
            sustained_ratio = min_rms / (max_rms + 1e-6)
            
            if sustained_ratio > 0.3:  # 지속적으로 큰 소리
                score += 1
                reasons.append(f"지속적({sustained_ratio:.2f})")
            
            # 3. 고주파 집중도 (비명은 1000-4000Hz에 에너지 집중)
            stft = np.abs(librosa.stft(segment, n_fft=2048))
            freq_bins = stft.shape[0]
            
            # 주파수 대역별 에너지
            low_band = np.mean(stft[:int(freq_bins*0.2), :])   # 0-1600Hz
            mid_band = np.mean(stft[int(freq_bins*0.2):int(freq_bins*0.5), :])  # 1600-4000Hz (비명 대역)
            high_band = np.mean(stft[int(freq_bins*0.5):, :])  # 4000Hz+
            
            # 비명 대역 비율
            scream_band_ratio = mid_band / (low_band + high_band + 1e-6)
            
            if scream_band_ratio > 1.5:  # 비명 대역에 에너지 집중
                score += 2
                reasons.append(f"비명대역({scream_band_ratio:.2f})")
            elif scream_band_ratio > 1.0:
                score += 1
                reasons.append(f"중고주파({scream_band_ratio:.2f})")
            
            # 4. 높은 평균 피치 (비명은 보통 500Hz 이상)
            pitches, magnitudes = librosa.piptrack(y=segment, sr=self.sample_rate, fmin=100, fmax=4000)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 100:  # 유효한 피치만
                    pitch_values.append(pitch)
            
            if len(pitch_values) > 5:
                mean_pitch = np.mean(pitch_values)
                if mean_pitch > 600:  # 매우 높은 피치
                    score += 2
                    reasons.append(f"고피치({mean_pitch:.0f}Hz)")
                elif mean_pitch > 400:  # 높은 피치
                    score += 1
                    reasons.append(f"피치높음({mean_pitch:.0f}Hz)")
            
            # 5. Spectral Rolloff (비명은 에너지가 고주파까지 분포)
            rolloff = librosa.feature.spectral_rolloff(y=segment, sr=self.sample_rate, roll_percent=0.85)[0]
            mean_rolloff = np.mean(rolloff)
            
            if mean_rolloff > 4000:  # 에너지가 4000Hz 이상까지 분포
                score += 1
                reasons.append(f"롤오프({mean_rolloff:.0f}Hz)")
            
            # 최종 판정: 5점 이상이어야 비명 (더 엄격)
            # 최대 가능 점수: 2+1+2+2+1 = 8점
            is_intense = score >= 5
            reason = ", ".join(reasons) if reasons else "일반 음성"
            
            return is_intense, reason
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
        
        # 0. 무음(Silence) 체크 0 나누기 에러 방지
        # 아주 작은 소리는 아예 계산 안 함 (GPU 전송 전에 차단)
        max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0.0
        if max_amplitude < 0.001:
            logger.warning(f"무음 필터링: max_amplitude={max_amplitude:.6f} (너무 낮음)")
            return {
                "is_scream": False,
                "prob": 0.0,
                "status": "FILTERED",
                "reason": "무음(Silence)",
                "confidence": 0.0,
                "threshold": self.threshold
            }
        
        # 1. 기계음 필터링 (활성화된 경우) - 원본과 동일: 필터링되어도 모델 추론은 실행
        filtered_by_voice = False
        voice_reason = ""
        if self.enable_filtering:
            is_human, voice_reason = self._is_human_voice(audio_data)
            if not is_human:
                filtered_by_voice = True
                logger.warning(f"기계음 필터링: {voice_reason}")
                # 원본 코드는 필터링되어도 prob는 반환하지만, is_scream은 False
                # 여기서는 모델 추론을 건너뛰고 바로 반환 (원본과 약간 다름)
                return {
                    "is_scream": False,
                    "prob": 0.0,  # 원본도 필터링 시 prob=0.0 반환
                    "status": "FILTERED",
                    "reason": f"기계음 감지 ({voice_reason})",
                    "confidence": 0.0,
                    "threshold": self.threshold
                }
        
        # 2. 모델 추론 (원본과 동일)
        try:
            feature = self._preprocess(audio_data)
            with torch.no_grad():
                outputs = self.model(feature)
                probabilities = F.softmax(outputs, dim=1)
                prob_scream = probabilities[0][1].item()
                logger.warning(f"🤖 모델 추론 결과: prob_scream={prob_scream:.4f}, threshold={self.threshold}, is_scream={prob_scream > self.threshold}")
            
            # GPU 메모리 정리: 추론 후 텐서를 CPU로 이동 후 삭제 (더 확실한 정리)
            if self.device == "cuda":
                # GPU 작업 완료 대기
                torch.cuda.synchronize()
                # 텐서를 CPU로 이동 후 삭제 (GPU 메모리 즉시 해제)
                if feature.is_cuda:
                    feature = feature.cpu()
                if outputs.is_cuda:
                    outputs = outputs.cpu()
                if probabilities.is_cuda:
                    probabilities = probabilities.cpu()
                del feature, outputs, probabilities
                # Python 가비지 컬렉터도 호출
                gc.collect()
                # GPU 캐시 정리
                torch.cuda.empty_cache()
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
        
        # 3. 비명 강도 필터링 (원본과 동일: prob_scream > THRESHOLD일 때만 체크)
        final_is_scream = prob_scream > self.threshold
        intensity_reason = ""
        
        if self.enable_filtering and prob_scream > self.threshold:
            is_intense, intensity_reason = self._is_scream_intensity(audio_data)
            if not is_intense:
                final_is_scream = False  # 원본: final_is_scream = False
        
        # 원본과 동일: prob는 항상 반환, is_scream만 필터링으로 결정
        status = "SCREAM" if final_is_scream else "SAFE"
        
        return {
            "is_scream": final_is_scream,
            "prob": prob_scream,  # 원본과 동일: 필터링되어도 실제 모델 확률 반환
            "status": status,
            "reason": intensity_reason if intensity_reason else "정상",
            "confidence": prob_scream,
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
