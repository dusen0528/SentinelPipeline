import torch
import torch.nn as nn
from torchvision import models
import librosa
import numpy as np
import os

class ScreamDetector:
    def __init__(self, model_path, device=None, threshold=0.6):
        """
        초기화 및 모델 로드
        :param model_path: .pth 모델 가중치 파일 경로
        :param device: 'cuda' or 'cpu' (None일 경우 자동 설정)
        :param threshold: 비명 판정 임계값 (기본 0.6)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sr = 16000
        self.window_sec = 2.0
        self.threshold = threshold
        
        # 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
        self.model = self._load_model_architecture()
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"모델 가중치 로드 실패: {e}")
            
        self.model.to(self.device)
        self.model.eval()
        print(f"[ScreamDetector] 모델 로드 완료 ({self.device})")

    def _load_model_architecture(self):
        """ResNet18 기반 커스텀 아키텍처 정의"""
        model = models.resnet18(weights=None)
        # 1채널(Grayscale) 입력 수용
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 2개 클래스(Non-Scream, Scream) 출력
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        return model

    def _preprocess(self, y):
        """오디오 전처리: 길이 맞춤 -> Mel-Spectrogram -> 정규화 -> 텐서"""
        target_len = int(self.sr * self.window_sec)
        
        # 길이 맞추기 (패딩 또는 자르기)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Mel-Spectrogram 변환
        melspec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, 
            n_mels=64, n_fft=1024, hop_length=512
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
        return tensor.to(self.device)

    def _is_human_voice(self, segment):
        """기계음/비프음 필터링 (True면 사람 목소리/환경음 가능성 높음)"""
        if len(segment) < self.sr * 0.1:
            return True, "너무 짧음"
            
        # Spectral Flatness (낮을수록 톤성 강함 -> 비프음 등)
        flatness = librosa.feature.spectral_flatness(y=segment)[0]
        mean_flatness = np.mean(flatness)
        
        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.sr)[0]
        mean_bandwidth = np.mean(bandwidth)
        
        score = 0
        reasons = []
        
        if mean_flatness < 0.01:
            score -= 2; reasons.append("순수톤")
        elif mean_flatness < 0.05:
            score -= 1
        else:
            score += 1
            
        if mean_bandwidth < 500:
            score -= 2; reasons.append("좁은대역")
        elif mean_bandwidth < 1000:
            score -= 1
        else:
            score += 1
            
        return score > 0, ", ".join(reasons)

    def _is_scream_intensity(self, segment):
        """비명 강도 분석 (말소리와 비명 구분)"""
        if len(segment) < self.sr * 0.1:
            return False, "너무 짧음"
            
        score = 0
        reasons = []
        
        # 1. RMS (에너지)
        rms = librosa.feature.rms(y=segment)[0]
        mean_rms = np.mean(rms)
        if mean_rms > 0.08: score += 2; reasons.append("매우큼")
        elif mean_rms > 0.05: score += 1; reasons.append("큼")
        
        # 2. 주파수 대역 비율 (비명은 1.6k~4k 대역 에너지 집중)
        stft = np.abs(librosa.stft(segment, n_fft=2048))
        freq_bins = stft.shape[0]
        low = np.mean(stft[:int(freq_bins*0.2), :])
        mid = np.mean(stft[int(freq_bins*0.2):int(freq_bins*0.5), :]) # Scream band
        high = np.mean(stft[int(freq_bins*0.5):, :])
        
        ratio = mid / (low + high + 1e-6)
        if ratio > 1.5: score += 2; reasons.append("비명대역")
        elif ratio > 1.0: score += 1
        
        # 3. 피치 (높은음)
        pitches, magnitudes = librosa.piptrack(y=segment, sr=self.sr, fmin=100, fmax=4000)
        pitch_vals = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            p = pitches[idx, t]
            if p > 100: pitch_vals.append(p)
            
        if len(pitch_vals) > 5:
            mean_pitch = np.mean(pitch_vals)
            if mean_pitch > 600: score += 2; reasons.append("초고음")
            elif mean_pitch > 400: score += 1; reasons.append("고음")

        # 5점 이상이어야 '강렬한 비명'으로 인정
        return score >= 5, ", ".join(reasons)

    def predict(self, audio_data, sr=None):
        """
        오디오 데이터(numpy array)를 받아 비명 여부 판정
        :param audio_data: numpy array (오디오 신호)
        :param sr: 샘플링 레이트 (다를 경우 16000으로 리샘플링됨)
        :return: dict (결과 정보)
        """
        # 리샘플링
        if sr and sr != self.sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sr)
            
        # 1. 기계음 필터링
        is_human, voice_reason = self._is_human_voice(audio_data)
        if not is_human:
            return {
                "is_scream": False,
                "prob": 0.0,
                "status": "FILTERED",
                "reason": f"기계음 감지 ({voice_reason})"
            }

        # 2. 모델 추론
        with torch.no_grad():
            feature = self._preprocess(audio_data)
            outputs = self.model(feature)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prob_scream = probabilities[0][1].item()

        # 3. 비명 강도 필터링 (확률이 높을 때만 체크)
        status = "SAFE"
        final_decision = False
        intensity_reason = ""
        
        if prob_scream > self.threshold:
            is_intense, intensity_reason = self._is_scream_intensity(audio_data)
            if is_intense:
                final_decision = True
                status = "SCREAM"
            else:
                status = "SPEECH" # 모델은 비명이라 했지만 강도가 약함 (말소리 등)
        
        return {
            "is_scream": final_decision,
            "prob": prob_scream,
            "status": status,
            "reason": intensity_reason
        }
