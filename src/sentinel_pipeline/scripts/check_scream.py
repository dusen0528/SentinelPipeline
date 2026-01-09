"""
비명 감지 모델 디버깅 스크립트

scream_test1.wav가 모델에 들어갈 때 Mel-Spectrogram이 정상적으로 그려지는지 확인합니다.
모델이 실제로 무엇을 보고 있는지 눈으로 확인할 수 있습니다.
"""

import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from sentinel_pipeline.infrastructure.audio.processors.scream_detector import ScreamDetector
from sentinel_pipeline.common.logging import get_logger

logger = get_logger(__name__)


def inspect_audio(file_path: str, model_path: str = None):
    """
    오디오 파일을 분석하고 Mel-Spectrogram을 시각화합니다.
    
    Args:
        file_path: 분석할 오디오 파일 경로
        model_path: 모델 가중치 파일 경로 (None이면 기본 경로 사용)
    """
    print(f"🔍 Inspecting: {file_path}")
    
    # 1. 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_path is None:
        # 기본 모델 경로
        model_path = project_root / "models" / "audio" / "resnet18_scream_detector_v2.pth"
    
    if not Path(model_path).exists():
        print(f"⚠️ Model file not found: {model_path}")
        print("   Using detector without model weights (for visualization only)")
        detector = None
    else:
        detector = ScreamDetector(
            model_path=str(model_path),
            device=device,
            enable_filtering=False,  # 디버깅용으로 필터링 비활성화
        )
        print(f"✅ Model loaded on {device}")

    # 2. 오디오 로드 (Librosa 사용 - 모델과 동일한 방식)
    target_sr = 16000  # ScreamDetector의 sample_rate와 동일
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
        print(f"📊 Audio Info: SR={sr}, Length={len(y)} samples ({len(y)/sr:.2f} sec)")
    except Exception as e:
        print(f"❌ Failed to load audio: {e}")
        return
    
    # 길이 맞추기 (2초 - 모델 입력 규격)
    target_len = target_sr * 2
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant', constant_values=0.0)
        print(f"   Padded to {len(y)} samples")
    else:
        y = y[:target_len]
        print(f"   Trimmed to {len(y)} samples")

    # 3. 전처리 및 Mel-Spectrogram 변환 (ScreamDetector 내부 로직 모사)
    # 실제 코드에서 _preprocess 메서드가 사용하는 파라미터와 동일하게
    n_mels = 64
    n_fft = 1024
    hop_length = 512
    
    print(f"\n🎵 Generating Mel-Spectrogram...")
    print(f"   Parameters: n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}")
    
    try:
        # Mel-Spectrogram 변환 (ScreamDetector와 동일)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=target_sr,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 정규화 (Min-Max Scaling) - 학습 시와 동일한 방식
        min_val, max_val = mel_spec_db.min(), mel_spec_db.max()
        if max_val - min_val > 0:
            mel_spec_norm = (mel_spec_db - min_val) / (max_val - min_val)
        else:
            mel_spec_norm = mel_spec_db
        
        print(f"   Mel-Spectrogram shape: {mel_spec_norm.shape}")
        print(f"   Value range: [{mel_spec_norm.min():.3f}, {mel_spec_norm.max():.3f}]")
        
        # 4. 시각화 (여기가 핵심)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 원본 Mel-Spectrogram (dB 스케일)
        ax1 = axes[0]
        img1 = librosa.display.specshow(
            mel_spec_db, sr=target_sr, 
            x_axis='time', y_axis='mel',
            hop_length=hop_length, ax=ax1
        )
        ax1.set_title(f'Mel-Spectrogram (dB): {Path(file_path).name}', fontsize=12, fontweight='bold')
        plt.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # 정규화된 Mel-Spectrogram (모델 입력)
        ax2 = axes[1]
        img2 = librosa.display.specshow(
            mel_spec_norm, sr=target_sr,
            x_axis='time', y_axis='mel',
            hop_length=hop_length, ax=ax2
        )
        ax2.set_title('Normalized Mel-Spectrogram (Model Input)', fontsize=12, fontweight='bold')
        plt.colorbar(img2, ax=ax2, format='%.2f')
        
        plt.tight_layout()
        
        output_img = f"debug_{Path(file_path).stem}.png"
        plt.savefig(output_img, dpi=150, bbox_inches='tight')
        print(f"\n🖼️ Saved spectrogram image to: {output_img}")
        print("👉 이 이미지를 열어보세요.")
        print("   - 까맣거나, 노이즈만 보이거나, 주파수가 잘려있으면 전처리 문제입니다.")
        print("   - 비명은 고주파 영역(위쪽)에 진한 불규칙한 패턴이 보여야 합니다.")
        
        plt.close()

    except Exception as e:
        print(f"❌ Visualization Error: {e}")
        import traceback
        traceback.print_exc()

    # 5. 실제 추론 결과 확인
    if detector is not None:
        print("\n🤖 Running Inference...")
        try:
            result = detector.predict(y)
            print(f"🎯 Prediction Result:")
            print(f"   is_scream: {result.get('is_scream', False)}")
            print(f"   prob: {result.get('prob', 0.0):.4f}")
            print(f"   threshold: {detector.threshold}")
            
            if result.get('prob', 0.0) < detector.threshold:
                print(f"   ⚠️ Low probability! Check if preprocessing matches training data.")
        except Exception as e:
            print(f"❌ Inference Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️ Skipping inference (model not loaded)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="비명 감지 모델 디버깅 도구")
    parser.add_argument(
        "audio_file",
        type=str,
        help="분석할 오디오 파일 경로"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="모델 가중치 파일 경로 (기본: models/audio/resnet18_scream_detector_v2.pth)"
    )
    
    args = parser.parse_args()
    
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        sys.exit(1)
    
    inspect_audio(str(audio_path), args.model)
