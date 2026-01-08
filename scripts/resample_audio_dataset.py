"""
오디오 데이터셋 리샘플링 스크립트

test/dataset의 모든 .wav 파일을 16000Hz로 리샘플링하여 tests/assets/에 저장합니다.
CPU 환경에서 런타임 리샘플링을 피하기 위해 사전에 리샘플링된 파일을 준비합니다.
"""

import os
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# 설정
TARGET_SAMPLE_RATE = 16000
SOURCE_DIR = Path("test/dataset")
TARGET_DIR = Path("tests/assets")

def resample_audio_file(source_path: Path, target_path: Path, target_sr: int = 16000):
    """
    오디오 파일을 리샘플링하여 저장합니다.
    
    Args:
        source_path: 원본 파일 경로
        target_path: 저장할 파일 경로
        target_sr: 목표 샘플링 레이트
    """
    try:
        # 오디오 파일 로드
        audio, sr = librosa.load(str(source_path), sr=None, mono=True)
        
        # 이미 목표 샘플링 레이트와 같으면 리샘플링 불필요
        if sr == target_sr:
            # 그냥 복사
            sf.write(str(target_path), audio, target_sr)
            return True, sr, "copied (already correct sample rate)"
        
        # 리샘플링
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # 저장
        target_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target_path), audio_resampled, target_sr)
        
        return True, sr, f"resampled from {sr}Hz to {target_sr}Hz"
    except Exception as e:
        return False, None, str(e)

def main():
    """메인 함수"""
    print(f"오디오 데이터셋 리샘플링 시작")
    print(f"원본 디렉토리: {SOURCE_DIR}")
    print(f"대상 디렉토리: {TARGET_DIR}")
    print(f"목표 샘플링 레이트: {TARGET_SAMPLE_RATE}Hz\n")
    
    # 카테고리별 처리
    categories = ["scream", "noise"]
    
    total_files = 0
    success_count = 0
    error_count = 0
    
    for category in categories:
        source_category_dir = SOURCE_DIR / category
        target_category_dir = TARGET_DIR / category
        
        if not source_category_dir.exists():
            print(f"[WARNING] {source_category_dir} 디렉토리가 존재하지 않습니다.")
            continue
        
        # .wav 파일 찾기
        wav_files = list(source_category_dir.glob("*.wav"))
        
        if not wav_files:
            print(f"[WARNING] {category} 카테고리에 .wav 파일이 없습니다.")
            continue
        
        print(f"\n[{category}] 카테고리 처리 중... ({len(wav_files)}개 파일)")
        
        for source_file in tqdm(wav_files, desc=f"  {category}"):
            total_files += 1
            target_file = target_category_dir / source_file.name
            
            success, original_sr, message = resample_audio_file(
                source_file, 
                target_file, 
                TARGET_SAMPLE_RATE
            )
            
            if success:
                success_count += 1
                if "copied" not in message:
                    print(f"  [OK] {source_file.name}: {message}")
            else:
                error_count += 1
                print(f"  [ERROR] {source_file.name}: 오류 - {message}")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print(f"리샘플링 완료")
    print(f"{'='*60}")
    print(f"전체 파일: {total_files}개")
    print(f"성공: {success_count}개")
    print(f"실패: {error_count}개")
    print(f"\n리샘플링된 파일 위치: {TARGET_DIR}")
    
    if error_count > 0:
        print(f"\n[WARNING] 일부 파일 처리 중 오류가 발생했습니다.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

