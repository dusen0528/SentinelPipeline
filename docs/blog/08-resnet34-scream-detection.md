# ResNet34 기반 비명 감지 모델

## 왜 딥러닝 CNN을 선택했나?

전통적인 특징 추출 방식(MFCC, Spectral Contrast 등)은 **사람이 직접 설계한 특징**을 사용합니다. 하지만 딥러닝 CNN은 **이미지 자체를 입력**으로 받아서, 모델이 스스로 중요한 패턴을 학습합니다.

**멜 스펙트로그램 이미지**를 CNN에 입력하면:

- 비명의 고주파 패턴을 시각적으로 인식
- 시간에 따른 주파수 변화 패턴을 학습
- 복잡한 비선형 패턴을 자동으로 추출

## 샘플링 레이트 선택: 22,050 Hz

**나이퀴스트 이론에 따라 22,050 Hz로 샘플링 진행하기**

- **이론적 근거**: 나이퀴스트 이론에 따르면 최대 주파수 = 샘플링 레이트 / 2
- **22,050 Hz 샘플링** → 최대 11,025 Hz까지 표현 가능
- **비명의 주요 주파수 대역**: 대부분 1,000 Hz ~ 8,000 Hz 범위
- **결론**: 22,050 Hz로도 비명 감지에 충분하며, 처리 속도와 메모리 효율이 더 좋음

### 샘플링 레이트에 따른 차이

- **만약 낮다면?**
  - 샘플이 부족하여 고주파 성분이 손실됨
  - 비명의 특징적인 고주파 패턴을 놓칠 수 있음
- **만약 보통이라면?**
  - 어느 정도 비슷하지만 아직 디테일이 부족
- **22,050 Hz 이상이라면?**
  - 샘플이 충분하여 비명의 고주파 패턴을 정확히 포착
  - 디지털 소리가 원본 소리와 거의 완벽하게 일치

## 오디오를 이미지로 변환: 멜 스펙트로그램

**주의: 샘플링 레이트 설정**
- 본 문서의 데이터셋 분석은 **22,050 Hz**를 기준으로 설명되었으나,
- 실제 프로덕션 파이프라인(`ScreamDetector`)은 **16,000 Hz**로 다운샘플링하여 추론합니다.
- 이는 STT 모델(Whisper)과의 호환성 및 처리 속도 최적화를 위함입니다.

```python
# 실제 코드 (infrastructure/audio/processors/scream_detector.py)
self.sample_rate = 16000 
```

### 1. 멜 스펙트로그램 변환

```python
# 멜 스펙트로그램 변환 (문서 예시용 코드)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,  # 실제 코드에서는 16000 사용
    n_mels=64,          # 멜 필터 뱅크 개수 (Y축 높이)
    n_fft=1024,         # FFT 윈도우 크기
    hop_length=512      # 윈도우 이동 간격
)


**멜 스펙트로그램 파라미터 설명**

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| sample_rate | 22,050 Hz | 오디오 샘플링 레이트 |
| n_mels | 64 | 멜 주파수 빈 개수 (세로축) |
| n_fft | 1024 | FFT 윈도우 크기 (주파수 해상도) |
| hop_length | 512 | 프레임 간격 (시간 해상도) |

**비명 탐지 시 특징 패턴**

- **비명 (Scream)**: 고주파 영역(위쪽)에 진한 불규칙한 패턴
  - 스펙트로그램 상단에 에너지가 집중
  - 시간에 따라 불규칙하게 변하는 패턴
  - 색깔이 진하고 지저분한 모양
- **일반 소리 (Noise)**: 저주파 영역(아래쪽)에 집중되거나 전체적으로 고르게 분포
  - 스펙트로그램 하단에 에너지가 집중
  - 시간에 따라 규칙적이거나 안정적인 패턴
  - 색깔이 연하고 매끄러운 모양

![멜 스펙트로그램 예시](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdna%2FbP846Y%2FdJMcabbJmoR%2FAAAAAAAAAAAAAAAAAAAAAMkOf_qG9T1R9nCNnOVQHhIDUS8KOmfcmMmBlsBjCywv%2Fimg.png%3Fcredential%3DyqXZFxpELC7KVnFOS48ylbz2pIh7yKj8%26expires%3D1767193199%26allow_ip%3D%26allow_referer%3D%26signature%3DzBv6pN4fFQnry%252BN9qYd%252FVlAf4IM%253D)

### 2. 로그 스케일 변환

**왜 로그 변환을 하나?**

- 인간의 청각은 로그 스케일로 소리를 인식
- 소리의 크기 차이가 매우 크기 때문에 (예: 1 ~ 1,000,000)
- 로그 변환으로 시각적 표현 개선

```python
# 로그 변환
log_spec = spectrogram_tensor.log2().numpy() + 1e-10  # 작은 값 추가하여 log(0) 방지
```

### 3. 이미지로 변환

**스펙트로그램을 이미지 파일로 저장**

- Viridis 컬러맵 사용 (녹색-노란색-파란색)
- PIL Image로 변환하여 ResNet34 입력 형식에 맞춤
- 크기: (64, 862) 픽셀

```python
# 스펙트로그램을 이미지로 저장
plt.imsave(image_path, log_spec, cmap='viridis')
image = Image.open(image_path)
```

## 데이터 전처리 파이프라인

### 1. 오디오 파일 로드

```python
# librosa로 오디오 로드
audio, sample_rate = librosa.load(file_path, sr=22050)
```

**주요 설정:**

- sr=22050: 샘플링 레이트를 22,050 Hz로 고정
- 모든 오디오 파일을 동일한 샘플링 레이트로 통일

### 2. 오디오 길이 정규화 (패딩/자르기)

**왜 필요한가?**

- ResNet34는 고정 크기 이미지를 입력으로 받음
- 모든 오디오를 동일한 길이로 맞춰야 함
- 20초 길이로 표준화

```python
def pad_waveform(waveform, target_length):
    """파형을 목표 길이로 패딩하거나 자르기"""
    if len(waveform) < target_length:
        padding = target_length - len(waveform)
        waveform = np.pad(waveform, (0, padding), mode='constant')
    elif len(waveform) > target_length:
        waveform = waveform[:target_length]
    return waveform
```

### 3. 멜 스펙트로그램 변환

```python
# 멜 스펙트로그램 생성
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64,
    n_fft=1024,
    hop_length=512
)
spectrogram = mel_transform(torch.tensor(waveform, dtype=torch.float32))
```

### 4. 로그 변환 및 정규화

```python
# 로그 변환
log_spec = torch.log2(spectrogram + 1e-10)

# 정규화 (0~1 범위로)
log_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min())
```

## ResNet34 모델 구조

### 1. ResNet34 아키텍처

**ResNet의 핵심 아이디어: Skip Connection**

- 깊은 네트워크에서 발생하는 그래디언트 소실 문제 해결
- 잔차(Residual) 학습으로 더 나은 성능 달성
- 34개 레이어로 구성된 중간 크기 모델

**비명 감지에 적합한 이유:**

- 이미지의 시각적 패턴을 자동으로 학습
- 비명의 고주파 패턴을 직접 인식
- 복잡한 비선형 패턴 추출 가능

### 2. 전이 학습 활용

**ImageNet 사전 학습 가중치:**

- 일반적인 이미지 패턴(엣지, 텍스처, 형태)을 이미 학습
- 스펙트로그램 이미지에도 적용 가능
- 적은 데이터로도 좋은 성능

### 3. 실시간 처리 최적화

**효율적인 처리 파이프라인:**

- 22,050 Hz 샘플링으로 메모리 효율
- 2초마다 예측으로 실시간성 확보
- 슬라이딩 윈도우로 연속 모니터링

## 전통적 특징 추출 vs 딥러닝 CNN 비교

| 방식 | 특징 | 장점 | 단점 |
|------|------|------|------|
| 전통적 특징 추출 (MFCC, Spectral Contrast 등) | 사람이 직접 설계한 특징 | 해석 가능, 빠른 처리 | 특징 설계에 의존, 복잡한 패턴 놓칠 수 있음 |
| 딥러닝 CNN (ResNet34) | 이미지에서 자동 특징 추출 | 복잡한 패턴 자동 발견, 높은 성능 | 많은 데이터 필요, 해석 어려움 |

**ResNet34의 장점:**

- 이미지의 시각적 패턴을 자동으로 학습
- 비명의 고주파 패턴을 직접 인식
- 복잡한 비선형 패턴 추출 가능

## 기술 스택

### 주요 라이브러리

- **PyTorch**: 딥러닝 프레임워크
- **torchaudio**: 오디오 처리 및 멜 스펙트로그램 변환
- **librosa**: 오디오 파일 로드 및 전처리
- **PIL (Pillow)**: 이미지 처리
- **matplotlib**: 스펙트로그램 시각화

## 결론

이 구현은 **딥러닝 CNN 기반 이미지 분류 방식**을 사용하여 비명을 감지합니다. 전통적인 특징 추출 방식과 달리, 모델이 스펙트로그램 이미지에서 직접 패턴을 학습하여 **95.22%의 높은 정확도**를 달성했습니다.

**핵심 아이디어:**

1. 오디오를 멜 스펙트로그램 이미지로 변환
2. ResNet34 CNN으로 이미지 패턴 학습
3. 실시간 슬라이딩 윈도우로 연속 모니터링

**실용성:**

- 실시간 비명 감지 가능
- 높은 정확도로 오경보 최소화
- 스펙트럼 이미지 저장으로 분석 및 보고서 작성 용이

---

**참고 자료:**
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch torchaudio Documentation](https://pytorch.org/audio/stable/index.html)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)

