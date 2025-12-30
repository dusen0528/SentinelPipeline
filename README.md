# SentinelPipeline

<div align="center">

**실시간 AI 영상·오디오 처리를 위한 모듈형 파이프라인 엔진**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-orange.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20Architecture-purple.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

</div>

---

## 📋 프로젝트 개요

**SentinelPipeline**은 RTSP(Real-Time Streaming Protocol) 다중 스트림을 입력받아 실시간으로 AI 기반 위험 감지(화재, 비명, 침입, 긴급 키워드 등)와 영상 변형(얼굴 블러 등)을 수행하는 모듈형 파이프라인 엔진입니다.

### 🎯 핵심 가치

- **모듈형 파이프라인**: 감지 및 변형 모듈을 플러그인 방식으로 추가/제거 가능
- **Clean Architecture**: 4계층 분리로 유지보수성과 확장성 확보
- **설정 기반 운영**: 코드 수정 없이 설정 파일만으로 동작 제어
- **장애 격리**: 개별 모듈의 오류가 전체 파이프라인에 영향 없음
- **자동 복구**: 스트림 끊김 시 지수 백오프 방식으로 자동 재연결

---

## ✨ 주요 기능

### 🎥 비디오 스트림 처리

- **RTSP 다중 스트림 지원**: 여러 카메라 스트림을 동시에 처리
- **실시간 AI 감지**: 화재, 침입 등 위험 상황 실시간 감지
- **영상 변형**: 얼굴 블러 등 프라이버시 보호 기능
- **FFmpeg 기반 퍼블리싱**: 처리된 영상을 RTSP로 재스트리밍

### 🎤 오디오 스트림 처리

- **비명 감지**: ResNet34 기반 딥러닝 모델로 비명 소리 실시간 감지 (95.22% 정확도)
- **STT 통합**: Whisper 모델을 활용한 음성-텍스트 변환
- **하이브리드 키워드 감지**: 3단계 계층형 아키텍처로 긴급 키워드 감지
  - **Fast Path**: 해시맵 기반 O(1) 조회 (90% 이상 케이스 처리)
  - **Medium Path**: Kiwi 형태소 분석 + Levenshtein 거리 (오타/어미 변형 처리)
  - **Heavy Path**: Sentence Transformers 기반 의미 유사도 (문맥적 매칭)
- **RTSP/Microphone 지원**: RTSP 오디오 스트림 및 마이크 입력 지원

### 🖥️ 통합 대시보드

- **실시간 모니터링**: 스트림 상태, 시스템 리소스, 성능 메트릭 실시간 표시
- **비디오/오디오 분리 관리**: 각각의 전용 인터페이스 제공
- **WebSocket 실시간 업데이트**: 이벤트 및 상태 변경 실시간 반영
- **스트림 제어**: REST API를 통한 스트림 시작/중지/재시작

---

## 🏗️ 아키텍처

### Clean Architecture 4계층 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      Interface Layer                        │
│   (REST API, WebSocket, 대시보드, 설정 로더, 메트릭)          │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│   (RTSP 디코더, FFmpeg, 오디오 리더/프로세서, AI 모델, HTTP/WS)│
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                         │
│   (파이프라인 엔진, 스케줄러, 스트림 관리자, 이벤트 발행자)     │
├─────────────────────────────────────────────────────────────┤
│                      Domain Layer                            │
│   (모듈 인터페이스, Event 모델, Stream 모델)                  │
└─────────────────────────────────────────────────────────────┘
```

### 의존성 규칙

> 소스 코드 의존성은 반드시 안쪽(고수준)을 향해야 합니다.

- 바깥쪽 계층은 안쪽 계층을 참조할 수 있지만, 안쪽 계층은 바깥쪽 계층을 알지 못합니다
- 각 계층을 독립적으로 테스트하고 확장할 수 있습니다
- 비즈니스 로직과 외부 의존성을 명확히 분리합니다

---

## 🛠️ 기술 스택

### 언어 및 프레임워크

- **Python 3.10+**: 메인 개발 언어
- **FastAPI**: 고성능 비동기 REST API 서버
- **uv**: 빠른 패키지 관리 (pyproject.toml 기반)

### 비디오 처리

- **OpenCV**: RTSP 스트림 디코딩
- **FFmpeg**: 영상 인코딩 및 퍼블리싱
- **ONNX Runtime / PyTorch**: AI 모델 추론

### 오디오 처리

- **PyTorch / torchaudio**: 딥러닝 모델 실행 및 오디오 처리
- **faster-whisper**: 고성능 음성-텍스트 변환
- **librosa**: 오디오 특징 추출 및 전처리
- **PyAudio**: 마이크 입력 처리

### 자연어 처리 (한국어)

- **kiwipiepy**: C++ 기반 고성능 한국어 형태소 분석기 (Java 의존성 없음)
- **python-Levenshtein**: 문자열 편집 거리 계산
- **sentence-transformers**: 문장 임베딩 및 의미 유사도 계산

### 기타

- **loguru**: 구조화 로깅
- **Pydantic v2**: 설정 검증 및 타입 안전성
- **WebSocket**: 실시간 이벤트 전송
- **Prometheus**: 메트릭 수집 및 모니터링

---

## 🎯 핵심 기능 상세

### 1. 하이브리드 키워드 감지 시스템

한국어의 교착어 특성과 STT 오인식을 고려한 3단계 계층형 아키텍처:

```
[Whisper STT] → 텍스트
    ↓
[HybridKeywordDetector]
    ├─ [1단계: ExpandedKeywordDict] → O(1) 조회
    │   └─ 매칭 성공 → 즉시 반환 (90% 케이스)
    │
    ├─ [2단계: MorphologicalMatcher] → Kiwi + Levenshtein
    │   └─ 매칭 성공 → 반환 (오타/어미 변형)
    │
    └─ [3단계: SemanticSimilarityMatcher] → 임베딩 + 코사인
        └─ 비동기 처리 (후처리 분석)
```

**주요 특징:**
- **실시간성**: 90% 이상의 케이스가 1ms 이내에 판별
- **정확도**: Kiwi의 정규화 덕분에 STT 오인식 상황에서도 강건하게 키워드 감지
- **배포 간소화**: Java 의존성 제거로 Docker 이미지 크기 감소

### 2. ResNet34 기반 비명 감지

- **모델**: ImageNet 사전 학습 ResNet34
- **입력**: Mel Spectrogram (22,050 Hz 샘플링)
- **정확도**: 95.22%
- **처리 주기**: 2초마다 예측으로 실시간성 확보

### 3. 모듈형 파이프라인 엔진

- **플러그인 시스템**: 새로운 감지 모듈을 코드 수정 없이 추가 가능
- **우선순위 기반 실행**: 모듈별 priority 설정으로 실행 순서 제어
- **타임아웃 관리**: 개별 모듈의 처리 시간 제한
- **자동 비활성화**: 연속 오류 발생 시 모듈 자동 비활성화

---

## 📊 시스템 구조

### 데이터 흐름

#### 비디오 스트림 파이프라인
```
[RTSP Source] → [RTSPDecoder] → [PipelineEngine] → [FFmpegPublisher] → [RTSP Output]
                                      │
                                      └─> [EventEmitter] → [VMS/WebSocket]
```

#### 오디오 스트림 파이프라인
```
[RTSP/Mic] → [AudioReader] → [AudioManager]
                                ├─> [ScreamDetector] → 비명 감지 이벤트
                                └─> [RiskAnalyzer]
                                     ├─> [Whisper STT] → 텍스트 변환
                                     └─> [HybridKeywordDetector] → 키워드 감지
```

---

## 🎨 대시보드 기능

### 통합 대시보드
- **시스템 개요**: 총 스트림 수, 활성 스트림, 시스템 리소스, 성능 메트릭
- **실시간 모니터링**: CPU/메모리 사용률, 평균 FPS, AI 모델 상태

### 비디오 관리
- **스트림 목록**: 모든 비디오 스트림의 상태 조회
- **스트림 제어**: 시작/중지/재시작/삭제
- **실시간 프리뷰**: 스트림 영상 실시간 확인

### 오디오 모니터
- **스트림 목록**: 모든 오디오 스트림의 상태 조회
- **실시간 분석 결과**: 비명 감지, STT 결과, 키워드 감지 이벤트
- **파이프라인 상태**: 각 처리 단계별 상태 표시

---

## 📚 기술 문서

프로젝트의 상세한 기술 문서와 블로그 포스트:

### 아키텍처 문서
- [아키텍처 설계 원칙](docs/architecture.md)
- [Domain Layer 설계](docs/blog/02-domain-layer-design.md)
- [Application Layer 설계](docs/blog/03-application-layer-design.md)
- [Infrastructure Layer 설계](docs/blog/04-infrastructure-layer-design.md)
- [Interface Layer 설계](docs/blog/05-interface-api-design.md)

### 기술 블로그
- [오디오 데이터 전처리 (Librosa)](docs/blog/07-audio-librosa-preprocessing.md)
- [ResNet34 기반 비명 감지](docs/blog/08-resnet34-scream-detection.md)
- [하이브리드 키워드 감지 파이프라인](docs/blog/09-hybrid-keyword-detection-pipeline.md)

---

## 🎓 설계 원칙

### Clean Architecture
- **의존성 규칙**: 안쪽 계층으로의 단방향 의존성
- **계층 분리**: Domain → Application → Infrastructure → Interface
- **인터페이스 기반 설계**: Protocol을 통한 느슨한 결합

### SOLID 원칙
- **Single Responsibility**: 각 모듈은 하나의 책임만 가짐
- **Dependency Inversion**: 고수준 모듈이 저수준 모듈에 의존하지 않음
- **Open/Closed**: 확장에는 열려있고 수정에는 닫혀있음

---

## 📈 성능 최적화

### 오디오 처리
- **캐싱 전략**: 형태소 분석 결과 LRU 캐싱
- **배치 처리**: 의미 기반 매칭은 배치로 처리하여 GPU 효율성 향상
- **비동기 처리**: Heavy Path는 실시간 블로킹 없이 백그라운드 처리

### 비디오 처리
- **멀티스레딩**: 스트림별 독립 스레드로 병렬 처리
- **프레임 드롭**: FPS 제한으로 리소스 효율성 확보
- **자동 재연결**: 지수 백오프 방식으로 네트워크 장애 복구

---

## 🔧 주요 API

### 비디오 스트림 API
- `GET /api/video/streams` - 스트림 목록 조회
- `POST /api/video/streams/{id}/start` - 스트림 시작
- `POST /api/video/streams/{id}/stop` - 스트림 중지
- `GET /api/video/streams/by-input` - 입력 URL로 출력 URL 조회

### 오디오 스트림 API
- `GET /api/audio/streams` - 오디오 스트림 목록 조회
- `POST /api/audio/streams` - 오디오 스트림 등록 및 시작
- `DELETE /api/audio/streams/{id}` - 오디오 스트림 삭제
- `GET /api/audio/streams/{id}/status` - 오디오 스트림 상태 조회

### WebSocket
- `ws://localhost:8000/ws/admin` - 실시간 이벤트 수신

---

## 🚀 주요 성과

- ✅ **모듈형 아키텍처**: 코드 수정 없이 설정만으로 기능 제어
- ✅ **실시간 처리**: 비디오/오디오 스트림 동시 처리
- ✅ **높은 정확도**: 비명 감지 95.22%, 키워드 감지 3단계 하이브리드 시스템
- ✅ **확장성**: 플러그인 방식으로 새로운 모듈 추가 용이
- ✅ **안정성**: 장애 격리 및 자동 복구 메커니즘

---


<div align="center">

**SentinelPipeline** - 실시간 AI 기반 위험 감지 시스템


</div>
