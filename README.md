# SentinelPipeline

실시간 AI 영상 처리를 위한 모듈형 파이프라인 엔진

![SentinelPipeline 개요](docs/imgae/image.png)

---

## 아키텍처


![아키텍처 구조](docs/imgae/image2.png)

![계층별 설명](docs/imgae/image3.png)


## REST API

FastAPI 기반의 HTTP API를 통해 외부에서 시스템을 제어할 수 있습니다.

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 시스템 상태 확인 |
| `/streams` | GET | 전체 스트림 목록 조회 |
| `/streams/{id}/start` | POST | 스트림 시작 |
| `/streams/{id}/stop` | POST | 스트림 중지 |
| `/config` | GET | 현재 설정 조회 |
| `/config` | PUT | 설정 동적 변경 |
| `/metrics` | GET | Prometheus 메트릭 |

---

## 관리자 대시보드

웹 기반 대시보드를 통해 시스템을 시각적으로 모니터링하고 제어할 수 있습니다.

| 페이지 | 경로 | 설명 |
|--------|------|------|
| 메인 대시보드 | `/admin` | 시스템 개요, 스트림/모듈 상태 요약 |
| 스트림 관리 | `/admin/streams` | 스트림 목록, 시작/중지/재시작 제어 |
| 모듈 관리 | `/admin/modules` | 모듈 상태, 활성화/비활성화, 설정 변경 |
| 이벤트 로그 | `/admin/events` | 실시간 이벤트 목록, 필터링 |
| 설정 | `/admin/settings` | 전역 설정 변경 |

---

## 개발 로드맵

1. 파이프라인 프레임워크 구축 (현재 단계)
2. REST API 서버 구현
3. 관리자 대시보드 구현
4. 화재 감지 모델 개발 및 통합
5. 비명 감지 모델 개발 및 통합
6. 침입 감지 모델 개발 및 통합
7. 얼굴 블러 모델 고도화
8. 성능 최적화 및 안정화

---

## 문서

- [아키텍처 설명](docs/architecture.md)
- [모듈 개발 가이드](docs/modules.md)
- [설정 가이드](docs/configuration.md)
- [문제 해결 가이드](docs/troubleshooting.md)
