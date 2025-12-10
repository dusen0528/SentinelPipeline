# Domain Layer

## 들어가며

Domain Layer는 Clean Architecture의 가장 안쪽 계층으로, **순수한 비즈니스 규칙과 엔티티**를 정의합니다. 이 계층은 외부 라이브러리에 전혀 의존하지 않으며, 표준 라이브러리만 사용합니다. 

---

## Domain Layer의 역할

### 왜 Domain Layer가 필요한가?

Domain Layer는 시스템의 **핵심 비즈니스 로직**을 담고 있습니다. 이 계층이 없으면:

1. **비즈니스 규칙이 분산됨**: 각 계층마다 비즈니스 로직이 섞여 있어 변경 시 영향 범위가 넓어짐
2. **테스트가 어려움**: 외부 의존성 때문에 단위 테스트 작성이 복잡해짐
3. **재사용이 어려움**: 다른 프로젝트에서 핵심 로직을 재사용하기 어려움

Domain Layer를 분리함으로써:

- **비즈니스 규칙의 단일 진실 공급원(Single Source of Truth)** 확보
- **외부 의존성 없이 테스트** 가능
- **다른 프로젝트에서도 재사용** 가능

### Domain Layer의 구성 요소

```
src/sentinel_pipeline/domain/
├── interfaces/
│   └── module.py          # ModuleBase Protocol
└── models/
    ├── event.py           # Event, EventType, EventStage
    └── stream.py          # StreamConfig, StreamState, StreamStatus

```

---

## ModuleBase Protocol 설계

### 설계 목표

파이프라인에서 실행되는 모든 모듈(감지, 변형 등)이 일관된 인터페이스를 가지도록 해야 합니다. 이를 위해 Python의 `Protocol`을 사용했습니다.

### 초기 설계 고민

처음에는 `@property` 기반으로 Protocol을 설계했습니다:

```python
@runtime_checkable
class ModuleBase(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def enabled(self) -> bool: ...
    @enabled.setter
    def enabled(self, value: bool) -> None: ...

```

하지만 이 방식은 플러그인 개발자가 단순 속성만 정의할 때 타입 체커에서 불일치가 발생할 수 있었습니다.

### 최종 설계: 속성 기반 + BaseModule 제공

**ModuleBase Protocol (속성 기반)**:

```python
@runtime_checkable
class ModuleBase(Protocol):
    name: str
    enabled: bool
    priority: int
    timeout_ms: int
    options: dict[str, Any]

    def process_frame(self, frame, metadata) -> tuple[FrameType, list[Event], MetadataType]: ...
    def process_audio(self, chunk, metadata) -> tuple[list[Event], MetadataType]: ...

```

**BaseModule 베이스 클래스 (편의성 제공)**:

```python
class BaseModule:
    name: str
    enabled: bool = True
    priority: int
    timeout_ms: int
    options: dict[str, Any]

    def __init__(self, **options):
        # 필수 속성 검증
        if not hasattr(self, "name"):
            raise TypeError(f"{self.__class__.__name__}는 'name' 속성을 정의해야 합니다")
        # ...

    def process_frame(self, frame, metadata):
        return frame, [], metadata  # 기본 구현

    def process_audio(self, chunk, metadata):
        return [], metadata  # 기본 구현

```

이렇게 하면:

- **Protocol 직접 구현**: 유연성 (필요한 경우)
- **BaseModule 상속**: 편의성 (대부분의 경우)

### ModuleBase 속성 설명

| 속성 | 타입 | 설명 |
| --- | --- | --- |
| `name` | `str` | 모듈 고유 식별자 (config.json과 일치) |
| `enabled` | `bool` | 활성화 여부 (런타임 변경 가능) |
| `priority` | `int` | 실행 우선순위 (낮을수록 먼저 실행) |
| `timeout_ms` | `int` | 처리 제한 시간 (밀리초) |
| `options` | `dict[str, Any]` | 모듈별 설정 옵션 |

**priority 권장 범위**:

- 0~99: 전처리 (크기 조정, 정규화)
- 100~199: 감지 (화재, 침입, 비명)
- 200~299: 변형 (블러, 마스킹)
- 300~399: 후처리 (오버레이, 워터마크)

### ModuleContext: 모듈 실행 상태 추적

모듈의 실행 상태를 추적하기 위해 `ModuleContext` 클래스를 추가했습니다:

```python
class ModuleContext:
    def __init__(self, module: ModuleBase):
        self.module = module
        self.error_count: int = 0
        self.timeout_count: int = 0
        self.total_processed: int = 0
        self.total_events: int = 0
        self.avg_latency_ms: float = 0.0

    def should_disable(self, max_errors: int = 5, max_timeouts: int = 10) -> bool:
        return self.error_count >= max_errors or self.timeout_count >= max_timeouts

```

이를 통해 파이프라인 엔진이 모듈의 상태를 모니터링하고, 문제가 발생하면 자동으로 비활성화할 수 있습니다.

---

## Event 모델 설계

### Event의 역할

파이프라인 모듈이 생성하는 감지 이벤트를 나타냅니다. 이 이벤트는 VMS(Video Management System)로 전송되어 알람, 녹화, 정책 적용 등에 사용됩니다.

### EventType: 이벤트 유형 정의

처음에는 문자열로 이벤트 유형을 표현하려고 했지만, 타입 안전성을 위해 Enum을 사용했습니다:

```python
class EventType(str, Enum):
    # 화재 관련
    FIRE = "FIRE"
    SMOKE = "SMOKE"

    # 침입 관련
    INTRUSION = "INTRUSION"
    LOITERING = "LOITERING"
    PERIMETER = "PERIMETER"

    # 음향 관련
    SCREAM = "SCREAM"
    GUNSHOT = "GUNSHOT"
    GLASS_BREAK = "GLASS_BREAK"

    # 객체 관련
    FACE_DETECTED = "FACE_DETECTED"
    PERSON_DETECTED = "PERSON_DETECTED"
    VEHICLE_DETECTED = "VEHICLE_DETECTED"

    # ... 총 18개 유형

```

`str`을 상속받아 Enum을 사용하면서도 JSON 직렬화 시 문자열로 변환됩니다.

### EventStage: 이벤트 단계

이벤트의 확정 수준을 나타냅니다:

```python
class EventStage(str, Enum):
    DETECTED = "DETECTED"       # 최초 감지 (단일 프레임)
    CONFIRMED = "CONFIRMED"     # 확정 (연속 감지 조건 충족)
    CLEARED = "CLEARED"         # 해제 (이벤트 종료)

```

예를 들어, 화재 감지 모듈은:

1. 첫 프레임에서 화재를 감지하면 `DETECTED` 이벤트 생성
2. 연속 3프레임 이상 감지되면 `CONFIRMED` 이벤트 생성
3. 더 이상 감지되지 않으면 `CLEARED` 이벤트 생성

### Event 클래스 구현

```python
@dataclass
class Event:
    type: EventType
    stage: EventStage
    confidence: float
    stream_id: str
    module_name: str
    latency_ms: float = 0.0
    ts: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: _generate_event_id())

    def __post_init__(self):
        # 유효성 검사
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence는 0.0~1.0 범위여야 합니다: {self.confidence}")

```

**주요 특징**:

- `dataclass` 사용으로 간결한 코드
- `__post_init__`에서 유효성 검사
- `to_dict()` / `from_dict()`로 JSON 직렬화 지원
- `event_id` 자동 생성 (타임스탬프 + 랜덤)

### BoundingBox: 바운딩 박스 지원

객체 감지 모듈에서 사용하는 바운딩 박스를 별도 클래스로 분리했습니다:

```python
@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def to_xyxy(self) -> list[int]:
        return [self.x, self.y, self.x2, self.y2]

```

`Event` 클래스에서 `bbox` 속성으로 자동 변환도 지원합니다.

### EventFilter: 이벤트 필터링

이벤트 로그 조회 시 필터링을 위한 클래스입니다:

```python
@dataclass
class EventFilter:
    types: list[EventType] | None = None
    stages: list[EventStage] | None = None
    stream_ids: list[str] | None = None
    min_confidence: float = 0.0
    start_ts: float | None = None
    end_ts: float | None = None

    def matches(self, event: Event) -> bool:
        # 필터 조건 확인
        if self.start_ts is not None and event.ts < self.start_ts:
            return False
        # ...

```

**주의사항**: `start_ts`와 `end_ts`를 `is not None`으로 비교해야 합니다. `if self.start_ts:`로 하면 `start_ts=0` 같은 값이 무시됩니다.

---

## Stream 모델 설계

### StreamConfig: 스트림 설정

RTSP 스트림 연결 및 처리에 필요한 설정을 정의합니다:

```python
@dataclass
class StreamConfig:
    stream_id: str
    rtsp_url: str
    enabled: bool = True
    max_fps: int = 15
    downscale: float = 1.0
    buffer_size: int = 2

    # 재연결 설정
    reconnect_enabled: bool = True
    reconnect_max_retries: int = 5
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 8.0

```

**유효성 검사**: `__post_init__`에서 설정값의 유효성을 검사합니다:

- `max_fps > 0`
- `0.0 < downscale <= 1.0`
- `buffer_size >= 1`

**보안**: `to_dict()`에서 RTSP URL의 비밀번호를 마스킹합니다:

```python
rtsp://admin:password@192.168.1.100:554/stream1
→ rtsp://admin:****@192.168.1.100:554/stream1

```

### StreamStatus: 스트림 상태

스트림의 현재 동작 상태를 나타냅니다:

```python
class StreamStatus(str, Enum):
    IDLE = "IDLE"                   # 유휴 상태
    STARTING = "STARTING"           # 시작 중
    RUNNING = "RUNNING"             # 실행 중
    STOPPING = "STOPPING"           # 중지 중
    STOPPED = "STOPPED"             # 중지됨
    ERROR = "ERROR"                 # 오류 발생
    RECONNECTING = "RECONNECTING"   # 재연결 중

```

상태 전이를 명확히 정의하여 스트림 관리 로직을 단순화했습니다.

### StreamStats: 실시간 통계

스트림 처리의 실시간 통계를 관리합니다:

```python
@dataclass
class StreamStats:
    frame_count: int = 0
    event_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0
    fps: float = 0.0
    avg_latency_ms: float = 0.0
    last_frame_ts: float | None = None
    start_ts: float | None = None

    def record_frame(self, latency_ms: float):
        # FPS 계산 (최근 1초 기준)
        self._fps_frame_times.append(time.time())
        cutoff = time.time() - 1.0
        self._fps_frame_times = [t for t in self._fps_frame_times if t > cutoff]
        self.fps = len(self._fps_frame_times)

```

**FPS 계산**: 최근 1초 동안의 프레임 수를 카운트하여 실시간 FPS를 계산합니다. 이 방식은 이동 평균보다 더 정확한 현재 상태를 반영합니다.

### StreamState: 종합 상태

`StreamConfig`, `StreamStatus`, `StreamStats`를 하나로 묶은 종합 상태입니다:

```python
@dataclass
class StreamState:
    config: StreamConfig
    status: StreamStatus = StreamStatus.IDLE
    stats: StreamStats = field(default_factory=StreamStats)
    last_error: str | None = None
    retry_count: int = 0
    next_retry_ts: float | None = None

    def calculate_next_retry_delay(self) -> float:
        """지수 백오프 계산"""
        delay = self.config.reconnect_base_delay * (2 ** self.retry_count)
        return min(delay, self.config.reconnect_max_delay)

```

**지수 백오프**: 재연결 시도 간격을 점진적으로 늘려 서버 부하를 줄입니다.

- 1차 시도: 1초
- 2차 시도: 2초
- 3차 시도: 4초
- 4차 시도: 8초 (최대)

---

## 설계 결정 과정

### 1. Protocol vs Abstract Base Class

**선택**: Protocol 사용

**이유**:

- Python 3.8+의 `typing.Protocol`은 구조적 서브타이핑을 지원
- 상속 없이도 인터페이스를 만족하는 클래스를 만들 수 있음
- `@runtime_checkable`로 런타임 체크도 가능

**하지만**: 편의성을 위해 `BaseModule` 베이스 클래스도 제공

### 2. Enum vs 문자열 상수

**선택**: `str`을 상속받은 Enum

**이유**:

- 타입 안전성 확보
- IDE 자동완성 지원
- JSON 직렬화 시 문자열로 변환됨 (`str` 상속)

### 3. dataclass vs Pydantic

**선택**: `dataclass` 사용

**이유**:

- Domain Layer는 외부 라이브러리 의존 금지
- `dataclass`는 표준 라이브러리
- 유효성 검사는 `__post_init__`에서 처리

**단점**: Pydantic처럼 자동 검증은 없지만, Domain Layer의 순수성을 유지하기 위해 선택

### 4. 직렬화 메서드

**선택**: `to_dict()` / `from_dict()` 메서드 제공

**이유**:

- REST API 응답에 사용
- VMS 전송 시 JSON 변환 필요
- 설정 파일 저장/로드 시 사용

---

## 구현 결과

### 파일 구조

```
src/sentinel_pipeline/domain/
├── __init__.py
├── interfaces/
│   ├── __init__.py
│   └── module.py          # ModuleBase Protocol, BaseModule, ModuleContext
└── models/
    ├── __init__.py
    ├── event.py           # Event, EventType, EventStage, BoundingBox, EventFilter
    └── stream.py          # StreamConfig, StreamState, StreamStatus, StreamStats

```

### 타입 체커 통과

```bash
$ mypy --strict src/sentinel_pipeline/domain
Success: no issues found

```

---

## 테스트 예시

Domain Layer는 외부 의존성이 없어 테스트가 간단합니다:

```python
def test_event_creation():
    event = Event(
        type=EventType.FIRE,
        stage=EventStage.DETECTED,
        confidence=0.87,
        stream_id="camera_01",
        module_name="FireDetectModule",
    )

    assert event.type == EventType.FIRE
    assert event.confidence == 0.87
    assert event.event_id is not None

def test_event_serialization():
    event = Event(...)
    data = event.to_dict()
    restored = Event.from_dict(data)

    assert restored.type == event.type
    assert restored.confidence == event.confidence

def test_stream_retry_delay():
    config = StreamConfig(
        stream_id="cam_01",
        rtsp_url="rtsp://...",
        reconnect_base_delay=1.0,
        reconnect_max_delay=8.0,
    )
    state = StreamState(config=config)

    # 1차 시도
    state.retry_count = 0
    assert state.calculate_next_retry_delay() == 1.0

    # 2차 시도
    state.retry_count = 1
    assert state.calculate_next_retry_delay() == 2.0

    # 최대값 제한
    state.retry_count = 10
    assert state.calculate_next_retry_delay() == 8.0

```

---

## 참고사항

- [프로젝트 Domain Layer 코드](https://github.com/dusen0528/SentinelPipeline/tree/dev/src/sentinel_pipeline/domain)