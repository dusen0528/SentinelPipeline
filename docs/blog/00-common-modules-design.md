# 공통 모듈

## 들어가며

공통 모듈은 프로젝트 전체에서 사용하는 기반 기능을 제공합니다. 에러 처리, 로깅, 타입 검사 등은 모든 계층에서 필요하므로, 프로젝트 초기에 구축하는 것이 중요합니다.

이 글에서는 왜 이런 설계를 선택했는지, 어떤 고민을 했는지 정리했습니다.

---

## 공통 모듈의 역할

### 왜 공통 모듈이 필요한가?

프로젝트 전체에서 일관된 에러 처리와 로깅이 필요합니다. 각 계층마다 다른 방식으로 에러를 처리하거나 로그를 남기면:

1. **디버깅이 어려움**: 로그 형식이 달라 추적이 어려움
2. **에러 처리 불일치**: 같은 에러를 다르게 처리
3. **운영 복잡도 증가**: 로그 집계 시스템과 연동이 어려움

공통 모듈을 통해:

- **일관된 에러 처리**: 모든 계층에서 동일한 예외 클래스 사용
- **구조화 로깅**: JSON 형식으로 로그 집계 시스템과 연동
- **컨텍스트 추적**: trace_id, stream_id, module_name 자동 포함

---

## 에러 처리 체계 설계

### ErrorCode Enum 정의

먼저 모든 에러를 분류하기 위해 `ErrorCode` Enum을 정의했습니다:

```python
class ErrorCode(str, Enum):
    # 모듈 관련 (5xx)
    MODULE_TIMEOUT = "MODULE_TIMEOUT"
    MODULE_FAILED = "MODULE_FAILED"
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"

    # 스트림 관련 (4xx, 5xx)
    STREAM_NOT_FOUND = "STREAM_NOT_FOUND"
    STREAM_ALREADY_RUNNING = "STREAM_ALREADY_RUNNING"
    STREAM_CONNECTION_FAILED = "STREAM_CONNECTION_FAILED"

    # 설정 관련 (4xx)
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"

    # 인증/권한 관련 (4xx)
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"

    # 전송 관련 (5xx)
    TRANSPORT_FAILED = "TRANSPORT_FAILED"
    TRANSPORT_TIMEOUT = "TRANSPORT_TIMEOUT"

    # ... 총 23개 에러 코드

```

**`str`을 상속받은 이유**:

- JSON 직렬화 시 문자열로 변환됨
- REST API 응답에 바로 사용 가능
- 로그에 기록할 때도 문자열로 출력

### HTTP 상태 코드 매핑

각 에러 코드에 해당하는 HTTP 상태 코드를 매핑합니다:

```python
_ERROR_CODE_TO_HTTP_STATUS: dict[ErrorCode, int] = {
    # 4xx Client Errors
    ErrorCode.STREAM_NOT_FOUND: 404,
    ErrorCode.CONFIG_INVALID: 400,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.STREAM_ALREADY_RUNNING: 409,

    # 5xx Server Errors
    ErrorCode.MODULE_TIMEOUT: 504,
    ErrorCode.MODULE_FAILED: 500,
    ErrorCode.STREAM_CONNECTION_FAILED: 500,
    # ...
}

def get_http_status(error_code: ErrorCode) -> int:
    return _ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)

```

이렇게 하면 REST API에서 예외를 잡아 적절한 HTTP 상태 코드로 변환할 수 있습니다.

### 예외 클래스 계층

예외 클래스를 계층적으로 설계했습니다:

```python
class SentinelError(Exception):
    """기본 예외 클래스"""
    def __init__(self, code: ErrorCode, message: str, details: dict = None):
        self.code = code
        self.message = message
        self.details = details or {}

    @property
    def http_status(self) -> int:
        return get_http_status(self.code)

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }

class ModuleError(SentinelError):
    """모듈 관련 예외"""
    def __init__(self, code, message, module_name, details=None):
        self.module_name = module_name
        _details = {"module_name": module_name}
        if details:
            _details.update(details)
        super().__init__(code, message, _details)

class StreamError(SentinelError):
    """스트림 관련 예외"""
    def __init__(self, code, message, stream_id, details=None):
        self.stream_id = stream_id
        # ...

```

**계층 구조의 장점**:

- 특정 타입의 예외만 잡을 수 있음 (`except ModuleError:`)
- 각 예외 타입에 특화된 정보 포함 (module_name, stream_id 등)
- REST API에서 예외 타입별로 다른 처리 가능

### 사용 예시

```python
# 모듈 타임아웃
raise ModuleError(
    ErrorCode.MODULE_TIMEOUT,
    f"모듈 처리 시간 초과: {module_name}",
    module_name=module_name,
    details={"timeout_ms": 50, "actual_ms": 75}
)

# 스트림 없음
raise StreamError(
    ErrorCode.STREAM_NOT_FOUND,
    f"스트림을 찾을 수 없습니다: {stream_id}",
    stream_id=stream_id
)

```

---

## 구조화 로깅 시스템

### 왜 구조화 로깅인가?

일반적인 로그는 다음과 같습니다:

```
2024-12-10 10:23:45 ERROR 모듈 처리 실패
```

이런 로그는:

- 로그 집계 시스템(ELK, Loki 등)에서 검색이 어려움
- 필터링이 어려움 (특정 스트림의 로그만 찾기 어려움)
- 통계 분석이 어려움

구조화 로깅은 JSON 형식으로 로그를 남깁니다:

```json
{
  "timestamp": "2024-12-10T10:23:45.123Z",
  "level": "ERROR",
  "module_name": "FireDetectModule",
  "stream_id": "camera_01",
  "trace_id": "abc123",
  "message": "모듈 처리 실패",
  "error": "timeout",
  "latency_ms": 52
}
```

이렇게 하면:

- 로그 집계 시스템에서 쉽게 검색/필터링 가능
- `stream_id`로 특정 스트림의 로그만 조회 가능
- `trace_id`로 요청 추적 가능

### loguru 선택 이유

Python의 표준 `logging` 모듈 대신 `loguru`를 선택한 이유:

1. **설정이 간단함**: 기본 설정만으로도 충분히 사용 가능
2. **구조화 로깅 지원**: JSON 형식 출력 가능
3. **컨텍스트 바인딩**: `logger.bind()`로 컨텍스트 추가 용이
4. **예외 처리**: `logger.exception()`으로 자동 스택 트레이스

### 컨텍스트 변수 (Context Variables)

요청별로 trace_id를 추적하기 위해 Python의 `contextvars`를 사용합니다:

```python
from contextvars import ContextVar

_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_stream_id_var: ContextVar[str | None] = ContextVar("stream_id", default=None)
_module_name_var: ContextVar[str | None] = ContextVar("module_name", default=None)

def get_trace_id() -> str:
    trace_id = _trace_id_var.get()
    if trace_id is None:
        trace_id = generate_trace_id()
        _trace_id_var.set(trace_id)
    return trace_id
```

**contextvars의 장점**:

- 스레드 안전: 각 스레드가 독립적인 컨텍스트를 가짐
- 비동기 지원: async/await에서도 컨텍스트가 유지됨
- 자동 전파: 함수 호출 시 자동으로 전파됨

### BoundLogger 구현

컨텍스트가 자동으로 포함되는 로거를 구현했습니다:

```python
class BoundLogger:
    def __init__(self, name: str, module_name: str = None, stream_id: str = None):
        self._name = name
        self._module_name = module_name
        self._stream_id = stream_id
        self._logger = logger.bind(name=name)

    def _get_extra(self, **kwargs):
        extra = _get_context_extra()  # contextvars에서 가져옴

        # 인스턴스 레벨 컨텍스트
        if self._module_name:
            extra["module_name"] = self._module_name
        if self._stream_id:
            extra["stream_id"] = self._stream_id

        extra.update(kwargs)
        return extra

    def info(self, message: str, **kwargs):
        self._logger.bind(**self._get_extra(**kwargs)).info(message)

```

**사용 예시**:

```python
# 모듈에서 사용
module_logger = get_logger(__name__, module_name="FireDetectModule")
module_logger.info("화재 감지 모듈 초기화")

# 출력:
# {
#   "timestamp": "...",
#   "level": "INFO",
#   "module_name": "FireDetectModule",
#   "message": "화재 감지 모듈 초기화"
# }

# 스트림 처리에서 사용
stream_logger = get_logger(__name__, stream_id="camera_01")
stream_logger.info("프레임 처리 시작", frame_number=12345)

# 출력:
# {
#   "timestamp": "...",
#   "level": "INFO",
#   "stream_id": "camera_01",
#   "frame_number": 12345,
#   "message": "프레임 처리 시작"
# }

```

### JSON vs Console 포맷

운영 환경과 개발 환경에서 다른 포맷을 사용합니다

**운영 환경 (JSON)**:

```json
{"timestamp":"2024-12-10T10:23:45.123Z","level":"INFO","module_name":"FireDetect","message":"처리 완료"}

```

**개발 환경 (Console)**:

```
2024-12-10 10:23:45.123 | INFO     | FireDetect: 처리 완료 | module=FireDetect

```

환경변수로 제어합니다:

```bash
LOG_FORMAT=json  # JSON 형식
LOG_FORMAT=console  # 컬러 콘솔 형식 (기본)

```

---

## 타입 체커 및 린터 설정

### mypy: 정적 타입 검사

Python은 동적 타입 언어이지만, 타입 힌트를 사용하면 타입 안전성을 확보할 수 있습니다. `mypy`로 타입 체크를 수행합니다:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true

```

**strict 모드의 장점**:

- 모든 함수에 타입 힌트 필수
- 암묵적 타입 변환 금지
- 타입 안전성 최대화

**단점**:

- 초기 개발 시 타입 힌트 작성에 시간 소요
- 하지만 장기적으로 유지보수 비용 감소

### ruff: 린터 + 포맷터

`ruff`는 Python의 빠른 린터이자 포맷터입니다. `flake8`, `isort`, `black` 등을 대체할 수 있습니다:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "F",      # pyflakes
    "I",      # isort
    "N",      # pep8-naming
    "W",      # pycodestyle warnings
    "UP",     # pyupgrade
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "SIM",    # flake8-simplify
]
ignore = [
    "E501",   # line too long (formatter가 처리)
    "B008",   # function call in default argument (FastAPI Depends)
]

```

**ruff의 장점**:

- 매우 빠름 (Rust로 작성됨)
- 하나의 도구로 린팅과 포맷팅 모두 처리
- CI/CD에서 빠르게 실행 가능

### 사용 방법

```bash
# 타입 체크
uv run mypy src/sentinel_pipeline

# 린트
uv run ruff check src/sentinel_pipeline

# 포맷팅
uv run ruff format src/sentinel_pipeline

```

---

## 설계 결정 과정

### 1. 예외 클래스 계층 vs 단일 예외 클래스

**선택**: 계층 구조

**이유**:

- 각 예외 타입에 특화된 정보 포함 가능 (module_name, stream_id 등)
- 특정 타입의 예외만 잡을 수 있음
- REST API에서 예외 타입별로 다른 처리 가능

**단점**: 클래스 수가 많아짐 (하지만 관리 가능한 수준)

### 2. loguru vs 표준 logging

**선택**: loguru

**이유**:

- 설정이 간단함
- 구조화 로깅 지원
- 컨텍스트 바인딩이 용이
- 예외 처리 편의성

**단점**: 외부 의존성 추가 (하지만 가치가 충분함)

### 3. contextvars vs threading.local

**선택**: contextvars

**이유**:

- 비동기 코드에서도 동작 (향후 확장성)
- 스레드 안전
- 표준 라이브러리

### 4. mypy strict vs 일반 모드

**선택**: strict 모드

**이유**:

- 타입 안전성 최대화
- 초기 개발 시 타입 힌트를 확실히 작성
- 장기적으로 유지보수 비용 감소

**단점**: 개발 속도가 약간 느려질 수 있음 (하지만 가치가 충분함)

---

## 구현 결과

### 파일 구조

```
src/sentinel_pipeline/common/
├── __init__.py
├── errors.py          # 317줄 - 예외 클래스, ErrorCode
└── logging.py         # 326줄 - 구조화 로깅, BoundLogger

```

### 사용 예시

```python
from sentinel_pipeline.common.errors import ModuleError, ErrorCode
from sentinel_pipeline.common.logging import get_logger

# 로거 생성
logger = get_logger(__name__, module_name="FireDetectModule")

try:
    # 모듈 처리
    result = module.process_frame(frame, metadata)
except Exception as e:
    # 에러 로깅
    logger.error("모듈 처리 실패", error=str(e), latency_ms=52)

    # 예외 발생
    raise ModuleError(
        ErrorCode.MODULE_FAILED,
        f"모듈 처리 실패: {module.name}",
        module_name=module.name,
        details={"error": str(e)}
    )

```

---

## 테스트 예시

공통 모듈은 외부 의존성이 적어 테스트가 간단합니다:

```python
def test_error_code_to_http_status():
    assert get_http_status(ErrorCode.STREAM_NOT_FOUND) == 404
    assert get_http_status(ErrorCode.MODULE_TIMEOUT) == 504
    assert get_http_status(ErrorCode.UNKNOWN_ERROR) == 500  # 기본값

def test_module_error():
    error = ModuleError(
        ErrorCode.MODULE_TIMEOUT,
        "타임아웃 발생",
        module_name="FireDetectModule",
        details={"timeout_ms": 50}
    )

    assert error.code == ErrorCode.MODULE_TIMEOUT
    assert error.module_name == "FireDetectModule"
    assert error.http_status == 504
    assert error.to_dict()["code"] == "MODULE_TIMEOUT"

def test_logger_context():
    logger = get_logger(__name__, module_name="TestModule")

    # 로그에 module_name이 자동 포함되는지 확인
    with capture_logs() as logs:
        logger.info("테스트 메시지")

    assert "TestModule" in logs[0]

```

---

## 마무리

공통 모듈은 프로젝트의 기반이 됩니다. 일관된 에러 처리와 구조화 로깅을 통해:

1. **디버깅 용이성**: trace_id로 요청 추적 가능
2. **운영 편의성**: 로그 집계 시스템과 연동 용이
3. **타입 안전성**: mypy로 타입 오류 사전 방지
4. **코드 품질**: ruff로 일관된 코드 스타일 유지

이러한 기반 위에 Domain Layer, Application Layer를 구축하게 됩니다.

---

## 참고 자료

- [프로젝트 공통 모듈 코드](https://github.com/dusen0528/SentinelPipeline/tree/dev/src/sentinel_pipeline/common)