# Interface Layer - REST API 설계

## 들어가며

이번에는 Interface Layer에 REST API 어댑터를 추가하고, 설정 로더를 확장했습니다. 핵심 목표는 Clean Architecture 경계를 지키면서 외부 세계(HTTP 클라이언트)와의 통신 경계를 명확히 하는 것입니다.


---

## Interface Layer의 역할

### 왜 Interface Layer가 필요한가?

Clean Architecture에서 Interface Layer는 외부 세계와의 경계를 담당합니다. 외부 라이브러리(FastAPI, Pydantic)에 대한 의존성을 이 계층에만 집중시켜, 내부 계층(Domain, Application, Infrastructure)이 외부 변화에 영향받지 않도록 합니다.

**Interface Layer 없이 직접 Application Layer에 FastAPI를 붙이면**:

1. **의존성 역전 위반**: Application Layer가 FastAPI에 의존하게 됨
2. **테스트 어려움**: FastAPI 없이는 테스트 불가
3. **교체 어려움**: FastAPI를 다른 프레임워크로 교체하려면 Application Layer 수정 필요

**Interface Layer를 두면**:

- **의존성 방향 명확**: Interface → Application → Domain (안쪽으로만 의존)
- **테스트 용이**: Application Layer는 FastAPI 없이도 테스트 가능
- **프레임워크 교체 용이**: FastAPI를 다른 프레임워크로 교체해도 Application Layer는 변경 불필요

---

## REST API 어댑터 설계

### FastAPI 앱 팩토리 패턴

FastAPI 앱을 생성하는 함수를 분리하여 테스트 가능하게 만들었습니다:

```python
# interface/api/app.py

def create_app(allowed_origins: Iterable[str] | None = None) -> FastAPI:
    """FastAPI 애플리케이션을 생성합니다."""
    app = FastAPI(title="SentinelPipeline API", version="0.1.0")
    
    # CORS 설정
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # 예외 핸들러 등록
    @app.exception_handler(SentinelError)
    async def handle_sentinel_error(_: Request, exc: SentinelError) -> JSONResponse:
        """SentinelError → JSON 응답 매핑."""
        return JSONResponse(
            status_code=exc.http_status,
            content={"success": False, "error": exc.to_dict()},
        )
    
    # 라우터 등록
    app.include_router(health.router)
    app.include_router(streams.router)
    app.include_router(config.router)
    
    return app
```

**팩토리 패턴의 장점**:

- 테스트에서 다른 설정으로 앱 생성 가능
- 의존성 주입이 명확함
- 앱 생성 로직을 한 곳에 집중

### 예외 핸들러 설계

3가지 예외 핸들러를 등록했습니다:

1. **SentinelError 핸들러**: 도메인 예외 → HTTP 상태 코드 매핑
2. **ValidationError 핸들러**: Pydantic 검증 오류 → 422 응답
3. **Exception 핸들러**: 알 수 없는 예외 → 500 응답 (최소 정보만 노출)

```python
@app.exception_handler(SentinelError)
async def handle_sentinel_error(_: Request, exc: SentinelError) -> JSONResponse:
    """SentinelError → JSON 응답 매핑."""
    logger.error(
        "SentinelError 발생",
        code=exc.code.value,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(
        status_code=exc.http_status,  # ErrorCode → HTTP 상태 코드
        content={"success": False, "error": exc.to_dict()},
    )
```

**응답 형식**:

```json
{
  "success": false,
  "error": {
    "code": "STREAM_NOT_FOUND",
    "message": "스트림을 찾을 수 없습니다: camera_01",
    "details": {
      "stream_id": "camera_01"
    }
  }
}
```

---

## 의존성 주입 시스템

### 왜 DI가 필요한가?

FastAPI의 `Depends`는 함수 호출 시점에 의존성을 주입합니다. 하지만 실제 인스턴스(StreamManager, PipelineEngine 등)는 애플리케이션 시작 시 생성되어야 합니다.

**문제**: 라우트에서 `Depends(get_stream_manager)`를 호출할 때, 실제 StreamManager 인스턴스가 어디에 있는가?

**해결**: 전역 변수에 인스턴스를 저장하고, `set_app_context()`로 주입:

```python
# interface/api/dependencies.py

_stream_manager: Optional[StreamManager] = None
_pipeline_engine: Optional[PipelineEngine] = None
_config_manager: Optional[ConfigManager] = None
_config_loader: Optional[ConfigLoader] = None
_event_emitter: Optional[EventEmitter] = None

def set_app_context(
    stream_manager: StreamManager,
    pipeline_engine: PipelineEngine,
    config_manager: ConfigManager,
    config_loader: ConfigLoader,
    event_emitter: EventEmitter,
) -> None:
    """애플리케이션 의존성을 주입합니다."""
    global _stream_manager, _pipeline_engine, _config_manager, _config_loader, _event_emitter
    _stream_manager = stream_manager
    _pipeline_engine = pipeline_engine
    _config_manager = config_manager
    _config_loader = config_loader
    _event_emitter = event_emitter

@lru_cache(maxsize=1)
def get_stream_manager() -> StreamManager:
    if _stream_manager is None:
        raise RuntimeError("StreamManager가 설정되지 않았습니다")
    return _stream_manager
```

**사용 예시** (라우트에서):

```python
@router.get("/streams")
async def list_streams(
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamListResponse:
    """스트림 목록 조회."""
    streams = manager.get_all_streams()
    return StreamListResponse(data=[s.to_summary() for s in streams])
```

**장점**:

- 테스트에서 mock 객체 주입 가능
- 실제 인스턴스는 Composition Root(main.py)에서 생성
- 라우트는 인터페이스에만 의존

### 인증 시스템

API Key와 Basic Auth를 선택적으로 적용할 수 있도록 설계했습니다:

```python
async def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """
    API Key 검증 (선택적).
    환경변수 API_KEY가 설정된 경우에만 검증합니다.
    """
    expected = os.getenv("API_KEY")
    if expected:
        if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )

async def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    """
    Admin 인증 (Basic Auth).
    환경변수 ADMIN_USER / ADMIN_PASSWORD 기준으로 검증합니다.
    """
    admin_password = os.getenv("ADMIN_PASSWORD")
    if admin_password is None:
        # 설정되지 않았다면 인증을 요구하지 않음
        return
    
    # secrets.compare_digest로 타이밍 공격 방지
    user_ok = secrets.compare_digest(credentials.username, admin_user)
    pass_ok = secrets.compare_digest(credentials.password, admin_password)
    if not (user_ok and pass_ok):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
```

**사용 예시**:

```python
@router.put("/config", dependencies=[Depends(verify_admin)])
async def update_config(...):
    """설정 변경은 Admin 인증 필요."""
    ...

@router.get("/streams", dependencies=[Depends(verify_api_key)])
async def list_streams(...):
    """스트림 조회는 API Key 필요 (환경변수 설정 시)."""
    ...
```

---

## API 엔드포인트 설계

### 헬스 체크 API

Kubernetes, 로드밸런서 등에서 사용할 수 있도록 liveness/readiness를 분리했습니다:

```python
# routes/health.py

@router.get("/health/live")
async def health_live() -> dict[str, str]:
    """프로세스 생존 확인 (liveness probe)."""
    return {"status": "alive"}

@router.get("/health/ready")
async def health_ready(
    manager: StreamManager = Depends(get_stream_manager),
) -> dict[str, Any]:
    """서비스 준비 상태 (readiness probe)."""
    # 주요 컴포넌트 초기화 여부 확인
    return {"status": "ready"}

@router.get("/health")
async def health(
    manager: StreamManager = Depends(get_stream_manager),
    pipeline: PipelineEngine = Depends(get_pipeline_engine),
) -> dict[str, Any]:
    """전체 상태 조회."""
    streams = manager.get_all_streams()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "streams": {
            s.stream_id: {
                "status": s.status.value,
                "fps": round(s.stats.fps, 1),
            }
            for s in streams
        },
        "modules": {
            # PipelineEngine에서 모듈 상태 조회
        },
    }
```

### 스트림 제어 API

스트림의 생명주기를 제어하는 API입니다:

```python
# routes/streams.py

@router.get("/streams", response_model=StreamListResponse)
async def list_streams(
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamListResponse:
    """전체 스트림 목록 조회."""
    streams = manager.get_all_streams()
    return StreamListResponse(
        data=[s.to_summary() for s in streams]
    )

@router.get("/streams/{stream_id}", response_model=StreamDetailResponse)
async def get_stream(
    stream_id: str,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamDetailResponse:
    """특정 스트림 상세 조회."""
    state = manager.get_stream_state(stream_id)
    if not state:
        raise StreamError(
            ErrorCode.STREAM_NOT_FOUND,
            f"스트림을 찾을 수 없습니다: {stream_id}",
            stream_id=stream_id,
        )
    return StreamDetailResponse(
        data={
            "stream": state.to_dict(),
            "stats": state.stats.to_dict(),
        }
    )

@router.post("/streams/{stream_id}/start", response_model=StreamControlResponse)
async def start_stream(
    stream_id: str,
    request: StreamStartRequest,
    manager: StreamManager = Depends(get_stream_manager),
) -> StreamControlResponse:
    """스트림 시작."""
    try:
        state = manager.start_stream(
            stream_id=stream_id,
            rtsp_url=request.rtsp_url,
            max_fps=request.max_fps,
            downscale=request.downscale,
            output_url=request.output_url,
        )
        return StreamControlResponse(
            data={"status": "started", "stream_id": stream_id}
        )
    except StreamError as e:
        # 이미 실행 중인 경우
        if e.code == ErrorCode.STREAM_ALREADY_RUNNING:
            raise
        raise
```

**요청/응답 DTO**:

```python
class StreamStartRequest(BaseModel):
    rtsp_url: str
    max_fps: int = 15
    downscale: float = 1.0
    output_url: str | None = None

class StreamControlResponse(BaseModel):
    data: dict[str, Any]  # {"status": "started", "stream_id": "..."}
```

### 설정 관리 API

런타임에 설정을 조회하고 변경할 수 있는 API입니다:

```python
# routes/config.py

@router.get("/config", response_model=ConfigResponse)
async def get_config(
    manager: ConfigManager = Depends(get_config_manager),
) -> ConfigResponse:
    """현재 설정 조회."""
    return ConfigResponse(data=manager.to_dict())

@router.put("/config", response_model=ConfigResponse, dependencies=[Depends(verify_admin)])
async def update_config(
    payload: UpdateConfigRequest,
    manager: ConfigManager = Depends(get_config_manager),
    loader: ConfigLoader = Depends(get_config_loader),
) -> ConfigResponse:
    """전체 설정 교체."""
    # 1. Pydantic 스키마 검증
    config = loader.merge_with_defaults(payload)
    
    # 2. 교차 검증 (event.batch_size <= max_queue_size 등)
    ok, errors = loader.validate(config)
    if not ok:
        raise ConfigError(
            ErrorCode.CONFIG_INVALID,
            "설정 검증에 실패했습니다",
            details={"errors": errors},
        )
    
    # 3. Runtime 객체로 변환
    runtime_config = loader.to_runtime(config)
    bundle = loader.to_runtime_bundle(config)  # pipeline/event/observability 포함
    
    # 4. ConfigManager에 적용
    manager.update_config(runtime_config)
    
    # 5. 응답에 전체 설정 번들 포함
    return ConfigResponse(
        data={
            "app": manager.to_dict(),  # modules/global
            "bundle": bundle,  # pipeline/event/observability/streams
        }
    )
```

**설정 번들 응답의 의미**:

현재 ConfigManager는 `modules`와 `global` 설정만 보유합니다. 하지만 `pipeline`, `event`, `observability`, `streams` 설정도 존재하므로, 응답에 `bundle`을 함께 내려 클라이언트가 전체 설정을 확인할 수 있게 했습니다.

실제 런타임 적용은 8단계(엔트리포인트)에서 `to_runtime_bundle()`을 사용해 PipelineEngine/EventEmitter/StreamManager 초기화에 연결할 예정입니다.

---

## ConfigLoader 확장

### to_runtime_bundle() 메서드

기존 `to_runtime()`은 `modules`와 `global`만 반환했습니다. 하지만 PipelineEngine, EventEmitter, StreamManager 초기화에는 `pipeline`, `event`, `observability`, `streams` 설정도 필요합니다.

```python
# interface/config/loader.py

def to_runtime_bundle(self, config: AppConfig) -> dict[str, Any]:
    """
    전체 설정을 런타임 번들로 변환합니다.
    
    Application Layer 컴포넌트 초기화에 필요한 모든 설정을 포함합니다.
    """
    app = self.to_runtime(config)  # modules/global
    
    return {
        "app": app,  # ConfigManager에 전달
        "pipeline": {
            "max_consecutive_errors": config.pipeline.max_consecutive_errors,
            "max_consecutive_timeouts": config.pipeline.max_consecutive_timeouts,
            "max_workers": config.pipeline.max_workers,
        },
        "event": {
            "max_queue_size": config.event.max_queue_size,
            "batch_size": config.event.batch_size,
            "flush_interval_ms": config.event.flush_interval_ms,
            "drop_strategy": config.event.drop_strategy,
        },
        "observability": {
            "metrics_enabled": config.observability.metrics_enabled,
            "metrics_port": config.observability.metrics_port,
        },
        "streams": [
            {
                "stream_id": s.stream_id,
                "rtsp_url": s.rtsp_url,
                "enabled": s.enabled,
                # ... 기타 설정
            }
            for s in config.streams
        ],
    }
```

**사용 예시** (8단계에서):

```python
# main.py

bundle = loader.to_runtime_bundle(config)

# ConfigManager에 app 설정 적용
config_manager.load_config(bundle["app"])

# PipelineEngine 초기화 시 pipeline 설정 적용
pipeline_engine = PipelineEngine(
    max_consecutive_errors=bundle["pipeline"]["max_consecutive_errors"],
    max_consecutive_timeouts=bundle["pipeline"]["max_consecutive_timeouts"],
)

# EventEmitter 초기화 시 event 설정 적용
event_emitter = EventEmitter(
    max_queue_size=bundle["event"]["max_queue_size"],
    batch_size=bundle["event"]["batch_size"],
    flush_interval_ms=bundle["event"]["flush_interval_ms"],
    drop_strategy=bundle["event"]["drop_strategy"],
)

# StreamManager에 global 설정 적용
stream_manager.apply_global_config(
    max_fps=bundle["app"]["global_config"]["max_fps"],
    downscale=bundle["app"]["global_config"]["downscale"],
)
```

### 교차 검증 추가

Pydantic 스키마 검증만으로는 부족한 경우가 있습니다. 예를 들어, `event.batch_size`는 `event.max_queue_size`보다 작거나 같아야 합니다.

```python
def validate(self, config: AppConfig) -> tuple[bool, list[str]]:
    """
    설정을 검증합니다.
    
    Returns:
        (성공 여부, 에러 목록)
    """
    errors: list[str] = []
    
    # Pydantic 검증은 이미 완료됨 (load_from_dict에서)
    # 여기서는 교차 필드 검증만 수행
    
    # 이벤트 큐/배치 크기 검증
    if config.event.batch_size > config.event.max_queue_size:
        errors.append(
            f"event.batch_size({config.event.batch_size})는 "
            f"event.max_queue_size({config.event.max_queue_size})보다 작거나 같아야 합니다"
        )
    
    # 향후 확장: transport URL 필수, downscale/max_fps 범위,
    # stream/module 중복, reconnect 값 합리성 등
    
    return (len(errors) == 0, errors)
```

**향후 확장 예정**:

- `transport.url` 필수 검증
- `downscale` 범위 검증 (0.0 < downscale <= 1.0)
- `max_fps` 범위 검증 (1 <= max_fps <= 60)
- `stream_id` 중복 검증
- `module.name` 중복 검증
- `reconnect` 값 합리성 (base_delay < max_delay 등)

---

## 설계 결정 과정

### 1. 전역 변수 vs 의존성 컨테이너

**선택**: 전역 변수 + `set_app_context()`

**이유**:

- FastAPI의 `Depends`는 함수 기반 DI이므로 전역 변수가 자연스러움
- 의존성 컨테이너(예: `dependency-injector`)는 오버엔지니어링
- 테스트에서 mock 주입이 간단함

**단점**: 전역 상태 (하지만 애플리케이션 수명 동안 단일 인스턴스이므로 문제 없음)

### 2. DTO vs Domain 모델 직접 노출

**선택**: DTO (Pydantic 모델)

**이유**:

- Domain 모델은 외부에 노출하지 않음 (캡슐화)
- API 버전 변경 시 Domain 모델 수정 불필요
- 필요한 필드만 선택적으로 노출 가능

**예시**:

```python
# Domain 모델 (StreamState)에는 많은 필드가 있음
# 하지만 API 응답에는 요약 정보만 필요

class StreamListResponse(BaseModel):
    data: list[dict[str, Any]]  # StreamState.to_summary() 결과

# StreamState.to_summary()는 필요한 필드만 반환
def to_summary(self) -> dict[str, Any]:
    return {
        "stream_id": self.stream_id,
        "status": self.status.value,
        "fps": round(self.stats.fps, 1),
        # ... 요약 정보만
    }
```

### 3. 설정 번들 응답 vs ConfigManager 확장

**선택**: 설정 번들 응답 (현재 단계)

**이유**:

- ConfigManager는 `modules`와 `global`만 보유하는 것이 단순함
- 전체 설정은 `to_runtime_bundle()`로 한 번에 제공 가능
- 8단계에서 실제 적용 경로를 만들 때 정리 예정

**향후 개선**:

- ConfigManager에 `pipeline`/`event`/`observability` 상태 보관
- 또는 GET /api/config에서 `loader.to_runtime_bundle()` 기준으로 전체 설정 반환

---

## 현재 남은 빈칸

### 1. 런타임 적용 경로 없음

`to_runtime_bundle()`로 설정을 반환하지만, 실제로 PipelineEngine/EventEmitter/StreamManager에 적용하는 경로가 없습니다.

**해결**: 8단계(엔트리포인트)에서 배선 예정

```python
# main.py (8단계에서 구현)

bundle = loader.to_runtime_bundle(config)

# 실제 컴포넌트 초기화에 연결
pipeline_engine = PipelineEngine(
    max_consecutive_errors=bundle["pipeline"]["max_consecutive_errors"],
    ...
)
```

### 2. ConfigManager/GET /api/config 부분적 응답

현재 GET /api/config는 `modules`와 `global`만 반환합니다. 전체 설정을 보려면 PUT /api/config 응답의 `bundle`을 확인해야 합니다.

**해결 방안**:

- ConfigManager 확장: `pipeline`/`event`/`observability` 상태 보관
- 또는 GET /api/config에서 `loader.to_runtime_bundle()` 기준으로 전체 설정 반환

### 3. 교차 검증 보강 필요

현재는 `event.batch_size <= max_queue_size`만 검증합니다.

**향후 추가 예정**:

- `transport.url` 필수 검증
- `downscale` 범위 검증 (0.0 < downscale <= 1.0)
- `max_fps` 범위 검증 (1 <= max_fps <= 60)
- `stream_id` 중복 검증
- `module.name` 중복 검증
- `reconnect` 값 합리성 검증

---

## 구현 결과

### 파일 구조

```
src/sentinel_pipeline/interface/
├── api/
│   ├── __init__.py
│   ├── app.py              # FastAPI 앱 팩토리 (89줄)
│   ├── dependencies.py     # DI 헬퍼 (119줄)
│   └── routes/
│       ├── __init__.py
│       ├── health.py       # 헬스 체크 (59줄)
│       ├── streams.py      # 스트림 제어 (153줄)
│       └── config.py       # 설정 관리 (175줄)
└── config/
    └── loader.py           # to_runtime_bundle 추가 (160줄)
```

### 사용 예시

```python
# 애플리케이션 시작 (8단계에서 구현 예정)

from sentinel_pipeline.interface.api.app import create_app
from sentinel_pipeline.interface.api.dependencies import set_app_context

# 컴포넌트 초기화
stream_manager = StreamManager()
pipeline_engine = PipelineEngine()
config_manager = ConfigManager()
config_loader = ConfigLoader()
event_emitter = EventEmitter()

# DI 주입
set_app_context(
    stream_manager=stream_manager,
    pipeline_engine=pipeline_engine,
    config_manager=config_manager,
    config_loader=config_loader,
    event_emitter=event_emitter,
)

# FastAPI 앱 생성
app = create_app(allowed_origins=["http://localhost:3000"])

# uvicorn 실행
uvicorn.run(app, host="0.0.0.0", port=9000)
```

---


## 마무리

Interface Layer의 REST API 어댑터를 통해 외부 세계와의 경계를 명확히 했습니다. Clean Architecture 원칙을 지키면서도 실용적인 API를 제공할 수 있게 되었습니다.

다음 단계에서는 엔트리포인트에서 설정 번들을 실제 컴포넌트에 적용하여 전체 시스템이 동작하도록 만들 예정입니다.

---

## 참고 자료

- [헥사고날 아키텍처](https://medium.com/mo-zza/spring-boot-%EA%B8%B0%EB%B0%98-%ED%97%A5%EC%82%AC%EA%B3%A0%EB%82%A0-%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98-hexagonal-architecture-with-spring-boot-4daf81752756)
- [프로젝트 아키텍처 문서](../architecture.md)
