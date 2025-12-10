# Application Layer

## 들어가며

Application Layer는 Domain Layer만 참조하며, 실제 비즈니스 로직의 흐름을 조율합니다. 파이프라인 엔진, 스트림 관리, 이벤트 발행, 설정 관리 등 핵심 유스케이스를 구현합니다.

---

## Application Layer의 역할

### Domain Layer와의 관계

Application Layer는 Domain Layer의 모델과 인터페이스를 사용하여 비즈니스 로직을 구현합니다. 하지만 Domain Layer는 Application Layer를 전혀 알지 못합니다.

```
┌─────────────────────────────────────┐
│      Application Layer              │
│  (PipelineEngine, StreamManager)    │
│           ↓ 사용                    │
│      Domain Layer                   │
│  (ModuleBase, Event, Stream)        │
└─────────────────────────────────────┘

```

### Application Layer의 구성 요소

```
src/sentinel_pipeline/application/
├── pipeline/
│   ├── pipeline.py        # PipelineEngine
│   └── scheduler.py       # ModuleScheduler
├── stream/
│   ├── manager.py         # StreamManager
│   └── health.py          # HealthWatcher
├── config/
│   └── manager.py         # ConfigManager
└── event/
    └── emitter.py         # EventEmitter

```

---

## PipelineEngine: 파이프라인 엔진

### 설계 목표

여러 모듈을 순차적으로 실행하여 프레임을 처리하고, 각 모듈의 결과를 다음 모듈로 전달해야 합니다. 또한 개별 모듈의 오류가 전체 파이프라인에 영향을 주지 않도록 예외 격리가 필요합니다.

### 핵심 로직

```python
def process_frame(self, frame, metadata):
    all_events = []
    current_frame = frame
    current_metadata = metadata.copy()

    # priority 순으로 모듈 실행
    for context in self.scheduler.get_execution_order():
        result = self.scheduler.execute_with_timeout(
            context, current_frame, current_metadata
        )

        if result is None:
            # 타임아웃/에러 - 현재 프레임 유지, 다음 모듈 실행
            continue

        current_frame, events, current_metadata = result
        all_events.extend(events)

    return current_frame, all_events

```

**핵심 포인트**:

1. **예외 격리**: 모듈 실패 시 `result is None`으로 처리하고 다음 모듈 계속 실행
2. **메타데이터 전달**: 각 모듈이 메타데이터를 수정하여 다음 모듈에 전달
3. **이벤트 수집**: 모든 모듈에서 생성된 이벤트를 수집

### 모듈 실행 순서

`ModuleScheduler`가 모듈을 `priority` 오름차순으로 정렬합니다:

```python
def get_execution_order(self) -> list[ModuleContext]:
    active_modules = [
        ctx for ctx in self._modules.values()
        if ctx.module.enabled
    ]
    active_modules.sort(key=lambda ctx: ctx.module.priority)
    return active_modules

```

**예시**:

- FireDetectModule (priority=100) → 먼저 실행
- FaceBlurModule (priority=200) → 나중에 실행

---

## ModuleScheduler: 모듈 스케줄러

### 타임아웃 관리

각 모듈은 `timeout_ms` 제한 시간 내에 완료되어야 합니다. 이를 위해 `ThreadPoolExecutor`를 사용합니다:

```python
def execute_with_timeout(self, context, frame, metadata):
    module = context.module
    timeout_sec = module.timeout_ms / 1000.0

    try:
        future = self._executor.submit(
            module.process_frame, frame, metadata
        )
        result = future.result(timeout=timeout_sec)
        context.record_success(latency_ms, event_count)
        return result
    except FuturesTimeoutError:
        context.record_timeout()
        return None
    except Exception as e:
        context.record_error()
        return None

```

**장점**:

- 각 모듈이 독립적인 스레드에서 실행
- 타임아웃이 명확하게 적용됨
- 메인 스레드가 블로킹되지 않음

### 자동 비활성화 정책

연속으로 에러나 타임아웃이 발생하면 모듈을 자동으로 비활성화합니다:

```python
def _check_disable(self, context, reason):
    if context.should_disable(
        max_errors=5,
        max_timeouts=10,
    ):
        context.module.enabled = False
        self._disabled_until[module_name] = time.time() + 300  # 5분 쿨다운

```

**쿨다운 후 재활성화**:

```python
def get_execution_order(self):
    now = time.time()

    # 쿨다운이 끝난 모듈 재활성화
    for name, until_ts in list(self._disabled_until.items()):
        if now >= until_ts:
            ctx = self._modules.get(name)
            if ctx:
                ctx.module.enabled = True
                ctx.reset_counters()
                del self._disabled_until[name]

```

이렇게 하면 일시적인 문제로 인한 모듈 비활성화가 자동으로 복구됩니다.

---

## StreamManager: 스트림 관리자

### 스트림 생명주기

각 스트림은 별도의 스레드에서 실행되며, 다음과 같은 생명주기를 가집니다:

```
IDLE → STARTING → RUNNING → STOPPING → STOPPED
                ↓
            RECONNECTING (에러 시)

```

### 스트림 루프 구현

```python
def _stream_loop(self, ctx: StreamContext):
    # 1. 디코더 연결
    ctx.decoder = self._decoder_factory()
    if not ctx.decoder.connect(config.rtsp_url):
        raise StreamError(...)

    ctx.state.set_status(StreamStatus.RUNNING)

    # 2. 프레임 처리 루프
    frame_interval = 1.0 / config.max_fps

    while not ctx.should_stop():
        # FPS 제한
        elapsed = time.time() - last_frame_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
            continue

        # 프레임 읽기
        success, frame = ctx.decoder.read_frame()
        if not success:
            if self._should_reconnect(ctx):
                self._attempt_reconnect(ctx)
            continue

        # 프레임 처리
        if self._frame_callback:
            processed_frame, events = self._frame_callback(frame, metadata)
            ctx.state.stats.record_frame(latency_ms)

            # 퍼블리셔로 출력
            if ctx.publisher:
                ctx.publisher.write_frame(processed_frame)

```

### 지수 백오프 재연결

스트림이 끊기면 지수 백오프 방식으로 재연결을 시도합니다:

```python
def _attempt_reconnect(self, ctx):
    ctx.state.set_status(StreamStatus.RECONNECTING)
    ctx.state.record_retry()

    # 지수 백오프 계산
    delay = ctx.state.calculate_next_retry_delay()
    # 예: 1초 → 2초 → 4초 → 8초 (최대)

    time.sleep(delay)

    # 재연결 시도
    if ctx.decoder.connect(ctx.state.config.rtsp_url):
        ctx.state.set_status(StreamStatus.RUNNING)
        ctx.state.reset_retry()

```

**지수 백오프 공식**:

```python
delay = base_delay * (2 ** retry_count)
delay = min(delay, max_delay)

```

### Graceful 종료

스트림을 중지할 때는 graceful하게 종료합니다:

```python
def stop_stream(self, stream_id, force=False):
    ctx.request_stop()  # 중지 요청

    # 스레드 종료 대기
    if ctx.thread:
        ctx.thread.join(timeout=5.0)

        if ctx.thread.is_alive() and force:
            # 강제 종료 (데몬 스레드이므로 프로세스 종료 시 함께 종료)
            pass

    # 리소스 정리
    if ctx.decoder:
        ctx.decoder.release()
    if ctx.publisher:
        ctx.publisher.stop()

```

---

## HealthWatcher: 헬스 워처

### 모니터링 항목

스트림의 상태를 주기적으로 확인합니다:

```python
def _check_streams(self):
    for state in self._stream_manager.get_all_streams():
        issues = []

        # 프레임 타임아웃 확인
        if state.stats.last_frame_ts:
            elapsed = time.time() - state.stats.last_frame_ts
            if elapsed > self.frame_timeout_seconds:
                issues.append(f"프레임 타임아웃 ({elapsed:.1f}초)")

        # FPS 확인
        if state.stats.fps < self.min_fps_threshold:
            issues.append(f"낮은 FPS ({state.stats.fps:.1f})")

        if issues:
            self._on_unhealthy(stream_id, ", ".join(issues))
        else:
            self._on_recovered(stream_id)

```

**모니터링 주기**: 기본 5초마다 체크

### 비정상 상태 감지

문제가 감지되면 콜백을 호출하여 적절한 조치를 취할 수 있습니다:

```python
watcher.set_on_unhealthy(lambda stream_id, reason:
    logger.warning(f"스트림 비정상: {stream_id} - {reason}")
)

```

---

## ConfigManager: 설정 관리자

### 런타임 설정 변경

코드를 재시작하지 않고도 설정을 변경할 수 있습니다:

```python
def update_module(self, name, enabled=None, priority=None, **kwargs):
    module_config = self.get_module_config(name)

    if enabled is not None:
        module_config.enabled = enabled
    if priority is not None:
        module_config.priority = priority

    # 콜백 호출하여 다른 컴포넌트에 전파
    self._notify_module_change(name, module_config)

```

### 설정 변경 전파

설정이 변경되면 관련 컴포넌트에 자동으로 전파됩니다:

```python
# ConfigManager에 콜백 등록
config_manager.add_on_module_change(
    lambda name, config: pipeline_engine.reload_modules([config])
)

config_manager.add_on_global_change(
    lambda config: stream_manager.apply_global_config(
        max_fps=config.max_fps,
        downscale=config.downscale,
    )
)

```

이렇게 하면 설정 변경이 즉시 반영됩니다.

---

## EventEmitter: 이벤트 발행자

### 큐 기반 이벤트 처리

이벤트를 큐에 넣고, 별도 스레드에서 배치로 전송합니다:

```python
def emit(self, event: Event) -> bool:
    try:
        self._queue.put_nowait(event)
        return True
    except queue.Full:
        # 큐가 가득 찬 경우 드롭 전략 적용
        if self.drop_strategy == DropStrategy.DROP_OLDEST:
            dropped = self._queue.get_nowait()
            self._queue.put_nowait(event)
        else:
            # 새 이벤트 거부
            return False

```

### 배치 전송

주기적으로 배치를 모아서 전송합니다:

```python
def _emit_loop(self):
    while self._running:
        batch = self._collect_batch()  # 최대 batch_size만큼 수집

        if batch and self._sync_transport:
            self._sync_transport(batch)  # 일괄 전송

        time.sleep(self.flush_interval_ms / 1000.0)

```

**배치 전송의 장점**:

- 네트워크 오버헤드 감소
- VMS 서버 부하 감소
- 전송 실패 시 재시도 용이

### 백프레셔 관리

큐가 가득 찬 경우 두 가지 전략을 제공합니다:

1. **drop_oldest**: 가장 오래된 이벤트를 폐기하고 새 이벤트 추가
    - 실시간성이 중요한 경우
2. **drop_newest**: 새 이벤트를 거부
    - 데이터 손실을 최소화해야 하는 경우

---

## 동시성 처리

### 스레드 안전성

모든 공유 상태는 `threading.Lock`으로 보호합니다:

```python
class StreamManager:
    def __init__(self):
        self._streams: dict[str, StreamContext] = {}
        self._lock = threading.RLock()  # 재진입 가능한 락

    def start_stream(self, stream_id, rtsp_url):
        with self._lock:  # 락으로 보호
            if stream_id in self._streams:
                raise StreamError(...)
            # ...

```

**RLock 사용 이유**: 같은 스레드에서 중첩 호출이 발생할 수 있기 때문

### 데몬 스레드

스트림 처리 스레드는 데몬 스레드로 설정합니다:

```python
ctx.thread = threading.Thread(
    target=self._stream_loop,
    args=(ctx,),
    name=f"stream_{stream_id}",
    daemon=True,  # 데몬 스레드
)

```

**데몬 스레드의 장점**: 메인 프로세스가 종료되면 자동으로 종료됨

---

## 설계 결정 과정

### 1. ThreadPoolExecutor vs 직접 스레드 생성

**선택**: ThreadPoolExecutor 사용 (ModuleScheduler)

**이유**:

- 타임아웃을 `future.result(timeout=...)`으로 쉽게 적용 가능
- 스레드 풀 관리 자동화
- 예외 처리 용이

**하지만**: StreamManager는 직접 스레드 생성 (장시간 실행, 상태 관리 필요)

### 2. 동기 vs 비동기

**선택**: 동기 방식 (현재 단계)

**이유**:

- Application Layer는 Domain만 참조해야 함
- 비동기 라이브러리는 Infrastructure Layer에서 사용
- 단순성과 명확성

**향후**: Infrastructure Layer에서 비동기 HTTP/WebSocket 클라이언트 사용 예정

### 3. 콜백 vs 이벤트 버스

**선택**: 콜백 방식

**이유**:

- 단순하고 명확함
- 타입 안전성 (타입 힌트로 콜백 시그니처 명시)
- 디버깅 용이

**단점**: 콜백 체인이 복잡해질 수 있지만, 현재는 관리 가능한 수준

### 4. 큐 크기 제한

**선택**: 제한 있음 (기본 1000)

**이유**:

- 무한 큐는 메모리 누수 위험
- 백프레셔를 명시적으로 관리
- 운영자가 시스템 한계를 인지 가능

---

## 구현 결과

### 핵심 기능

| 컴포넌트 | 주요 기능 |
| --- | --- |
| PipelineEngine | 모듈 순차 실행, 예외 격리, 이벤트 수집 |
| ModuleScheduler | priority 정렬, 타임아웃 관리, 자동 비활성화 |
| StreamManager | 스트림 생명주기, 재연결, graceful 종료 |
| HealthWatcher | 상태 모니터링, 이상 감지 |
| ConfigManager | 런타임 설정 변경, 변경 전파 |
| EventEmitter | 이벤트 큐잉, 배치 전송, 백프레셔 관리 |

---

## 테스트 전략

### 단위 테스트

각 컴포넌트를 독립적으로 테스트합니다:

```python
def test_pipeline_module_isolation():
    """모듈 예외 격리 테스트"""
    engine = PipelineEngine()

    # 실패하는 모듈 등록
    engine.register_module(FailingModule())
    engine.register_module(SuccessModule())

    frame, events = engine.process_frame(frame, metadata)

    # 실패 모듈은 건너뛰고 성공 모듈은 실행됨
    assert len(events) > 0

def test_scheduler_timeout():
    """타임아웃 처리 테스트"""
    scheduler = ModuleScheduler()
    slow_module = SlowModule(timeout_ms=10)  # 10ms 타임아웃
    scheduler.register(slow_module)

    result = scheduler.execute_with_timeout(
        context, frame, metadata
    )

    assert result is None  # 타임아웃으로 None 반환
    assert context.timeout_count == 1

```

### 통합 테스트

여러 컴포넌트를 함께 테스트합니다:

```python
def test_stream_lifecycle():
    """스트림 생명주기 통합 테스트"""
    manager = StreamManager()
    manager.set_decoder_factory(lambda: MockDecoder())
    manager.set_frame_callback(mock_callback)

    # 시작
    state = manager.start_stream("cam_01", "rtsp://...")
    assert state.status == StreamStatus.STARTING

    # 실행 대기
    time.sleep(0.1)
    state = manager.get_stream_state("cam_01")
    assert state.status == StreamStatus.RUNNING

    # 중지
    manager.stop_stream("cam_01")
    state = manager.get_stream_state("cam_01")
    assert state.status == StreamStatus.STOPPED

```

---

## 성능 고려사항

### FPS 제한

스트림 처리 시 FPS를 제한하여 CPU 사용량을 관리합니다:

```python
frame_interval = 1.0 / config.max_fps
elapsed = time.time() - last_frame_time

if elapsed < frame_interval:
    time.sleep(frame_interval - elapsed)

```

### 메모리 관리

큐 크기를 제한하여 메모리 사용량을 제어합니다:

```python
self._queue: queue.Queue[Event] = queue.Queue(maxsize=max_queue_size)

```

큐가 가득 차면 드롭 전략에 따라 이벤트를 폐기합니다.

### 스레드 풀 크기

ModuleScheduler의 스레드 풀 크기를 제한합니다:

```python
self._executor = ThreadPoolExecutor(max_workers=4)

```

너무 많은 스레드는 컨텍스트 스위칭 오버헤드를 증가시킵니다.

---

## 마무리

Application Layer는 Domain Layer의 모델을 사용하여 실제 비즈니스 로직을 구현합니다. 핵심은:

1. **예외 격리**: 개별 모듈의 오류가 전체에 영향 없음
2. **타임아웃 관리**: 실시간 처리를 위한 시간 제한
3. **자동 복구**: 문제 발생 시 자동으로 재시도 및 복구
4. **설정 전파**: 런타임 설정 변경이 즉시 반영

다음 단계인 Infrastructure Layer에서는 실제 RTSP 디코딩, FFmpeg 퍼블리싱, AI 모델 추론 등을 구현합니다.

---

## 참고 자료

- [Python ThreadPoolExecutor 문서](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
- [threading 모듈 문서](https://docs.python.org/3/library/threading.html)
- [프로젝트 Application Layer 코드](https://github.com/dusen0528/SentinelPipeline/tree/dev/src/sentinel_pipeline/application)