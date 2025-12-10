# Infrastructure Layer

## 들어가며

Infrastructure Layer는 외부 시스템과의 통신을 담당합니다. RTSP 스트림 디코딩, FFmpeg 퍼블리싱, AI 모델 추론, 이벤트 전송 등 실제 하드웨어 및 외부 서비스와의 인터페이스를 구현합니다.

---

## Infrastructure Layer의 역할

### Application Layer와의 관계

Infrastructure Layer는 Application Layer가 요구하는 인터페이스를 구현합니다. 예를 들어, `StreamManager`는 `RTSPDecoder`를 사용하지만, `RTSPDecoder`의 구체적인 구현은 Infrastructure Layer에 있습니다.

```
┌─────────────────────────────────────┐
│      Application Layer              │
│  (StreamManager, EventEmitter)      │
│           ↓ 사용                    │
│      Infrastructure Layer           │
│  (RTSPDecoder, FFmpegPublisher)     │
└─────────────────────────────────────┘

```

### Infrastructure Layer의 구성 요소

```
src/sentinel_pipeline/infrastructure/
├── video/
│   ├── rtsp_decoder.py          # RTSP 스트림 디코더
│   └── ffmpeg_publisher.py       # FFmpeg 스트림 퍼블리셔
├── inference/
│   └── runtime.py               # AI 추론 런타임 (ONNX/PyTorch)
└── transport/
    ├── http_client.py           # HTTP 이벤트 클라이언트
    └── ws_client.py             # WebSocket 이벤트 클라이언트

```

---

## RTSPDecoder: RTSP 스트림 디코더

### 설계 목표

OpenCV의 `VideoCapture`를 사용하여 RTSP 스트림을 읽고, 프레임을 numpy 배열로 반환해야 합니다. 또한 연결 끊김 시 자동 재연결이 필요합니다.

### 핵심 구현

```python
def connect(self, rtsp_url: str) -> bool:
    # OpenCV VideoCapture 생성
    self._cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # 버퍼 크기 설정
    self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

    # 타임아웃 설정 (OpenCV 빌드에 따라 미지원될 수 있음)
    try:
        self._cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self._connection_timeout_ms)
        self._cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self._read_timeout_ms)
    except AttributeError:
        self._logger.warning("OpenCV 타임아웃 설정 미지원, 기본값 사용")

    # 연결 확인
    if not self._cap.isOpened():
        raise StreamError(...)

```

**플랫폼 호환성**: OpenCV 빌드에 따라 타임아웃 속성이 없을 수 있으므로 `try/except`로 처리합니다.

### 지수 백오프 재연결

연결이 끊기면 지수 백오프 방식으로 재연결을 시도합니다:

```python
def reconnect(self, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 30.0) -> bool:
    for attempt in range(max_retries):
        # 지수 백오프 계산
        delay = min(base_delay * (2 ** attempt), max_delay)

        time.sleep(delay)

        if self.connect(self._rtsp_url):
            return True

    return False

```

**재연결 전략**:

- 1차 시도: 1초 대기
- 2차 시도: 2초 대기
- 3차 시도: 4초 대기
- 4차 시도: 8초 대기
- 5차 시도: 16초 대기 (최대 30초로 제한)

### 비밀번호 마스킹

로그에 RTSP URL을 기록할 때 비밀번호를 마스킹합니다:

```python
def _mask_password(self, url: str) -> str:
    import re
    return re.sub(r"://([^:]+):([^@]+)@", r"://\\1:****@", url)

```

**예시**:

```
rtsp://admin:password123@192.168.1.100:554/stream1
→ rtsp://admin:****@192.168.1.100:554/stream1

```

---

## FFmpegPublisher: FFmpeg 스트림 퍼블리셔

### 설계 목표

처리된 프레임을 FFmpeg subprocess를 통해 RTSP/RTMP 서버로 출력해야 합니다. 실시간 스트리밍이므로 지연을 최소화해야 합니다.

### FFmpeg 명령 구성

```python
def _build_ffmpeg_command(self) -> list[str]:
    cmd = [
        "ffmpeg",
        "-y",  # 덮어쓰기
        "-f", "rawvideo",  # 입력 형식: raw video
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",  # OpenCV BGR 형식
        "-s", f"{self._width}x{self._height}",
        "-r", str(self._fps),
        "-i", "-",  # stdin에서 입력
        "-c:v", self._codec,  # libx264
        "-pix_fmt", self._pix_fmt,  # yuv420p
        "-preset", self._preset,  # ultrafast
        "-b:v", self._bitrate,  # 2M
        "-f", "rtsp",  # 출력 형식
        "-rtsp_transport", "tcp",
        self._output_url,
    ]
    return cmd

```

**주요 옵션**:

- `preset=ultrafast`: 인코딩 속도 최우선 (지연 최소화)
- `rtsp_transport=tcp`: TCP 전송 (안정성)
- `GOP 크기`: FPS * 2 (2초)

### 파이프 데드락 방지

FFmpeg의 stdout/stderr를 소비하지 않으면 파이프가 가득 차서 데드락이 발생할 수 있습니다:

```python
# 나쁜 예시
self._process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,  # 소비하지 않으면 데드락
    stderr=subprocess.PIPE,
)

# 좋은 예시
self._process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,  # 로그 무시
    stderr=subprocess.DEVNULL,  # 로그 무시
)

```

**대안**: 로그가 필요한 경우 별도 스레드에서 소비해야 합니다.

### Graceful 종료

프로세스를 종료할 때는 stdin을 먼저 닫아 FFmpeg가 정상 종료하도록 합니다:

```python
def stop(self, timeout: float = 5.0) -> None:
    # stdin 닫기 (FFmpeg에 종료 신호)
    if self._process.stdin:
        self._process.stdin.close()

    # Graceful 종료 대기
    try:
        self._process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        self._process.kill()  # 강제 종료

```

---

## InferenceRuntime: AI 추론 런타임

### 설계 목표

ONNX Runtime과 PyTorch 런타임을 모두 지원해야 합니다. GPU가 사용 가능하면 자동으로 GPU를 사용하고, 통계 정보를 수집해야 합니다.

### 추상 클래스 정의

```python
class InferenceRuntime(ABC):
    @abstractmethod
    def load_model(self, model_path: str | Path) -> None:
        """모델을 로드합니다."""
        ...

    @abstractmethod
    def infer(self, inputs: dict[str, TensorType]) -> dict[str, TensorType]:
        """추론을 수행합니다."""
        ...

```

### ONNXRuntime 구현

```python
def load_model(self, model_path: str | Path) -> None:
    import onnxruntime as ort

    # Provider 설정 (GPU 우선)
    providers = []
    if self._use_gpu:
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        elif "TensorrtExecutionProvider" in available_providers:
            providers.append("TensorrtExecutionProvider")

    providers.append("CPUExecutionProvider")

    # 세션 생성
    self._session = ort.InferenceSession(
        str(model_path),
        sess_options,
        providers=providers,
    )

```

**Provider 우선순위**:

1. CUDA (NVIDIA GPU)
2. TensorRT (최적화된 추론)
3. CPU (폴백)

### PyTorchRuntime 구현

```python
def load_model(self, model_path: str | Path) -> None:
    import torch

    # 디바이스 설정
    if self._use_gpu and torch.cuda.is_available():
        self._device = torch.device("cuda")
    else:
        self._device = torch.device("cpu")

    # TorchScript 모델 로드
    self._model = torch.jit.load(str(model_path), map_location=self._device)
    self._model.eval()  # 추론 모드

```

**TorchScript 사용**: 일반 PyTorch 모델보다 추론 속도가 빠릅니다.

### 통계 수집

각 추론의 지연 시간을 측정하여 평균을 계산합니다:

```python
def infer(self, inputs: dict[str, TensorType]) -> dict[str, TensorType]:
    start_time = time.perf_counter()

    # 추론 실행
    outputs = self._session.run(self._output_names, inputs)

    # 통계 업데이트
    latency_ms = (time.perf_counter() - start_time) * 1000
    self._inference_count += 1
    self._total_latency_ms += latency_ms

    return result

```

---

## HttpEventClient: HTTP 이벤트 전송 클라이언트

### 설계 목표

httpx를 사용하여 이벤트를 VMS 서버로 전송합니다. 동기/비동기 모두 지원하고, 재시도 정책을 적용해야 합니다.

### 동기/비동기 지원

```python
class HttpEventClient:
    def __init__(self, ...):
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    def start(self) -> None:
        """동기 클라이언트 시작"""
        self._sync_client = httpx.Client(...)

    async def start_async(self) -> None:
        """비동기 클라이언트 시작"""
        self._client = httpx.AsyncClient(...)

```

### 재시도 정책 (tenacity)

연결 오류나 타임아웃 시 자동으로 재시도합니다:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    reraise=True,
)
def _send_with_retry(self, payload: list[dict]) -> httpx.Response:
    return self._sync_client.post(self._endpoint, json=payload)

```

**재시도 전략**:

- 최대 3회 재시도
- 지수 백오프: 0.5초 → 1초 → 2초 → 4초 (최대)
- 타임아웃/연결 오류만 재시도

### 리소스 누수 방지

`create_http_transport()` 함수는 클라이언트를 시작만 하고 종료하지 않아 리소스 누수가 발생할 수 있습니다. 이를 해결하기 위해 `HttpTransport` 래퍼 클래스를 추가했습니다:

```python
class HttpTransport:
    """HTTP 전송 래퍼 클래스."""

    def __init__(self, base_url: str, ...):
        self._client = HttpEventClient(base_url, ...)
        self._client.start()

    def __call__(self, events: list[Event]) -> bool:
        """이벤트 전송 (EventEmitter 콜백으로 사용)."""
        return self._client.send_events(events)

    def close(self) -> None:
        """리소스 정리."""
        self._client.stop()

    def __enter__(self) -> HttpTransport:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

```

**사용 예시**:

```python
# 컨텍스트 매니저로 사용
with HttpTransport(base_url="<http://vms.example.com>") as transport:
    emitter.set_sync_transport(transport)
    # ... 사용
# 자동으로 close() 호출

```

---

## WebSocketEventClient: WebSocket 이벤트 전송 클라이언트

### 설계 목표

websockets를 사용하여 이벤트를 실시간으로 전송합니다. 연결 끊김 시 자동 재연결이 필요합니다.

### 연결 상태 관리

```python
class ConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSED = "closed"

```

### 자동 재연결

연결이 끊기면 지수 백오프 방식으로 재연결을 시도합니다:

```python
async def _reconnect(self) -> None:
    while self._running and self._reconnect_count < self._reconnect_max_retries:
        self._reconnect_count += 1

        # 지수 백오프 계산
        delay = min(
            self._reconnect_interval * (2 ** (self._reconnect_count - 1)),
            self._reconnect_max_interval,
        )

        await asyncio.sleep(delay)

        try:
            self._ws = await websockets.connect(self._url, ...)
            self._state = ConnectionState.CONNECTED
            return
        except Exception as e:
            self._logger.warning("재연결 실패", error=str(e))

```

### 메시지 수신 루프

백그라운드에서 메시지를 수신하고 콜백을 호출합니다:

```python
async def receive_loop(self) -> None:
    while self._running and self.is_connected:
        try:
            message = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
            data = json.loads(message)

            if self._on_message:
                self._on_message(data)
        except websockets.exceptions.ConnectionClosed:
            if self._running:
                await self._reconnect()
            break

```

---

## 설계 결정 과정

### 1. OpenCV vs FFmpeg 직접 사용

**선택**: OpenCV VideoCapture

**이유**:

- 간단한 API
- 프레임을 numpy 배열로 직접 반환
- 버퍼 관리 자동화

**단점**: 타임아웃 설정이 빌드에 따라 다름 (try/except로 처리)

### 2. FFmpeg stdout/stderr 처리

**선택**: DEVNULL

**이유**:

- 파이프 데드락 방지
- 로그가 필요 없음 (RTSP 퍼블리싱)

**대안**: 로그가 필요한 경우 별도 스레드에서 소비

### 3. ONNX vs PyTorch

**선택**: 둘 다 지원 (추상 클래스)

**이유**:

- 모델 형식에 따라 선택 가능
- ONNX: 범용성, 최적화
- PyTorch: 유연성, TorchScript

### 4. 동기 vs 비동기 HTTP

**선택**: 둘 다 지원

**이유**:

- Application Layer는 동기 (단순성)
- 향후 비동기 확장 가능

### 5. WebSocket 자동 재연결

**선택**: 지수 백오프

**이유**:

- 서버 부하 감소
- 네트워크 일시 장애 대응

---

### 핵심 기능

| 컴포넌트 | 주요 기능 |
| --- | --- |
| RTSPDecoder | RTSP 연결, 프레임 읽기, 지수 백오프 재연결 |
| FFmpegPublisher | FFmpeg subprocess 관리, 파이프 데드락 방지 |
| ONNXRuntime | ONNX 모델 로드, GPU/CPU 자동 선택, 통계 수집 |
| PyTorchRuntime | TorchScript 모델 로드, GPU/CPU 자동 선택 |
| HttpEventClient | 동기/비동기 전송, tenacity 재시도 |
| WebSocketEventClient | 실시간 전송, 자동 재연결, 메시지 수신 |

---

## 테스트 전략

### 단위 테스트

```python
def test_rtsp_decoder_reconnect():
    """RTSP 재연결 테스트"""
    decoder = RTSPDecoder("test_stream")

    # 첫 연결 실패 시뮬레이션
    with patch("cv2.VideoCapture") as mock_cap:
        mock_cap.return_value.isOpened.return_value = False

        # 재연결 시도
        result = decoder.reconnect(max_retries=3, base_delay=0.1)
        assert not result  # 실패

        # 성공 시뮬레이션
        mock_cap.return_value.isOpened.return_value = True
        result = decoder.reconnect(max_retries=3, base_delay=0.1)
        assert result  # 성공

def test_ffmpeg_publisher_pipe_deadlock():
    """FFmpeg 파이프 데드락 방지 테스트"""
    publisher = FFmpegPublisher(...)

    with patch("subprocess.Popen") as mock_popen:
        publisher.start()

        # DEVNULL 사용 확인
        call_args = mock_popen.call_args
        assert call_args[1]["stdout"] == subprocess.DEVNULL
        assert call_args[1]["stderr"] == subprocess.DEVNULL

```

### 통합 테스트

```python
def test_rtsp_to_ffmpeg_pipeline():
    """RTSP → FFmpeg 파이프라인 통합 테스트"""
    decoder = RTSPDecoder("test_stream")
    publisher = FFmpegPublisher(...)

    # 연결
    decoder.connect("rtsp://test")
    publisher.start()

    # 프레임 처리
    success, frame = decoder.read_frame()
    if success:
        publisher.write_frame(frame)

    # 정리
    decoder.release()
    publisher.stop()
```

---

## 성능 고려사항

### RTSP 버퍼 크기

버퍼가 너무 크면 지연이 증가하고, 너무 작으면 프레임 드롭이 발생합니다:

```python
buffer_size: int = 2  # 기본값: 2프레임
```

### FFmpeg 프리셋

`ultrafast` 프리셋을 사용하여 인코딩 지연을 최소화합니다:

```python
preset: str = "ultrafast"  # 지연 최소화
```

**트레이드오프**: 파일 크기 증가, 품질 약간 저하

### 추론 런타임 선택

- **ONNX**: 범용성, 최적화, 빠른 추론
- **PyTorch**: 유연성, TorchScript 최적화

---

## 마무리

Infrastructure Layer는 외부 시스템과의 인터페이스를 담당합니다. 핵심은:

1. **안정성**: 파이프 데드락 방지, 리소스 누수 방지
2. **호환성**: 플랫폼별 차이 처리 (OpenCV 타임아웃)
3. **복구**: 지수 백오프 재연결, 자동 재시도
4. **성능**: 최적화된 설정 (FFmpeg 프리셋, 버퍼 크기)

---

## 참고 자료

- [OpenCV VideoCapture 문서](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [FFmpeg 문서](https://ffmpeg.org/documentation.html)
- [ONNX Runtime 문서](https://onnxruntime.ai/docs/)
- [httpx 문서](https://www.python-httpx.org/)
- [websockets 문서](https://websockets.readthedocs.io/)
- [프로젝트 Infrastructure Layer 코드](https://github.com/dusen0528/SentinelPipeline/tree/dev/src/sentinel_pipeline/infrastructure)