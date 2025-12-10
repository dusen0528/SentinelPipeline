"""
모듈 인터페이스 정의

파이프라인에서 실행되는 모든 처리 모듈(감지, 변형 등)이 구현해야 하는
인터페이스를 Protocol로 정의합니다.

이 모듈은 외부 라이브러리에 의존하지 않습니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sentinel_pipeline.domain.models.event import Event


# 프레임 타입 힌트 (numpy.ndarray이지만 domain에서는 Any로 처리)
FrameType = Any

# 오디오 청크 타입 힌트
AudioChunkType = Any

# 메타데이터 타입
MetadataType = dict[str, Any]


@runtime_checkable
class ModuleBase(Protocol):
    """
    파이프라인 모듈 기본 인터페이스
    
    모든 감지/변형 모듈은 이 Protocol을 구현해야 합니다.
    PipelineEngine이 이 인터페이스를 통해 모듈을 실행합니다.
    
    Attributes:
        name: 모듈 고유 식별자 (예: "FireDetectModule")
        enabled: 활성화 여부. False이면 파이프라인에서 건너뜀
        priority: 실행 우선순위. 낮을수록 먼저 실행 (0~1000)
        timeout_ms: 처리 제한 시간(밀리초). 초과 시 결과 폐기
        options: 모듈별 설정 옵션
    
    Example:
        >>> class FireDetectModule:
        ...     name = "FireDetectModule"
        ...     enabled = True
        ...     priority = 100
        ...     timeout_ms = 50
        ...     options = {"threshold": 0.6}
        ...     
        ...     def process_frame(self, frame, metadata):
        ...         # 화재 감지 로직
        ...         events = []
        ...         return frame, events, metadata
        ...     
        ...     def process_audio(self, chunk, metadata):
        ...         return [], metadata
    """
    
    # === 필수 속성 ===
    
    @property
    def name(self) -> str:
        """
        모듈 고유 식별자
        
        config.json의 modules[].name과 일치해야 합니다.
        플러그인 레지스트리에서 이 이름으로 모듈을 찾습니다.
        """
        ...
    
    @property
    def enabled(self) -> bool:
        """
        활성화 여부
        
        False이면 파이프라인에서 이 모듈을 건너뜁니다.
        런타임에 동적으로 변경 가능합니다.
        """
        ...
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """enabled 속성 설정"""
        ...
    
    @property
    def priority(self) -> int:
        """
        실행 우선순위
        
        파이프라인에서 모듈 실행 순서를 결정합니다.
        낮은 값이 먼저 실행됩니다.
        
        권장 범위:
        - 0~99: 전처리 (크기 조정, 정규화)
        - 100~199: 감지 (화재, 침입, 비명)
        - 200~299: 변형 (블러, 마스킹)
        - 300~399: 후처리 (오버레이, 워터마크)
        """
        ...
    
    @property
    def timeout_ms(self) -> int:
        """
        처리 제한 시간 (밀리초)
        
        이 시간을 초과하면 해당 프레임의 결과를 폐기합니다.
        반복적으로 초과하면 모듈이 자동으로 비활성화될 수 있습니다.
        
        권장값: 실시간 처리를 위해 30~50ms
        """
        ...
    
    @property
    def options(self) -> dict[str, Any]:
        """
        모듈별 설정 옵션
        
        config.json의 modules[].options에서 로드됩니다.
        각 모듈이 필요한 설정을 자유롭게 정의합니다.
        
        예시:
        - threshold: 감지 임계값
        - min_consecutive: 연속 감지 횟수
        - blur_kernel: 블러 커널 크기
        """
        ...
    
    # === 필수 메서드 ===
    
    def process_frame(
        self,
        frame: FrameType,
        metadata: MetadataType,
    ) -> tuple[FrameType, list["Event"], MetadataType]:
        """
        프레임을 처리합니다.
        
        파이프라인에서 각 프레임마다 호출됩니다.
        감지 모듈은 이벤트를 생성하고, 변형 모듈은 프레임을 수정합니다.
        
        Args:
            frame: 입력 프레임 (numpy.ndarray, BGR 형식)
            metadata: 프레임 메타데이터
                - stream_id: 스트림 식별자
                - frame_number: 프레임 번호
                - timestamp: 타임스탬프
                - fps: 초당 프레임 수
                - 이전 모듈이 추가한 정보
        
        Returns:
            tuple: (처리된 프레임, 생성된 이벤트 목록, 업데이트된 메타데이터)
            
            - frame: 변형된 프레임 또는 원본 그대로
            - events: 감지된 이벤트 목록 (없으면 빈 리스트)
            - metadata: 다음 모듈에 전달할 메타데이터
        
        Raises:
            이 메서드에서 발생하는 예외는 파이프라인에서 격리됩니다.
            예외 발생 시 해당 모듈만 건너뛰고 다음 모듈이 실행됩니다.
        
        Example:
            >>> def process_frame(self, frame, metadata):
            ...     # 화재 감지
            ...     result = self.model.detect(frame)
            ...     
            ...     events = []
            ...     if result.confidence > self.options["threshold"]:
            ...         event = Event(
            ...             type=EventType.FIRE,
            ...             stage=EventStage.DETECTED,
            ...             confidence=result.confidence,
            ...             stream_id=metadata["stream_id"],
            ...             module_name=self.name,
            ...             details={"bbox": result.bbox}
            ...         )
            ...         events.append(event)
            ...     
            ...     return frame, events, metadata
        """
        ...
    
    def process_audio(
        self,
        chunk: AudioChunkType,
        metadata: MetadataType,
    ) -> tuple[list["Event"], MetadataType]:
        """
        오디오 청크를 처리합니다.
        
        오디오 기반 감지 모듈(비명 감지 등)에서 구현합니다.
        영상만 처리하는 모듈은 빈 리스트를 반환합니다.
        
        Args:
            chunk: 오디오 청크 (numpy.ndarray, PCM 형식)
            metadata: 오디오 메타데이터
                - stream_id: 스트림 식별자
                - sample_rate: 샘플레이트
                - channels: 채널 수
                - timestamp: 타임스탬프
        
        Returns:
            tuple: (생성된 이벤트 목록, 업데이트된 메타데이터)
        
        Example:
            >>> def process_audio(self, chunk, metadata):
            ...     # 비명 감지
            ...     result = self.audio_model.detect(chunk)
            ...     
            ...     events = []
            ...     if result.is_scream:
            ...         event = Event(
            ...             type=EventType.SCREAM,
            ...             stage=EventStage.DETECTED,
            ...             confidence=result.confidence,
            ...             stream_id=metadata["stream_id"],
            ...             module_name=self.name,
            ...         )
            ...         events.append(event)
            ...     
            ...     return events, metadata
        """
        ...


class ModuleContext:
    """
    모듈 실행 컨텍스트
    
    파이프라인에서 모듈을 실행할 때 필요한 상태 정보를 관리합니다.
    
    Attributes:
        module: 모듈 인스턴스
        error_count: 연속 에러 횟수
        timeout_count: 연속 타임아웃 횟수
        total_processed: 총 처리 프레임 수
        total_events: 총 생성 이벤트 수
        avg_latency_ms: 평균 처리 시간 (밀리초)
    """
    
    def __init__(self, module: ModuleBase) -> None:
        self.module = module
        self.error_count: int = 0
        self.timeout_count: int = 0
        self.total_processed: int = 0
        self.total_events: int = 0
        self._latency_sum: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """평균 처리 시간 (밀리초)"""
        if self.total_processed == 0:
            return 0.0
        return self._latency_sum / self.total_processed
    
    def record_success(self, latency_ms: float, event_count: int) -> None:
        """
        성공적인 처리를 기록합니다.
        
        Args:
            latency_ms: 처리 시간 (밀리초)
            event_count: 생성된 이벤트 수
        """
        self.error_count = 0  # 연속 에러 카운트 리셋
        self.timeout_count = 0  # 연속 타임아웃 카운트 리셋
        self.total_processed += 1
        self.total_events += event_count
        self._latency_sum += latency_ms
    
    def record_error(self) -> None:
        """에러 발생을 기록합니다."""
        self.error_count += 1
    
    def record_timeout(self) -> None:
        """타임아웃 발생을 기록합니다."""
        self.timeout_count += 1
    
    def should_disable(
        self,
        max_errors: int = 5,
        max_timeouts: int = 10,
    ) -> bool:
        """
        모듈을 비활성화해야 하는지 확인합니다.
        
        Args:
            max_errors: 최대 연속 에러 횟수
            max_timeouts: 최대 연속 타임아웃 횟수
        
        Returns:
            비활성화 필요 여부
        """
        return self.error_count >= max_errors or self.timeout_count >= max_timeouts
    
    def reset_counters(self) -> None:
        """에러/타임아웃 카운터를 리셋합니다."""
        self.error_count = 0
        self.timeout_count = 0
    
    def to_dict(self) -> dict[str, Any]:
        """상태를 딕셔너리로 반환합니다."""
        return {
            "name": self.module.name,
            "enabled": self.module.enabled,
            "priority": self.module.priority,
            "error_count": self.error_count,
            "timeout_count": self.timeout_count,
            "total_processed": self.total_processed,
            "total_events": self.total_events,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }

