"""
설정 스키마 (Pydantic v2)

config.json을 검증하기 위한 스키마를 정의합니다.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Pydantic 모델은 Interface Layer에서만 외부 라이브러리에 의존합니다.


class ModuleConfig(BaseModel):
    """모듈 설정 스키마."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    name: str = Field(..., description="모듈 이름")
    enabled: bool = Field(True, description="활성화 여부")
    priority: int = Field(100, description="우선순위 (낮을수록 먼저 실행)")
    timeout_ms: int = Field(50, description="처리 타임아웃 (밀리초)")
    options: dict[str, Any] = Field(default_factory=dict, description="모듈별 옵션")

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise ValueError("모듈 이름은 비워둘 수 없습니다")
        return value

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: int) -> int:
        if not 0 <= value <= 1000:
            raise ValueError("priority는 0~1000 범위여야 합니다")
        return value

    @field_validator("timeout_ms")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if not 10 <= value <= 5000:
            raise ValueError("timeout_ms는 10~5000 범위여야 합니다")
        return value


class StreamConfig(BaseModel):
    """스트림 설정 스키마."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    stream_id: str = Field(..., description="스트림 고유 ID")
    rtsp_url: str = Field(..., description="RTSP URL")
    enabled: bool = Field(True, description="활성화 여부")
    max_fps: int = Field(30, description="최대 FPS")
    downscale: float = Field(1.0, description="프레임 축소 비율 (0~1)")
    buffer_size: int = Field(2, description="프레임 버퍼 크기")

    reconnect_enabled: bool = Field(True, description="재연결 활성화 여부")
    reconnect_max_retries: int = Field(5, description="재연결 최대 시도")
    reconnect_base_delay: float = Field(1.0, description="재연결 기본 대기 (초)")
    reconnect_max_delay: float = Field(8.0, description="재연결 최대 대기 (초)")

    output_url: str | None = Field(None, description="출력 RTSP/RTMP URL")
    output_enabled: bool = Field(True, description="출력 활성화 여부")

    target_width: int | None = Field(None, description="목표 출력 너비 (고정)")
    target_height: int | None = Field(None, description="목표 출력 높이 (고정)")

    @field_validator("stream_id", "rtsp_url")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("필수 필드는 비워둘 수 없습니다")
        return value

    @field_validator("max_fps")
    @classmethod
    def validate_fps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_fps는 0보다 커야 합니다")
        return value

    @field_validator("downscale")
    @classmethod
    def validate_downscale(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("downscale은 0.0보다 크고 1.0 이하여야 합니다")
        return value

    @field_validator("buffer_size")
    @classmethod
    def validate_buffer(cls, value: int) -> int:
        if value < 1:
            raise ValueError("buffer_size는 1 이상이어야 합니다")
        return value

    @field_validator("reconnect_max_retries")
    @classmethod
    def validate_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("reconnect_max_retries는 0 이상이어야 합니다")
        return value

    @field_validator("reconnect_base_delay", "reconnect_max_delay")
    @classmethod
    def validate_delays(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("재연결 대기 시간은 0보다 커야 합니다")
        return value


class AudioStreamConfig(BaseModel):
    """오디오 스트림 설정 스키마."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    stream_id: str = Field(..., description="스트림 고유 ID")
    rtsp_url: str | None = Field(None, description="RTSP URL (마이크 미사용 시 필수)")
    use_microphone: bool = Field(False, description="마이크 사용 여부")
    mic_device_index: int | None = Field(None, description="마이크 디바이스 인덱스")
    enabled: bool = Field(True, description="활성화 여부")
    
    # 오디오 처리 설정
    sample_rate: int = Field(16000, description="샘플링 레이트")
    chunk_duration: float = Field(2.0, description="청크 길이 (초)")
    
    # 분석 설정
    scream_threshold: float = Field(0.8, description="비명 감지 임계값")
    scream_model_path: str | None = Field(None, description="비명 감지 모델 파일 경로 (None이면 기본값 사용)")
    scream_model_arch: str = Field("resnet18", description="비명 감지 모델 아키텍처 ('resnet18')")
    scream_enable_filtering: bool = Field(True, description="비명 감지 필터링 활성화 여부")
    stt_enabled: bool = Field(True, description="STT 활성화 여부")
    stt_model_size: str = Field("base", description="Whisper 모델 크기")
    
    # 하이브리드 키워드 감지 설정
    enable_medium_path: bool = Field(True, description="Medium Path(형태소 분석) 활성화")
    enable_heavy_path: bool = Field(True, description="Heavy Path(의미 유사도) 활성화")
    heavy_path_async: bool = Field(True, description="Heavy Path 비동기 처리 여부")
    semantic_threshold: float = Field(0.7, description="의미 유사도 임계값")
    use_korean_model: bool = Field(False, description="한국어 특화 모델 사용 여부")

    @model_validator(mode="after")
    def validate_source(self) -> "AudioStreamConfig":
        if not self.use_microphone and not self.rtsp_url:
            raise ValueError("마이크를 사용하지 않을 경우 rtsp_url은 필수입니다")
        return self


class PipelineConfig(BaseModel):
    """파이프라인 설정 스키마."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    max_workers: int = Field(4, description="모듈 실행 스레드 풀 크기")
    max_consecutive_errors: int = Field(5, description="연속 오류 임계값")
    max_consecutive_timeouts: int = Field(10, description="연속 타임아웃 임계값")
    error_window_seconds: int = Field(60, description="에러 윈도우 (초)")
    auto_disable_threshold: int = Field(10, description="자동 비활성화 임계값(레거시, 추후 제거 예정)")
    cooldown_seconds: int = Field(300, description="재활성화 대기 (초)")

    @model_validator(mode="after")
    def validate_values(self) -> "PipelineConfig":
        if self.max_workers < 1:
            raise ValueError("max_workers는 1 이상이어야 합니다")
        if self.max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors는 1 이상이어야 합니다")
        if self.max_consecutive_timeouts < 1:
            raise ValueError("max_consecutive_timeouts는 1 이상이어야 합니다")
        if self.error_window_seconds < 1:
            raise ValueError("error_window_seconds는 1 이상이어야 합니다")
        if self.auto_disable_threshold < 1:
            raise ValueError("auto_disable_threshold는 1 이상이어야 합니다")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds는 0 이상이어야 합니다")
        return self


class EventTransportConfig(BaseModel):
    """이벤트 전송 설정."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    type: Literal["http", "ws"] = Field(..., description="전송 타입")
    url: str = Field(..., description="대상 URL")
    enabled: bool = Field(True, description="활성화 여부")
    headers: dict[str, str] = Field(default_factory=dict, description="추가 헤더")
    batch_size: int = Field(10, description="전송 배치 크기")
    flush_interval_ms: int = Field(100, description="플러시 주기")

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not value:
            raise ValueError("url은 비워둘 수 없습니다")
        return value

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다")
        return value

    @field_validator("flush_interval_ms")
    @classmethod
    def validate_flush_interval(cls, value: int) -> int:
        if value < 10:
            raise ValueError("flush_interval_ms는 10 이상이어야 합니다")
        return value


class EventConfig(BaseModel):
    """이벤트 처리 설정."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    max_queue_size: int = Field(1000, description="이벤트 큐 최대 크기")
    batch_size: int = Field(10, description="배치 크기")
    flush_interval_ms: int = Field(100, description="플러시 주기")
    drop_strategy: Literal["drop_oldest", "drop_newest"] = Field(
        "drop_oldest", description="드롭 전략"
    )
    transports: list[EventTransportConfig] = Field(default_factory=list, description="전송 설정")

    @model_validator(mode="after")
    def validate_values(self) -> "EventConfig":
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size는 1 이상이어야 합니다")
        if self.batch_size < 1:
            raise ValueError("batch_size는 1 이상이어야 합니다")
        if self.flush_interval_ms < 10:
            raise ValueError("flush_interval_ms는 10 이상이어야 합니다")
        return self


class ObservabilityConfig(BaseModel):
    """관측 설정."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    metrics_enabled: bool = Field(True, description="메트릭 활성화")
    metrics_host: str = Field("0.0.0.0", description="메트릭 바인딩 호스트")
    metrics_port: int = Field(9100, description="메트릭 포트")
    log_level: str = Field("INFO", description="로그 레벨")

    @field_validator("metrics_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("metrics_port는 1~65535 범위여야 합니다")
        return value


class GlobalConfig(BaseModel):
    """전역 설정 (런타임 파라미터)."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    max_fps: int = Field(30, description="최대 FPS")
    downscale: float = Field(0.5, description="프레임 축소 비율")
    queue_max: int = Field(2, description="프레임 큐 최대 크기")
    drop_strategy: Literal["drop_oldest", "drop_newest"] = Field(
        "drop_oldest", description="프레임 드롭 전략"
    )
    inference_timeout_ms: int = Field(50, description="추론 타임아웃 (밀리초)")

    target_width: int | None = Field(None, description="기본 목표 출력 너비")
    target_height: int | None = Field(None, description="기본 목표 출력 높이")

    @model_validator(mode="after")
    def validate_values(self) -> "GlobalConfig":
        if self.max_fps <= 0:
            raise ValueError("max_fps는 0보다 커야 합니다")
        if not 0.0 < self.downscale <= 1.0:
            raise ValueError("downscale은 0.0보다 크고 1.0 이하여야 합니다")
        if self.queue_max < 1:
            raise ValueError("queue_max는 1 이상이어야 합니다")
        if self.inference_timeout_ms < 1:
            raise ValueError("inference_timeout_ms는 1 이상이어야 합니다")
        return self


class AppConfig(BaseModel):
    """애플리케이션 전체 설정 스키마."""

    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
    )

    modules: list[ModuleConfig] = Field(default_factory=list, description="모듈 설정 목록")
    streams: list[StreamConfig] = Field(default_factory=list, description="비디오 스트림 설정 목록")
    audio_streams: list[AudioStreamConfig] = Field(default_factory=list, description="오디오 스트림 설정 목록")
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig, description="파이프라인 설정")
    event: EventConfig = Field(default_factory=EventConfig, description="이벤트 설정")
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="관측 설정",
    )
    global_config: GlobalConfig = Field(
        default_factory=GlobalConfig,
        alias="global",
        description="전역 처리 설정",
    )

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "AppConfig":
        module_names = [m.name for m in self.modules]
        if len(module_names) != len(set(module_names)):
            raise ValueError("모듈 이름이 중복됩니다")

        stream_ids = [s.stream_id for s in self.streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise ValueError("비디오 스트림 ID가 중복됩니다")
            
        audio_stream_ids = [s.stream_id for s in self.audio_streams]
        if len(audio_stream_ids) != len(set(audio_stream_ids)):
            raise ValueError("오디오 스트림 ID가 중복됩니다")
            
        # 성능 테스트 결과에 따른 오디오 스트림 개수 제한 (Hard Limit: 9)
        if len(audio_stream_ids) > 9:
            raise ValueError(f"오디오 스트림 개수({len(audio_stream_ids)})가 시스템 허용 한계(9개)를 초과했습니다. 성능 저하를 방지하기 위해 스트림 수를 줄여주세요.")

        return self


