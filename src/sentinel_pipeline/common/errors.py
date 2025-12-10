"""
에러 처리 모듈

SentinelPipeline 전체에서 사용하는 예외 클래스와 에러 코드를 정의합니다.
모든 예외는 SentinelError를 상속받아 일관된 에러 처리가 가능합니다.
"""

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """
    에러 코드 열거형
    
    HTTP 상태 코드와 매핑되어 REST API 응답에 사용됩니다.
    """
    
    # 모듈 관련 (5xx)
    MODULE_TIMEOUT = "MODULE_TIMEOUT"           # 모듈 처리 시간 초과
    MODULE_FAILED = "MODULE_FAILED"             # 모듈 내부 오류
    MODULE_NOT_FOUND = "MODULE_NOT_FOUND"       # 모듈 없음
    MODULE_DISABLED = "MODULE_DISABLED"         # 모듈 비활성화됨
    MODULE_LOAD_FAILED = "MODULE_LOAD_FAILED"   # 모듈 로드 실패
    
    # 스트림 관련 (4xx, 5xx)
    STREAM_NOT_FOUND = "STREAM_NOT_FOUND"               # 스트림 없음
    STREAM_ALREADY_RUNNING = "STREAM_ALREADY_RUNNING"   # 스트림 중복 시작
    STREAM_ALREADY_STOPPED = "STREAM_ALREADY_STOPPED"   # 스트림 이미 중지됨
    STREAM_CONNECTION_FAILED = "STREAM_CONNECTION_FAILED"  # RTSP 연결 실패
    STREAM_DECODE_ERROR = "STREAM_DECODE_ERROR"         # 디코딩 오류
    
    # 설정 관련 (4xx)
    CONFIG_INVALID = "CONFIG_INVALID"           # 설정 검증 실패
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"       # 설정 파일 없음
    CONFIG_PARSE_ERROR = "CONFIG_PARSE_ERROR"   # 설정 파싱 오류
    
    # 인증/권한 관련 (4xx)
    UNAUTHORIZED = "UNAUTHORIZED"               # 인증 실패
    FORBIDDEN = "FORBIDDEN"                     # 권한 없음
    
    # 전송 관련 (5xx)
    TRANSPORT_FAILED = "TRANSPORT_FAILED"       # 이벤트 전송 실패
    TRANSPORT_TIMEOUT = "TRANSPORT_TIMEOUT"     # 전송 타임아웃
    
    # 리소스 관련 (5xx)
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"   # 리소스 부족 (메모리, CPU)
    QUEUE_FULL = "QUEUE_FULL"                   # 큐 가득 참
    
    # 추론 관련 (5xx)
    INFERENCE_FAILED = "INFERENCE_FAILED"               # 추론 실패
    INFERENCE_MODEL_NOT_FOUND = "INFERENCE_MODEL_NOT_FOUND"  # 모델 파일 없음
    INFERENCE_LOAD_FAILED = "INFERENCE_LOAD_FAILED"     # 모델 로드 실패
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"             # 모델 로드 실패 (deprecated)
    
    # 일반 (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"           # 내부 오류
    UNKNOWN_ERROR = "UNKNOWN_ERROR"             # 알 수 없는 오류


# 에러 코드 → HTTP 상태 코드 매핑
_ERROR_CODE_TO_HTTP_STATUS: dict[ErrorCode, int] = {
    # 4xx Client Errors
    ErrorCode.STREAM_NOT_FOUND: 404,
    ErrorCode.MODULE_NOT_FOUND: 404,
    ErrorCode.CONFIG_NOT_FOUND: 404,
    
    ErrorCode.CONFIG_INVALID: 400,
    ErrorCode.CONFIG_PARSE_ERROR: 400,
    
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    
    ErrorCode.STREAM_ALREADY_RUNNING: 409,
    ErrorCode.STREAM_ALREADY_STOPPED: 409,
    
    # 5xx Server Errors
    ErrorCode.MODULE_TIMEOUT: 504,
    ErrorCode.TRANSPORT_TIMEOUT: 504,
    
    ErrorCode.MODULE_FAILED: 500,
    ErrorCode.MODULE_DISABLED: 500,
    ErrorCode.MODULE_LOAD_FAILED: 500,
    ErrorCode.STREAM_CONNECTION_FAILED: 500,
    ErrorCode.STREAM_DECODE_ERROR: 500,
    ErrorCode.TRANSPORT_FAILED: 500,
    ErrorCode.RESOURCE_EXHAUSTED: 500,
    ErrorCode.QUEUE_FULL: 500,
    ErrorCode.INFERENCE_FAILED: 500,
    ErrorCode.INFERENCE_MODEL_NOT_FOUND: 404,
    ErrorCode.INFERENCE_LOAD_FAILED: 500,
    ErrorCode.MODEL_LOAD_FAILED: 500,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.UNKNOWN_ERROR: 500,
}


def get_http_status(error_code: ErrorCode) -> int:
    """
    에러 코드에 해당하는 HTTP 상태 코드를 반환합니다.
    
    Args:
        error_code: 에러 코드
        
    Returns:
        HTTP 상태 코드 (기본값: 500)
    """
    return _ERROR_CODE_TO_HTTP_STATUS.get(error_code, 500)


class SentinelError(Exception):
    """
    SentinelPipeline 기본 예외 클래스
    
    모든 커스텀 예외의 부모 클래스입니다.
    에러 코드, 메시지, 상세 정보를 포함합니다.
    
    Attributes:
        code: 에러 코드 (ErrorCode)
        message: 사용자에게 표시할 메시지
        details: 추가 상세 정보 (디버깅용)
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    @property
    def http_status(self) -> int:
        """HTTP 상태 코드 반환"""
        return get_http_status(self.code)
    
    def to_dict(self) -> dict[str, Any]:
        """
        예외 정보를 딕셔너리로 변환합니다.
        
        REST API 응답에서 사용됩니다.
        """
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code.value}, message={self.message!r})"


class ModuleError(SentinelError):
    """
    모듈 관련 예외
    
    파이프라인 모듈의 처리 중 발생하는 오류를 나타냅니다.
    
    Attributes:
        module_name: 오류가 발생한 모듈 이름
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        module_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.module_name = module_name
        _details = {"module_name": module_name}
        if details:
            _details.update(details)
        super().__init__(code, message, _details)


class StreamError(SentinelError):
    """
    스트림 관련 예외
    
    RTSP 스트림의 연결, 디코딩, 관리 중 발생하는 오류를 나타냅니다.
    
    Attributes:
        stream_id: 오류가 발생한 스트림 ID
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        stream_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.stream_id = stream_id
        _details = {"stream_id": stream_id}
        if details:
            _details.update(details)
        super().__init__(code, message, _details)


class ConfigError(SentinelError):
    """
    설정 관련 예외
    
    설정 파일의 로드, 파싱, 검증 중 발생하는 오류를 나타냅니다.
    
    Attributes:
        config_path: 오류가 발생한 설정 파일 경로 (선택)
        field_name: 오류가 발생한 필드 이름 (선택)
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        config_path: str | None = None,
        field_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = config_path
        self.field_name = field_name
        _details: dict[str, Any] = {}
        if config_path:
            _details["config_path"] = config_path
        if field_name:
            _details["field_name"] = field_name
        if details:
            _details.update(details)
        super().__init__(code, message, _details)


class TransportError(SentinelError):
    """
    전송 관련 예외
    
    이벤트 전송(HTTP, WebSocket) 중 발생하는 오류를 나타냅니다.
    
    Attributes:
        transport_type: 전송 타입 (http, websocket)
        target_url: 대상 URL
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        transport_type: str,
        target_url: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.transport_type = transport_type
        self.target_url = target_url
        _details: dict[str, Any] = {"transport_type": transport_type}
        if target_url:
            _details["target_url"] = target_url
        if details:
            _details.update(details)
        super().__init__(code, message, _details)


class InferenceError(SentinelError):
    """
    추론 관련 예외
    
    AI 모델 로드, 추론 중 발생하는 오류를 나타냅니다.
    
    Attributes:
        model_name: 모델 이름 또는 경로
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        model_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        _details = {"model_name": model_name}
        if details:
            _details.update(details)
        super().__init__(code, message, _details)


class ResourceError(SentinelError):
    """
    리소스 관련 예외
    
    메모리, CPU, 큐 등 리소스 부족 시 발생하는 오류를 나타냅니다.
    
    Attributes:
        resource_type: 리소스 타입 (memory, cpu, queue)
        current_usage: 현재 사용량
        limit: 제한값
    """
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        resource_type: str,
        current_usage: float | None = None,
        limit: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        _details: dict[str, Any] = {"resource_type": resource_type}
        if current_usage is not None:
            _details["current_usage"] = current_usage
        if limit is not None:
            _details["limit"] = limit
        if details:
            _details.update(details)
        super().__init__(code, message, _details)

