"""
구조화 로깅 모듈

SentinelPipeline 전체에서 사용하는 로깅 설정과 유틸리티를 제공합니다.
loguru 기반으로 구조화된 로깅을 지원합니다.

주요 기능:
- JSON 형식 출력 (운영 환경)
- 컬러 콘솔 출력 (개발 환경)
- 컨텍스트 바인딩 (module_name, stream_id, trace_id)
- 자동 trace_id 생성
"""

import sys
import os
import uuid
from contextvars import ContextVar
from typing import Any
from functools import lru_cache

from loguru import logger


# 컨텍스트 변수: 요청별 trace_id 저장
_trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
_stream_id_var: ContextVar[str | None] = ContextVar("stream_id", default=None)
_module_name_var: ContextVar[str | None] = ContextVar("module_name", default=None)


def get_trace_id() -> str:
    """
    현재 컨텍스트의 trace_id를 반환합니다.
    
    설정되지 않은 경우 새로 생성합니다.
    """
    trace_id = _trace_id_var.get()
    if trace_id is None:
        trace_id = generate_trace_id()
        _trace_id_var.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> None:
    """trace_id를 현재 컨텍스트에 설정합니다."""
    _trace_id_var.set(trace_id)


def generate_trace_id() -> str:
    """새로운 trace_id를 생성합니다."""
    return uuid.uuid4().hex[:12]


def set_stream_context(stream_id: str | None) -> None:
    """스트림 ID를 현재 컨텍스트에 설정합니다."""
    _stream_id_var.set(stream_id)


def set_module_context(module_name: str | None) -> None:
    """모듈 이름을 현재 컨텍스트에 설정합니다."""
    _module_name_var.set(module_name)


def _get_context_extra() -> dict[str, Any]:
    """현재 컨텍스트의 추가 정보를 반환합니다."""
    extra: dict[str, Any] = {}
    
    trace_id = _trace_id_var.get()
    if trace_id:
        extra["trace_id"] = trace_id
    
    stream_id = _stream_id_var.get()
    if stream_id:
        extra["stream_id"] = stream_id
    
    module_name = _module_name_var.get()
    if module_name:
        extra["module_name"] = module_name
    
    return extra


def _json_formatter(record: dict[str, Any]) -> str:
    """
    JSON 형식의 로그 포맷터
    
    운영 환경에서 로그 집계 시스템(ELK, Loki 등)과 호환됩니다.
    """
    import orjson
    
    log_entry = {
        "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["name"],
    }
    
    # 컨텍스트 정보 추가
    if record.get("extra"):
        for key in ["trace_id", "stream_id", "module_name"]:
            if key in record["extra"]:
                log_entry[key] = record["extra"][key]
        
        # 추가 extra 필드
        for key, value in record["extra"].items():
            if key not in ["trace_id", "stream_id", "module_name"]:
                log_entry[key] = value
    
    # 예외 정보 추가
    if record["exception"]:
        log_entry["exception"] = {
            "type": record["exception"].type.__name__ if record["exception"].type else None,
            "value": str(record["exception"].value) if record["exception"].value else None,
            "traceback": record["exception"].traceback if record["exception"].traceback else None,
        }
    
    return orjson.dumps(log_entry).decode("utf-8") + "\n"


def _console_formatter(record: dict[str, Any]) -> str:
    """
    컬러 콘솔 형식의 로그 포맷터
    
    개발 환경에서 가독성을 높입니다.
    """
    # 기본 포맷
    fmt = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    fmt += "<level>{level: <8}</level> | "
    fmt += "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    
    # 컨텍스트 정보
    extra_parts = []
    if record.get("extra"):
        if "trace_id" in record["extra"]:
            extra_parts.append(f"<yellow>trace={record['extra']['trace_id']}</yellow>")
        if "stream_id" in record["extra"]:
            extra_parts.append(f"<blue>stream={record['extra']['stream_id']}</blue>")
        if "module_name" in record["extra"]:
            extra_parts.append(f"<magenta>module={record['extra']['module_name']}</magenta>")
    
    if extra_parts:
        fmt += " ".join(extra_parts) + " | "
    
    fmt += "<level>{message}</level>\n"
    
    if record["exception"]:
        fmt += "{exception}"
    
    return fmt


# 로그 레벨 매핑
_LOG_LEVELS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "WARN": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


def configure_logging(
    level: str = "INFO",
    json_output: bool | None = None,
    log_file: str | None = None,
) -> None:
    """
    로깅 설정을 초기화합니다.
    
    Args:
        level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: JSON 형식 출력 여부 (None이면 환경변수로 결정)
        log_file: 로그 파일 경로 (None이면 stdout만 출력)
    
    환경변수:
        LOG_LEVEL: 로그 레벨 (기본: INFO)
        LOG_FORMAT: 로그 포맷 (json 또는 console, 기본: console)
        LOG_FILE: 로그 파일 경로
    """
    # 환경변수에서 설정 읽기
    env_level = os.getenv("LOG_LEVEL", level).upper()
    log_level = _LOG_LEVELS.get(env_level, "INFO")
    
    if json_output is None:
        json_output = os.getenv("LOG_FORMAT", "console").lower() == "json"
    
    if log_file is None:
        log_file = os.getenv("LOG_FILE")
    
    # 기존 핸들러 제거
    logger.remove()
    
    # 콘솔 출력 설정
    if json_output:
        logger.add(
            sys.stdout,
            format=_json_formatter,
            level=log_level,
            serialize=False,
        )
    else:
        logger.add(
            sys.stdout,
            format=_console_formatter,
            level=log_level,
            colorize=True,
        )
    
    # 파일 출력 설정 (선택)
    if log_file:
        logger.add(
            log_file,
            format=_json_formatter,
            level=log_level,
            rotation="100 MB",
            retention="7 days",
            compression="gz",
        )
    
    logger.info(
        f"로깅 설정 완료: level={log_level}, json={json_output}, file={log_file}"
    )


class BoundLogger:
    """
    컨텍스트가 바인딩된 로거
    
    특정 모듈이나 스트림에서 사용하기 위해 컨텍스트 정보가 자동으로 포함됩니다.
    """
    
    def __init__(
        self,
        name: str,
        module_name: str | None = None,
        stream_id: str | None = None,
    ) -> None:
        self._name = name
        self._module_name = module_name
        self._stream_id = stream_id
        self._logger = logger.bind(name=name)
    
    def _get_extra(self, **kwargs: Any) -> dict[str, Any]:
        """로그에 포함할 extra 정보를 구성합니다."""
        extra = _get_context_extra()
        
        # 인스턴스 레벨 컨텍스트
        if self._module_name and "module_name" not in extra:
            extra["module_name"] = self._module_name
        if self._stream_id and "stream_id" not in extra:
            extra["stream_id"] = self._stream_id
        
        # 호출 시 전달된 추가 정보
        extra.update(kwargs)
        
        return extra
    
    def bind(self, **kwargs: Any) -> "BoundLogger":
        """추가 컨텍스트를 바인딩한 새 로거를 반환합니다."""
        new_logger = BoundLogger(
            name=self._name,
            module_name=kwargs.get("module_name", self._module_name),
            stream_id=kwargs.get("stream_id", self._stream_id),
        )
        return new_logger
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """DEBUG 레벨 로그를 기록합니다."""
        self._logger.bind(**self._get_extra(**kwargs)).debug(message)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """INFO 레벨 로그를 기록합니다."""
        self._logger.bind(**self._get_extra(**kwargs)).info(message)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """WARNING 레벨 로그를 기록합니다."""
        self._logger.bind(**self._get_extra(**kwargs)).warning(message)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """ERROR 레벨 로그를 기록합니다."""
        self._logger.bind(**self._get_extra(**kwargs)).error(message)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """CRITICAL 레벨 로그를 기록합니다."""
        self._logger.bind(**self._get_extra(**kwargs)).critical(message)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        """예외 정보와 함께 ERROR 레벨 로그를 기록합니다."""
        self._logger.bind(**self._get_extra(**kwargs)).exception(message)


@lru_cache(maxsize=128)
def get_logger(
    name: str,
    module_name: str | None = None,
    stream_id: str | None = None,
) -> BoundLogger:
    """
    로거 인스턴스를 반환합니다.
    
    동일한 인자로 호출하면 캐시된 인스턴스를 반환합니다.
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        module_name: 모듈 이름 (파이프라인 모듈용)
        stream_id: 스트림 ID (스트림 처리용)
    
    Returns:
        BoundLogger 인스턴스
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("처리 시작")
        
        >>> module_logger = get_logger(__name__, module_name="FireDetect")
        >>> module_logger.info("화재 감지 모듈 초기화")
    """
    return BoundLogger(name=name, module_name=module_name, stream_id=stream_id)


# 기본 로깅 설정 (모듈 임포트 시 실행)
# 애플리케이션에서 configure_logging()을 호출하여 재설정 가능
if not os.getenv("SENTINEL_SKIP_DEFAULT_LOGGING"):
    configure_logging()

