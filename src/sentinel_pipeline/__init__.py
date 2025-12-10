"""
SentinelPipeline - 실시간 AI 영상 처리를 위한 모듈형 파이프라인 엔진

RTSP 다중 스트림을 입력받아 실시간 AI 감지(화재, 비명, 침입 등)와 
영상 변형(얼굴 블러 등)을 모듈형 파이프라인으로 처리합니다.
"""

__version__ = "0.1.0"
__author__ = "SentinelPipeline Team"

from sentinel_pipeline.common.errors import (
    SentinelError,
    ModuleError,
    StreamError,
    ConfigError,
    ErrorCode,
)
from sentinel_pipeline.common.logging import get_logger

__all__ = [
    "__version__",
    "SentinelError",
    "ModuleError", 
    "StreamError",
    "ConfigError",
    "ErrorCode",
    "get_logger",
]

