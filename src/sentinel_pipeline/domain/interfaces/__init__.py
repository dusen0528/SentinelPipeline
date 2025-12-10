"""
인터페이스 모듈

파이프라인 모듈이 구현해야 하는 인터페이스(Protocol)를 정의합니다.
"""

from sentinel_pipeline.domain.interfaces.module import BaseModule, ModuleBase, ModuleContext

__all__ = ["ModuleBase", "BaseModule", "ModuleContext"]

