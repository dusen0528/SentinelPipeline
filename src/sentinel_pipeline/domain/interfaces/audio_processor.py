"""
오디오 처리 인터페이스

오디오 리더 및 프로세서의 추상 인터페이스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any
import numpy as np


class AudioReader(ABC):
    """
    오디오 소스에서 데이터를 읽어오는 인터페이스
    """
    
    @abstractmethod
    def start(self) -> None:
        """읽기 시작"""
        pass
        
    @abstractmethod
    def stop(self) -> None:
        """읽기 중지"""
        pass
        
    @abstractmethod
    def is_active(self) -> bool:
        """활성 상태 확인"""
        pass


class AudioProcessor(ABC):
    """
    오디오 데이터를 처리하는 인터페이스
    """
    
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> dict[str, Any]:
        """
        오디오 데이터 처리
        
        Args:
            audio_data: 오디오 데이터 (numpy array)
            
        Returns:
            처리 결과 딕셔너리
        """
        pass
