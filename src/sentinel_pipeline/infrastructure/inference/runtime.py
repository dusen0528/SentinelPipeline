# -*- coding: utf-8 -*-
"""
AI 모델 추론 런타임.

ONNX Runtime 또는 PyTorch를 사용한 모델 추론을 담당합니다.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sentinel_pipeline.common.errors import ErrorCode, InferenceError
from sentinel_pipeline.common.logging import get_logger


# 타입 정의
TensorType = NDArray[np.float32]


class RuntimeType(str, Enum):
    """추론 런타임 유형."""
    ONNX = "onnx"
    PYTORCH = "pytorch"


class InferenceRuntime(ABC):
    """
    추론 런타임 추상 클래스.
    
    ONNX Runtime과 PyTorch 런타임의 공통 인터페이스를 정의합니다.
    """
    
    @abstractmethod
    def load_model(self, model_path: str | Path) -> None:
        """모델을 로드합니다."""
        ...
    
    @abstractmethod
    def infer(self, inputs: dict[str, TensorType]) -> dict[str, TensorType]:
        """추론을 수행합니다."""
        ...
    
    @abstractmethod
    def get_input_info(self) -> list[dict[str, Any]]:
        """입력 정보를 반환합니다."""
        ...
    
    @abstractmethod
    def get_output_info(self) -> list[dict[str, Any]]:
        """출력 정보를 반환합니다."""
        ...
    
    @abstractmethod
    def release(self) -> None:
        """리소스를 해제합니다."""
        ...


class ONNXRuntime(InferenceRuntime):
    """
    ONNX Runtime 추론 엔진.
    
    ONNX 모델을 로드하고 추론을 수행합니다.
    GPU가 사용 가능하면 자동으로 GPU를 사용합니다.
    """
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = True,
        num_threads: int = 4,
    ):
        """
        ONNXRuntime 초기화.
        
        Args:
            model_name: 모델 이름 (로깅용)
            use_gpu: GPU 사용 여부 (기본 True)
            num_threads: CPU 스레드 수 (기본 4)
        """
        self._model_name = model_name
        self._use_gpu = use_gpu
        self._num_threads = num_threads
        
        self._session = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._loaded = False
        self._lock = threading.Lock()
        
        # 통계
        self._inference_count = 0
        self._total_latency_ms = 0.0
        
        self._logger = get_logger(__name__, module_name=model_name)
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 상태."""
        return self._loaded
    
    @property
    def avg_latency_ms(self) -> float:
        """평균 추론 지연 시간 (밀리초)."""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count
    
    def load_model(self, model_path: str | Path) -> None:
        """
        ONNX 모델을 로드합니다.
        
        Args:
            model_path: 모델 파일 경로 (.onnx)
            
        Raises:
            InferenceError: 모델 로드 실패 시
        """
        import onnxruntime as ort
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise InferenceError(
                ErrorCode.INFERENCE_MODEL_NOT_FOUND,
                f"모델 파일을 찾을 수 없습니다: {model_path}",
                model_name=self._model_name,
            )
        
        self._logger.info("ONNX 모델 로드 시작", model_path=str(model_path))
        
        with self._lock:
            try:
                # 세션 옵션 설정
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = self._num_threads
                sess_options.inter_op_num_threads = self._num_threads
                sess_options.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                
                # Provider 설정 (GPU 우선)
                providers = []
                if self._use_gpu:
                    # CUDA 사용 가능 여부 확인
                    available_providers = ort.get_available_providers()
                    if "CUDAExecutionProvider" in available_providers:
                        providers.append("CUDAExecutionProvider")
                        self._logger.info("CUDA 런타임 사용")
                    elif "TensorrtExecutionProvider" in available_providers:
                        providers.append("TensorrtExecutionProvider")
                        self._logger.info("TensorRT 런타임 사용")
                
                providers.append("CPUExecutionProvider")
                
                # 세션 생성
                self._session = ort.InferenceSession(
                    str(model_path),
                    sess_options,
                    providers=providers,
                )
                
                # 입출력 이름 저장
                self._input_names = [inp.name for inp in self._session.get_inputs()]
                self._output_names = [out.name for out in self._session.get_outputs()]
                
                self._loaded = True
                self._logger.info(
                    "ONNX 모델 로드 완료",
                    inputs=self._input_names,
                    outputs=self._output_names,
                    provider=self._session.get_providers()[0],
                )
                
            except Exception as e:
                self._logger.error("ONNX 모델 로드 실패", error=str(e))
                raise InferenceError(
                    ErrorCode.INFERENCE_LOAD_FAILED,
                    f"ONNX 모델 로드 실패: {e}",
                    model_name=self._model_name,
                    details={"error": str(e)},
                )
    
    def infer(self, inputs: dict[str, TensorType]) -> dict[str, TensorType]:
        """
        추론을 수행합니다.
        
        Args:
            inputs: 입력 텐서 딕셔너리 (이름 -> numpy 배열)
            
        Returns:
            출력 텐서 딕셔너리 (이름 -> numpy 배열)
            
        Raises:
            InferenceError: 추론 실패 시
        """
        if not self._loaded:
            raise InferenceError(
                ErrorCode.INFERENCE_FAILED,
                "모델이 로드되지 않았습니다",
                model_name=self._model_name,
            )
        
        start_time = time.perf_counter()
        
        try:
            # 추론 실행
            outputs = self._session.run(self._output_names, inputs)  # type: ignore
            
            # 결과를 딕셔너리로 변환
            result = dict(zip(self._output_names, outputs))
            
            # 통계 업데이트
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._inference_count += 1
            self._total_latency_ms += latency_ms
            
            return result
            
        except Exception as e:
            self._logger.error("추론 실패", error=str(e))
            raise InferenceError(
                ErrorCode.INFERENCE_FAILED,
                f"추론 실패: {e}",
                model_name=self._model_name,
                details={"error": str(e)},
            )
    
    def get_input_info(self) -> list[dict[str, Any]]:
        """입력 정보 반환."""
        if not self._loaded:
            return []
        
        result = []
        for inp in self._session.get_inputs():  # type: ignore
            result.append({
                "name": inp.name,
                "shape": inp.shape,
                "type": inp.type,
            })
        return result
    
    def get_output_info(self) -> list[dict[str, Any]]:
        """출력 정보 반환."""
        if not self._loaded:
            return []
        
        result = []
        for out in self._session.get_outputs():  # type: ignore
            result.append({
                "name": out.name,
                "shape": out.shape,
                "type": out.type,
            })
        return result
    
    def release(self) -> None:
        """리소스 해제."""
        with self._lock:
            if self._session is not None:
                self._session = None
            self._loaded = False
            self._logger.info("ONNX 런타임 리소스 해제")
    
    def get_stats(self) -> dict:
        """통계 정보 반환."""
        return {
            "model_name": self._model_name,
            "loaded": self._loaded,
            "inference_count": self._inference_count,
            "avg_latency_ms": self.avg_latency_ms,
            "use_gpu": self._use_gpu,
        }


class PyTorchRuntime(InferenceRuntime):
    """
    PyTorch 추론 런타임.
    
    PyTorch 모델을 로드하고 추론을 수행합니다.
    TorchScript (.pt) 또는 일반 모델을 지원합니다.
    """
    
    def __init__(
        self,
        model_name: str,
        use_gpu: bool = True,
        device: str | None = None,
    ):
        """
        PyTorchRuntime 초기화.
        
        Args:
            model_name: 모델 이름 (로깅용)
            use_gpu: GPU 사용 여부 (기본 True)
            device: 디바이스 지정 (기본 None - 자동 선택)
        """
        self._model_name = model_name
        self._use_gpu = use_gpu
        self._device_str = device
        
        self._model = None
        self._device = None
        self._loaded = False
        self._lock = threading.Lock()
        
        # 통계
        self._inference_count = 0
        self._total_latency_ms = 0.0
        
        self._logger = get_logger(__name__, module_name=model_name)
    
    @property
    def is_loaded(self) -> bool:
        """모델 로드 상태."""
        return self._loaded
    
    @property
    def avg_latency_ms(self) -> float:
        """평균 추론 지연 시간 (밀리초)."""
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count
    
    def load_model(self, model_path: str | Path) -> None:
        """
        PyTorch 모델을 로드합니다.
        
        Args:
            model_path: 모델 파일 경로 (.pt, .pth)
            
        Raises:
            InferenceError: 모델 로드 실패 시
        """
        import torch
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise InferenceError(
                ErrorCode.INFERENCE_MODEL_NOT_FOUND,
                f"모델 파일을 찾을 수 없습니다: {model_path}",
                model_name=self._model_name,
            )
        
        self._logger.info("PyTorch 모델 로드 시작", model_path=str(model_path))
        
        with self._lock:
            try:
                # 디바이스 설정
                if self._device_str:
                    self._device = torch.device(self._device_str)
                elif self._use_gpu and torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    self._logger.info("CUDA 디바이스 사용")
                else:
                    self._device = torch.device("cpu")
                    self._logger.info("CPU 디바이스 사용")
                
                # 모델 로드 (TorchScript 가정)
                self._model = torch.jit.load(str(model_path), map_location=self._device)
                self._model.eval()  # 추론 모드
                
                self._loaded = True
                self._logger.info("PyTorch 모델 로드 완료", device=str(self._device))
                
            except Exception as e:
                self._logger.error("PyTorch 모델 로드 실패", error=str(e))
                raise InferenceError(
                    ErrorCode.INFERENCE_LOAD_FAILED,
                    f"PyTorch 모델 로드 실패: {e}",
                    model_name=self._model_name,
                    details={"error": str(e)},
                )
    
    def infer(self, inputs: dict[str, TensorType]) -> dict[str, TensorType]:
        """
        추론을 수행합니다.
        
        Args:
            inputs: 입력 텐서 딕셔너리 (이름 -> numpy 배열)
            
        Returns:
            출력 텐서 딕셔너리 (이름 -> numpy 배열)
            
        Raises:
            InferenceError: 추론 실패 시
        """
        import torch
        
        if not self._loaded:
            raise InferenceError(
                ErrorCode.INFERENCE_FAILED,
                "모델이 로드되지 않았습니다",
                model_name=self._model_name,
            )
        
        start_time = time.perf_counter()
        
        try:
            # numpy를 torch tensor로 변환
            torch_inputs = []
            for name in sorted(inputs.keys()):
                tensor = torch.from_numpy(inputs[name]).to(self._device)
                torch_inputs.append(tensor)
            
            # 추론 실행
            with torch.no_grad():
                if len(torch_inputs) == 1:
                    outputs = self._model(torch_inputs[0])  # type: ignore
                else:
                    outputs = self._model(*torch_inputs)  # type: ignore
            
            # 결과를 딕셔너리로 변환
            if isinstance(outputs, torch.Tensor):
                result = {"output": outputs.cpu().numpy()}
            elif isinstance(outputs, (tuple, list)):
                result = {f"output_{i}": out.cpu().numpy() for i, out in enumerate(outputs)}
            else:
                result = {"output": outputs}
            
            # 통계 업데이트
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._inference_count += 1
            self._total_latency_ms += latency_ms
            
            return result
            
        except Exception as e:
            self._logger.error("추론 실패", error=str(e))
            raise InferenceError(
                ErrorCode.INFERENCE_FAILED,
                f"추론 실패: {e}",
                model_name=self._model_name,
                details={"error": str(e)},
            )
    
    def get_input_info(self) -> list[dict[str, Any]]:
        """입력 정보 반환 (TorchScript에서는 제한적)."""
        return [{"name": "input", "info": "TorchScript 모델"}]
    
    def get_output_info(self) -> list[dict[str, Any]]:
        """출력 정보 반환 (TorchScript에서는 제한적)."""
        return [{"name": "output", "info": "TorchScript 모델"}]
    
    def release(self) -> None:
        """리소스 해제."""
        import torch
        
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None
            
            # GPU 메모리 해제
            if self._device is not None and self._device.type == "cuda":
                torch.cuda.empty_cache()
            
            self._loaded = False
            self._logger.info("PyTorch 런타임 리소스 해제")
    
    def get_stats(self) -> dict:
        """통계 정보 반환."""
        return {
            "model_name": self._model_name,
            "loaded": self._loaded,
            "inference_count": self._inference_count,
            "avg_latency_ms": self.avg_latency_ms,
            "device": str(self._device) if self._device else None,
        }


def create_runtime(
    runtime_type: RuntimeType,
    model_name: str,
    use_gpu: bool = True,
    **kwargs,
) -> InferenceRuntime:
    """
    추론 런타임을 생성합니다.
    
    Args:
        runtime_type: 런타임 유형 (ONNX 또는 PYTORCH)
        model_name: 모델 이름
        use_gpu: GPU 사용 여부
        **kwargs: 추가 옵션
        
    Returns:
        InferenceRuntime 인스턴스
    """
    if runtime_type == RuntimeType.ONNX:
        return ONNXRuntime(model_name, use_gpu=use_gpu, **kwargs)
    elif runtime_type == RuntimeType.PYTORCH:
        return PyTorchRuntime(model_name, use_gpu=use_gpu, **kwargs)
    else:
        raise ValueError(f"지원하지 않는 런타임 유형: {runtime_type}")

