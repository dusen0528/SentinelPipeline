"""
얼굴/사람 블러 처리 모듈

YOLO를 사용하여 얼굴/사람을 감지하고 블러 처리합니다.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

from sentinel_pipeline.domain.interfaces.module import BaseModule
from sentinel_pipeline.common.logging import get_logger

logger = get_logger(__name__)

# 프레임 타입
FrameType = NDArray[np.uint8]
MetadataType = dict[str, Any]


class FaceTracker:
    """얼굴 위치 추적 (블러 안정화)"""

    def __init__(self, max_age: int = 50, smoothing: float = 0.5):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.smoothing = smoothing

    def tick(self) -> list:
        """추론을 건너뛴 프레임에서도 age를 증가시켜 잔상 최소화."""
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]
        return [track["box"] for track in self.tracks.values()]

    def update(self, detections: list) -> list:
        """새 감지 결과로 트랙 업데이트"""
        # 모든 트랙의 age 증가
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]["age"] += 1
            if self.tracks[track_id]["age"] > self.max_age:
                del self.tracks[track_id]

        # 새 감지와 기존 트랙 매칭
        used_tracks = set()
        for det in detections:
            best_iou = 0
            best_track_id = None

            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                iou = self._calc_iou(det, track["box"])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                old_box = self.tracks[best_track_id]["box"]
                new_box = [
                    int(old_box[i] * self.smoothing + det[i] * (1 - self.smoothing))
                    for i in range(4)
                ]
                self.tracks[best_track_id] = {"box": new_box, "age": 0}
                used_tracks.add(best_track_id)
            else:
                self.tracks[self.next_id] = {"box": list(det), "age": 0}
                self.next_id += 1

        return [track["box"] for track in self.tracks.values()]

    def _calc_iou(self, box1, box2) -> float:
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def reset(self):
        """트래커 초기화"""
        self.tracks = {}
        self.next_id = 0


class FaceBlurModule(BaseModule):
    """
    얼굴/사람 블러 처리 모듈

    YOLO를 사용하여 얼굴/사람을 감지하고 블러 처리합니다.
    프레임 스킵 최적화를 통해 성능을 향상시킵니다.
    """

    name = "FaceBlurModule"
    priority = 200  # 후처리 모듈이므로 다른 감지 모듈 이후 실행
    timeout_ms = 100

    # 클래스 변수: 모델 싱글톤
    _model: YOLO | None = None
    _model_lock = threading.Lock()

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)

        # 설정 파싱
        self.model_path = self.options.get("model_path", "models/yolov8n-face.pt")
        self.use_person_detection = self.options.get("use_person_detection", False)
        self.confidence_threshold = float(self.options.get("confidence_threshold", 0.15))
        self.anonymize_method = self.options.get("anonymize_method", "pixelate")
        self.blur_strength = int(self.options.get("blur_strength", 31))
        self.pixelate_size = int(self.options.get("pixelate_size", 10))
        self.imgsz = int(self.options.get("imgsz", 320))
        self.skip_interval = int(self.options.get("skip_interval", 3))
        self.max_age = int(self.options.get("max_age", 50))
        self.smoothing = float(self.options.get("smoothing", 0.5))
        self.iou_threshold = float(self.options.get("iou_threshold", 0.35))
        self.box_expand_ratio = float(self.options.get("box_expand_ratio", 0.15))

        # 인스턴스 변수
        self._frame_count = 0
        self._last_boxes: list[tuple[int, int, int, int]] = []
        self._face_tracker = FaceTracker(max_age=self.max_age, smoothing=self.smoothing)

        # 모델 로드
        self._ensure_model_loaded()

    @classmethod
    def _ensure_model_loaded(cls) -> None:
        """모델 싱글톤 로드"""
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    # 기본 모델 경로 (인스턴스별로 다를 수 있지만, 첫 로드 시 사용)
                    default_path = Path("models/yolov8n-face.pt")
                    if default_path.exists():
                        logger.info(f"YOLO 모델 로드: {default_path}")
                        cls._model = YOLO(str(default_path))
                    else:
                        # 폴백: yolov8n.pt
                        fallback_path = Path("models/yolov8n.pt")
                        if fallback_path.exists():
                            logger.info(f"YOLO 모델 로드 (폴백): {fallback_path}")
                            cls._model = YOLO(str(fallback_path))
                        else:
                            raise FileNotFoundError(
                                f"모델 파일을 찾을 수 없습니다: {default_path} 또는 {fallback_path}"
                            )

    def _load_model_for_instance(self) -> YOLO:
        """인스턴스별 모델 경로로 모델 로드"""
        model_path = Path(self.model_path)
        if not model_path.is_absolute():
            # 상대 경로인 경우 프로젝트 루트 기준
            model_path = Path(".") / model_path

        # yolov8n-face.pt 우선 확인 (얼굴 전용 모드일 때) — 로그는 한 번만 출력
        if not self.use_person_detection:
            face_model_path = model_path.parent / "yolov8n-face.pt"
            if face_model_path.exists():
                if not getattr(self, "_face_model_logged", False):
                    logger.info(f"얼굴 전용 모델 발견: {face_model_path}")
                    self._face_model_logged = True
                model_path = face_model_path

        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        # 싱글톤 로드 (경로가 같으면 재사용)
        with self._model_lock:
            if self._model is None:
                logger.info(f"YOLO 모델 로드: {model_path}")
                self._model = YOLO(str(model_path))

        return self._model

    @staticmethod
    def validate_options(options: dict[str, Any]) -> None:
        """옵션 검증"""
        model_path = options.get("model_path", "models/yolov8n-face.pt")
        if not isinstance(model_path, str):
            raise ValueError("FaceBlurModule.model_path는 문자열이어야 합니다")

        # 모델 파일 존재 확인 (상대 경로 허용)
        path = Path(model_path)
        if not path.is_absolute():
            path = Path(".") / path

        # yolov8n-face.pt 우선 확인
        if not path.exists():
            face_path = path.parent / "yolov8n-face.pt"
            if not face_path.exists():
                # 검증 시점에는 파일이 없을 수 있으므로 경고만
                logger.warning(f"모델 파일을 찾을 수 없습니다: {model_path} (런타임에 확인됨)")

        confidence = options.get("confidence_threshold", 0.15)
        if not isinstance(confidence, (int, float)) or not (0.01 <= confidence <= 1.0):
            raise ValueError("FaceBlurModule.confidence_threshold는 0.01~1.0 범위여야 합니다")

        imgsz = options.get("imgsz", 320)
        if imgsz not in [320, 480, 640]:
            raise ValueError("FaceBlurModule.imgsz는 320, 480, 640 중 하나여야 합니다")

    def process_frame(
        self,
        frame: FrameType,
        metadata: MetadataType,
    ) -> tuple[FrameType, list, MetadataType]:
        """
        프레임을 처리합니다.

        YOLO 추론을 통해 얼굴/사람을 감지하고 블러 처리합니다.
        프레임 스킵 최적화를 통해 성능을 향상시킵니다.

        Args:
            frame: 입력 프레임 (BGR 형식)
            metadata: 프레임 메타데이터

        Returns:
            (처리된 프레임, 이벤트 목록, 업데이트된 메타데이터)
        """
        try:
            # 모델 로드 확인
            model = self._load_model_for_instance()

            # 프레임 스킵 최적화: N프레임마다 1번만 추론
            self._frame_count += 1
            should_run_inference = (self._frame_count - 1) % self.skip_interval == 0

            if should_run_inference:
                # YOLO 추론
                model_kwargs = {
                    "verbose": False,
                    "imgsz": self.imgsz,
                    "conf": self.confidence_threshold,
                    "iou": self.iou_threshold,
                }

                if self.use_person_detection:
                    model_kwargs["classes"] = [0]  # person 클래스만 (COCO 데이터셋 기준)

                results = model(frame, **model_kwargs)

                # 박스 추출 및 후처리
                detections = self._extract_boxes(results, frame.shape)

                # 트래커로 안정화
                stable_boxes = self._face_tracker.update(detections)
                self._last_boxes = stable_boxes
            else:
                # 추론 안 하는 프레임은 이전 박스를 age 증가시켜 사용
                stable_boxes = self._face_tracker.tick()
                self._last_boxes = stable_boxes

            # 블러 처리
            processed_frame = frame.copy()
            for box in stable_boxes:
                processed_frame = self._apply_blur(processed_frame, box)

            # 이벤트 없음 (필요 시 추가 가능)
            return processed_frame, [], metadata

        except Exception as e:
            logger.error(f"프레임 처리 오류: {e}", error=str(e))
            # 오류 발생 시 원본 프레임 반환
            return frame, [], metadata

    def _extract_boxes(
        self, results: Any, frame_shape: tuple[int, int, int]
    ) -> list[tuple[int, int, int, int]]:
        """
        YOLO 결과에서 박스 추출 및 후처리

        Args:
            results: YOLO 추론 결과
            frame_shape: 프레임 크기 (height, width, channels)

        Returns:
            박스 리스트 [(x1, y1, x2, y2), ...]
        """
        detections = []
        h, w = frame_shape[:2]

        if not results or len(results) == 0:
            return detections

        for box in results[0].boxes:
            # 클래스 확인
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = results[0].names[cls_id] if hasattr(results[0], "names") else None

            # person 감지 모드인 경우 person 클래스만 필터링
            if self.use_person_detection:
                if cls_id != 0 and cls_name != "person":
                    continue

            # 박스 좌표 추출
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            # 박스 크기 계산
            box_width = x2 - x1
            box_height = y2 - y1

            # 박스 확장 (가로/세로 분리)
            expand_x_ratio = self.box_expand_ratio
            expand_y_ratio = self.box_expand_ratio

            expand_x = int(box_width * expand_x_ratio)
            expand_y = int(box_height * expand_y_ratio)

            # 박스 확장 적용
            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(w, x2 + expand_x)
            y2 = min(h, y2 + expand_y)

            # 머리 부분만 자르기: 상단 60% 영역만 사용
            head_ratio = 0.6
            new_height = int((y2 - y1) * head_ratio)
            y2 = y1 + new_height

            # 세로 길이에 맞춰서 정사각형으로 만들기
            height = y2 - y1
            center_x = (x1 + x2) // 2
            half_size = height // 2

            x1 = max(0, center_x - half_size)
            x2 = min(w, center_x + half_size)

            # 정사각형이 프레임을 벗어나면 조정
            if x2 - x1 < height:
                if x2 - x1 > 0:
                    y2 = y1 + (x2 - x1)

            detections.append((x1, y1, x2, y2))

        return detections

    def _apply_blur(self, frame: FrameType, box: tuple[int, int, int, int]) -> FrameType:
        """
        블러 처리 적용

        Args:
            frame: 입력 프레임
            box: 박스 좌표 (x1, y1, x2, y2)

        Returns:
            처리된 프레임
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box

        # 경계 체크
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return frame

        # 얼굴 영역 추출
        face_region = frame[y1:y2, x1:x2]
        face_h, face_w = face_region.shape[:2]

        if face_h == 0 or face_w == 0:
            return frame

        # 블러 방법에 따라 처리
        if self.anonymize_method == "blur":
            # 가우시안 블러 (느림, 부드러움)
            blur_strength = self.blur_strength
            if blur_strength % 2 == 0:
                blur_strength += 1
            processed = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)

        elif self.anonymize_method == "pixelate":
            # 픽셀화 (빠름, 효율적)
            pixel_size = self.pixelate_size
            small_w = max(1, face_w // pixel_size)
            small_h = max(1, face_h // pixel_size)
            # 작게 축소
            small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            # 원래 크기로 확대
            processed = cv2.resize(small, (face_w, face_h), interpolation=cv2.INTER_NEAREST)

        elif self.anonymize_method == "mosaic":
            # 모자이크 (ratio=0.05로 축소 후 확대, 매우 빠름)
            ratio = 0.05  # 1/20로 축소 (정보 날리기)
            small_w = max(1, int(face_w * ratio))
            small_h = max(1, int(face_h * ratio))
            # 아주 작게 축소
            small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            # 원래 크기로 다시 확대 (모자이크 효과)
            processed = cv2.resize(small, (face_w, face_h), interpolation=cv2.INTER_NEAREST)

        elif self.anonymize_method == "black":
            # 검은 박스 (가장 빠름)
            processed = np.zeros_like(face_region)

        elif self.anonymize_method == "solid":
            # 단색 채우기 (빠름)
            avg_color = np.mean(face_region, axis=(0, 1)).astype(np.uint8)
            processed = np.full_like(face_region, avg_color)

        else:
            # 기본값: 픽셀화
            pixel_size = self.pixelate_size
            small_w = max(1, face_w // pixel_size)
            small_h = max(1, face_h // pixel_size)
            small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            processed = cv2.resize(small, (face_w, face_h), interpolation=cv2.INTER_NEAREST)

        # 처리된 영역을 원본 프레임에 적용
        frame[y1:y2, x1:x2] = processed
        return frame
