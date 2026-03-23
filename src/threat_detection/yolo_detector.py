"""src/threat_detection/yolo_detector.py — YOLOv8 기반 인물 감지 및 ID 추적."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class PersonDetection:
    """단일 인물 감지 결과.

    Attributes:
        person_id: 프레임 간 추적되는 고유 ID.
        bbox: 정규화된 바운딩 박스 (x1, y1, x2, y2), 각 값 0~1.
        confidence: 감지 신뢰도.
    """

    person_id: int
    bbox: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        """바운딩 박스 중심 좌표 (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class YOLOPersonDetector:
    """YOLOv8으로 프레임에서 사람만 감지하고 프레임 간 ID를 추적합니다.

    Args:
        model: ultralytics YOLO 호환 객체. None이면 'yolov8n.pt'를 자동 로드합니다.
        confidence: 감지 신뢰도 임계값.
        match_threshold: 이전 프레임 ID 매칭 거리 임계값 (정규화 단위).
    """

    _PERSON_CLASS = 0

    def __init__(
        self,
        model: Any | None = None,
        confidence: float = 0.4,
        match_threshold: float = 0.3,
    ) -> None:
        if model is None:
            from ultralytics import YOLO  # noqa: PLC0415

            model = YOLO("yolov8n.pt")
        self._model = model
        self._conf = confidence
        self._match_threshold = match_threshold
        self._prev_centers: dict[int, tuple[float, float]] = {}
        self._next_id: int = 0

    def detect(self, frame: np.ndarray) -> list[PersonDetection]:
        """프레임에서 사람을 감지하고 PersonDetection 목록을 반환합니다.

        Args:
            frame: BGR 형식의 입력 프레임 (H x W x 3).

        Returns:
            감지된 PersonDetection 목록.
        """
        h, w = frame.shape[:2]
        raw = self._model(frame, classes=[self._PERSON_CLASS], conf=self._conf, verbose=False)

        candidates: list[tuple[tuple[float, float, float, float], float]] = []
        for result in raw:
            for box in result.boxes:
                if int(box.cls[0]) != self._PERSON_CLASS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (x1 / w, y1 / h, x2 / w, y2 / h)
                candidates.append((bbox, float(box.conf[0])))

        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b, _ in candidates]
        ids = self._match_ids(centers)

        return [
            PersonDetection(person_id=pid, bbox=bbox, confidence=conf)
            for pid, (bbox, conf) in zip(ids, candidates)
        ]

    def draw(self, frame: np.ndarray, detections: list[PersonDetection]) -> np.ndarray:
        """감지된 인물마다 바운딩 박스와 ID를 그린 프레임 복사본을 반환합니다.

        Args:
            frame: 원본 BGR 프레임.
            detections: detect()가 반환한 PersonDetection 목록.

        Returns:
            바운딩 박스가 그려진 BGR 프레임.
        """
        out = frame.copy()
        h, w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 0), 2)
            label = f"P{det.person_id} {det.confidence:.2f}"
            cv2.putText(
                out, label, (px1, max(py1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return out

    def reset(self) -> None:
        """ID 추적 상태를 초기화합니다."""
        self._prev_centers.clear()
        self._next_id = 0

    def _match_ids(self, centers: list[tuple[float, float]]) -> list[int]:
        """각 감지 중심 좌표에 프레임 간 일관된 ID를 할당합니다."""
        new_prev: dict[int, tuple[float, float]] = {}
        assigned: list[int] = []
        used: set[int] = set()

        for cx, cy in centers:
            best_id: int | None = None
            best_dist = float("inf")
            for pid, (px, py) in self._prev_centers.items():
                if pid in used:
                    continue
                d = math.dist((cx, cy), (px, py))
                if d < best_dist:
                    best_dist = d
                    best_id = pid

            if best_id is not None and best_dist < self._match_threshold:
                assigned.append(best_id)
                used.add(best_id)
            else:
                assigned.append(self._next_id)
                self._next_id += 1

        for pid, (cx, cy) in zip(assigned, centers):
            new_prev[pid] = (cx, cy)
        self._prev_centers = new_prev

        return assigned
