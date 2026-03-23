"""src/protection/protected_tracker.py — 보호 대상 등록 및 YOLO ID 기반 추적."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.threat_detection.yolo_detector import PersonDetection

logger = logging.getLogger(__name__)

_REIDENTIFY_DISTANCE: float = 0.2   # 정규화 좌표 기준 재연결 허용 거리
_DISAPPEAR_WINDOW: float = 5.0      # 이탈 후 재연결 대기 시간(초)


@dataclass
class ProtectedPersonStatus:
    """보호 대상의 현재 상태.

    Attributes:
        person_id: 보호 대상 YOLO 트래킹 ID.
        is_in_frame: 현재 프레임에서 보호 대상이 감지됐는지 여부.
        bbox: 마지막으로 감지된 정규화 바운딩 박스 (x1,y1,x2,y2). 미감지 시 None.
        just_disappeared: 직전 프레임에는 있었는데 이번 프레임에서 사라졌으면 True.
    """

    person_id: int
    is_in_frame: bool
    bbox: tuple[float, float, float, float] | None
    just_disappeared: bool


class ProtectedPersonTracker:
    """V사인 SOS가 확정된 인물을 "보호 대상"으로 등록하고 YOLO 트래킹 ID로 추적합니다.

    SOS 확정 이후 V사인을 그만둬도 person_id가 일치하는 한 계속 추적하며,
    화면에서 사라질 경우 ``just_disappeared=True`` 인 status를 반환합니다.

    사용 흐름::

        tracker = ProtectedPersonTracker()

        # SOS 확정 시 (파이프라인 측에서 호출)
        tracker.register(person_id=2)

        # 매 프레임마다
        status = tracker.update(persons)
        if status and status.just_disappeared:
            # "보호 대상 이탈" 경고 처리
            ...
        if status:
            frame = tracker.draw(frame, status)
    """

    def __init__(self) -> None:
        self._protected_id: int | None = None
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._prev_in_frame: bool = False
        self._disappeared_at: float | None = None  # 이탈 첫 프레임 monotonic 시각

    # ------------------------------------------------------------------
    # 속성
    # ------------------------------------------------------------------

    @property
    def protected_id(self) -> int | None:
        """등록된 보호 대상 person_id. 없으면 None."""
        return self._protected_id

    @property
    def is_registered(self) -> bool:
        """보호 대상이 등록돼 있는지 여부."""
        return self._protected_id is not None

    # ------------------------------------------------------------------
    # 등록 / 해제
    # ------------------------------------------------------------------

    def register(self, person_id: int) -> None:
        """지정된 person_id를 보호 대상으로 등록합니다.

        이미 다른 대상이 등록돼 있으면 덮어씁니다.

        Args:
            person_id: 보호 대상으로 지정할 YOLO 트래킹 ID.
        """
        self._protected_id = person_id
        self._prev_in_frame = False
        self._last_bbox = None
        self._disappeared_at = None
        logger.info("보호 대상 등록: P%d", person_id)

    def clear(self) -> None:
        """보호 대상 등록을 해제하고 내부 상태를 초기화합니다."""
        prev = self._protected_id
        self._protected_id = None
        self._last_bbox = None
        self._prev_in_frame = False
        self._disappeared_at = None
        if prev is not None:
            logger.info("보호 대상 해제: P%d", prev)

    # ------------------------------------------------------------------
    # 매 프레임 업데이트
    # ------------------------------------------------------------------

    def update(self, persons: list[PersonDetection]) -> ProtectedPersonStatus | None:
        """현재 프레임의 인물 목록으로 보호 대상 상태를 업데이트합니다.

        보호 대상이 등록되지 않은 경우 None을 반환합니다.
        보호 대상이 화면에서 사라진 첫 프레임에는 ``just_disappeared=True``
        인 status를 반환합니다.

        Args:
            persons: 현재 프레임의 PersonDetection 목록.

        Returns:
            보호 대상이 등록된 경우 ProtectedPersonStatus, 미등록이면 None.
        """
        if self._protected_id is None:
            return None

        found = next(
            (p for p in persons if p.person_id == self._protected_id), None
        )

        # ID로 못 찾은 경우: 이전에 이미 이탈한 상태라면 근접 재연결 시도
        if found is None and self._disappeared_at is not None:
            found = self._try_reidentify(persons)

        is_in_frame = found is not None
        just_disappeared = self._prev_in_frame and not is_in_frame

        if found is not None:
            self._last_bbox = found.bbox
            self._disappeared_at = None
        elif just_disappeared:
            self._disappeared_at = time.monotonic()
            logger.warning("보호 대상 이탈: P%d", self._protected_id)

        self._prev_in_frame = is_in_frame

        return ProtectedPersonStatus(
            person_id=self._protected_id,
            is_in_frame=is_in_frame,
            bbox=self._last_bbox,
            just_disappeared=just_disappeared,
        )

    # ------------------------------------------------------------------
    # 근접 재연결
    # ------------------------------------------------------------------

    def _try_reidentify(
        self, persons: list[PersonDetection]
    ) -> PersonDetection | None:
        """이탈 후 근접 위치에 재등장한 인물로 보호 대상 ID를 교체합니다.

        이탈 후 ``_DISAPPEAR_WINDOW`` 초 이내에 마지막 bbox 중심으로부터
        ``_REIDENTIFY_DISTANCE`` 이내 인물이 있으면 그 인물로 재연결합니다.

        Args:
            persons: 현재 프레임의 PersonDetection 목록.

        Returns:
            재연결된 PersonDetection. 해당 인물이 없으면 None.
        """
        if not persons or self._last_bbox is None:
            return None
        if (
            self._disappeared_at is not None
            and time.monotonic() - self._disappeared_at > _DISAPPEAR_WINDOW
        ):
            return None

        lx = (self._last_bbox[0] + self._last_bbox[2]) / 2
        ly = (self._last_bbox[1] + self._last_bbox[3]) / 2

        best_person: PersonDetection | None = None
        best_dist: float = _REIDENTIFY_DISTANCE
        for p in persons:
            cx, cy = p.center
            dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_person = p

        if best_person is not None:
            old_id = self._protected_id
            self._protected_id = best_person.person_id
            logger.info("보호 대상 재연결: P%d → P%d", old_id, best_person.person_id)

        return best_person

    # ------------------------------------------------------------------
    # 시각화
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        status: ProtectedPersonStatus,
    ) -> np.ndarray:
        """보호 대상 위치에 초록색 박스와 레이블을 그립니다.

        보호 대상이 화면에 없으면 프레임을 수정하지 않습니다.

        Args:
            frame: 원본 BGR 프레임.
            status: update()가 반환한 ProtectedPersonStatus.

        Returns:
            박스가 그려진 BGR 프레임 복사본.
        """
        out = frame.copy()
        if not status.is_in_frame or status.bbox is None:
            return out

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = status.bbox
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)

        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 0), 3)
        label = f"보호대상 P{status.person_id}"
        cv2.putText(
            out, label, (px1, max(py1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        return out
