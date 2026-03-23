"""공통 픽스처 및 헬퍼."""
from __future__ import annotations

import numpy as np

from src.gesture_recognition.recognizer import Gesture, GestureResult
from src.hand_tracking.tracker import HandLandmarks, TrackingResult
from src.threat_detection.detector import ThreatLevel, ThreatResult
from src.threat_detection.yolo_detector import PersonDetection

# ---------------------------------------------------------------------------
# 랜드마크 빌더
# ---------------------------------------------------------------------------

def make_landmarks(
    thumb_extended: bool = False,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
) -> list[tuple[float, float, float]]:
    """21개 랜드마크를 손가락 펴짐 상태에 맞게 생성합니다.

    MediaPipe 규칙:
    - 엄지(tip=4, pip=3): tip.x > pip.x 이면 펴짐
    - 나머지(tip, pip): tip.y < pip.y 이면 펴짐 (화면 좌표, y↓)
    """
    lm: list[tuple[float, float, float]] = [(0.5, 0.5, 0.0)] * 21

    # 엄지 (인덱스 3, 4)
    if thumb_extended:
        lm[3] = (0.4, 0.5, 0.0)
        lm[4] = (0.6, 0.5, 0.0)  # tip.x > pip.x
    else:
        lm[3] = (0.6, 0.5, 0.0)
        lm[4] = (0.4, 0.5, 0.0)

    # 검지 (tip=8, pip=6)
    lm[6] = (0.5, 0.6, 0.0)
    lm[8] = (0.5, 0.4 if index_extended else 0.8, 0.0)

    # 중지 (tip=12, pip=10)
    lm[10] = (0.5, 0.6, 0.0)
    lm[12] = (0.5, 0.4 if middle_extended else 0.8, 0.0)

    # 약지 (tip=16, pip=14)
    lm[14] = (0.5, 0.6, 0.0)
    lm[16] = (0.5, 0.4 if ring_extended else 0.8, 0.0)

    # 소지 (tip=20, pip=18)
    lm[18] = (0.5, 0.6, 0.0)
    lm[20] = (0.5, 0.4 if pinky_extended else 0.8, 0.0)

    return lm


def make_hand(
    handedness: str = "Right",
    confidence: float = 0.9,
    **finger_kwargs: bool,
) -> HandLandmarks:
    return HandLandmarks(
        landmarks=make_landmarks(**finger_kwargs),
        handedness=handedness,
        confidence=confidence,
    )


def make_tracking_result(*hands: HandLandmarks, frame_index: int = 1) -> TrackingResult:
    return TrackingResult(hands=list(hands), frame_index=frame_index)


def make_gesture_result(
    gesture: Gesture,
    confidence: float = 0.9,
    handedness: str = "Right",
) -> GestureResult:
    return GestureResult(gesture=gesture, confidence=confidence, handedness=handedness)


def make_threat_result(
    level: ThreatLevel = ThreatLevel.NONE,
    score: float = 0.0,
    reasons: list[str] | None = None,
) -> ThreatResult:
    return ThreatResult(level=level, score=score, reasons=reasons or [])


def blank_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_person(
    person_id: int,
    cx: float,
    cy: float,
    confidence: float = 0.9,
    half_w: float = 0.05,
    half_h: float = 0.1,
) -> PersonDetection:
    """중심 좌표 (cx, cy)를 기반으로 PersonDetection을 생성합니다."""
    return PersonDetection(
        person_id=person_id,
        bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        confidence=confidence,
    )
