"""src/gesture_recognition/recognizer.py — 제스처 분류 및 SOS 감지."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from src.hand_tracking.tracker import HandLandmarks, TrackingResult


class Gesture(Enum):
    """인식 가능한 손 제스처."""

    UNKNOWN = auto()
    OPEN_PALM = auto()
    FIST = auto()
    POINTING = auto()
    THUMBS_UP = auto()
    THUMBS_DOWN = auto()
    V_SIGN = auto()  # 검지 + 중지 펴짐, 나머지 접힘 → SOS 트리거


@dataclass
class GestureResult:
    """단일 손의 제스처 인식 결과."""

    gesture: Gesture
    confidence: float
    handedness: str


@dataclass
class SOSDetectionResult:
    """SOS 감지 결과.

    Attributes:
        is_detected: SOS 최종 확정 여부 (hold_seconds 이상 유지).
        is_pending: 1차 감지 상태 (confidence >= threshold, 유지 시간 미달).
        confidence: 현재 프레임에서의 V사인 확신도.
        held_duration: V사인 연속 유지 시간 (초).
    """

    is_detected: bool
    is_pending: bool
    confidence: float
    held_duration: float


class BaseSOSDetector(ABC):
    """SOS 감지기 인터페이스.

    규칙 기반 구현(RuleBasedSOSDetector)을 기본으로 제공하며,
    LSTM 등 다른 구현으로 교체할 때 이 인터페이스를 상속합니다.
    """

    @abstractmethod
    def update(
        self,
        gesture_results: list[GestureResult],
        timestamp: float | None = None,
    ) -> SOSDetectionResult:
        """새 프레임의 제스처 결과로 SOS 상태를 업데이트합니다.

        Args:
            gesture_results: 현재 프레임의 GestureResult 목록.
            timestamp: 현재 시각(monotonic). None이면 time.monotonic() 사용.

        Returns:
            현재 SOS 감지 상태.
        """

    @abstractmethod
    def reset(self) -> None:
        """내부 상태(타이머 등)를 초기화합니다."""


class RuleBasedSOSDetector(BaseSOSDetector):
    """규칙 기반 SOS 감지기.

    V사인(검지 + 중지 펴짐, 엄지·약지·소지 접힘)이 ``hold_seconds`` 이상
    연속 유지되면 SOS로 최종 확정합니다.

    2단계 확인:
    1. confidence >= ``confidence_threshold`` → is_pending = True (1차 감지)
    2. 해당 상태가 ``hold_seconds`` 이상 지속 → is_detected = True (최종 확정)

    V사인이 한 프레임이라도 끊기면 타이머를 리셋합니다.

    Args:
        hold_seconds: SOS 확정에 필요한 최소 연속 유지 시간 (기본 3.0초).
        confidence_threshold: 1차 감지 기준 확신도 (기본 0.85).
    """

    def __init__(
        self,
        hold_seconds: float = 3.0,
        confidence_threshold: float = 0.85,
    ) -> None:
        self._hold_seconds = hold_seconds
        self._threshold = confidence_threshold
        self._start_time: float | None = None

    def update(
        self,
        gesture_results: list[GestureResult],
        timestamp: float | None = None,
    ) -> SOSDetectionResult:
        """V사인 유지 시간을 누적하여 SOS 상태를 반환합니다.

        Args:
            gesture_results: 현재 프레임의 GestureResult 목록.
            timestamp: 현재 시각(monotonic). None이면 time.monotonic() 사용.

        Returns:
            SOSDetectionResult
        """
        now = timestamp if timestamp is not None else time.monotonic()
        best_confidence = self._best_v_sign_confidence(gesture_results)

        if best_confidence >= self._threshold:
            if self._start_time is None:
                self._start_time = now
            held = now - self._start_time
            is_detected = held >= self._hold_seconds
            return SOSDetectionResult(
                is_detected=is_detected,
                is_pending=not is_detected,
                confidence=best_confidence,
                held_duration=held,
            )

        # V사인 끊김 → 타이머 리셋
        self._start_time = None
        return SOSDetectionResult(
            is_detected=False,
            is_pending=False,
            confidence=best_confidence,
            held_duration=0.0,
        )

    def reset(self) -> None:
        """타이머와 상태를 초기화합니다."""
        self._start_time = None

    @staticmethod
    def _best_v_sign_confidence(gesture_results: list[GestureResult]) -> float:
        """V사인 GestureResult 중 최고 확신도를 반환합니다."""
        return max(
            (r.confidence for r in gesture_results if r.gesture == Gesture.V_SIGN),
            default=0.0,
        )


class GestureRecognizer:
    """손 랜드마크로부터 제스처를 분류합니다."""

    # MediaPipe hand landmark indices
    _FINGER_TIPS = [4, 8, 12, 16, 20]
    _FINGER_PIPS = [3, 6, 10, 14, 18]

    def __init__(self, confidence_threshold: float = 0.6) -> None:
        self._threshold = confidence_threshold

    def recognize(self, tracking_result: TrackingResult) -> list[GestureResult]:
        """TrackingResult를 받아 각 손의 GestureResult 목록을 반환합니다."""
        results: list[GestureResult] = []
        for hand in tracking_result.hands:
            gesture, confidence = self._classify(hand)
            if confidence >= self._threshold:
                results.append(
                    GestureResult(
                        gesture=gesture,
                        confidence=confidence,
                        handedness=hand.handedness,
                    )
                )
        return results

    def _classify(self, hand: HandLandmarks) -> tuple[Gesture, float]:
        """규칙 기반 제스처 분류. 모델 교체 시 이 메서드만 오버라이드합니다."""
        lm = hand.landmarks
        fingers_extended = self._fingers_extended(lm)
        thumb, index, middle, ring, pinky = fingers_extended
        extended_count = sum(fingers_extended)

        if extended_count == 5:
            return Gesture.OPEN_PALM, 0.9
        if extended_count == 0:
            return Gesture.FIST, 0.9
        # V사인: 검지 + 중지만 펴짐, 엄지 · 약지 · 소지는 접힘
        if index and middle and not thumb and not ring and not pinky:
            return Gesture.V_SIGN, 0.9
        # 검지만 펴짐 (포인팅)
        if index and not middle and not ring and not pinky:
            return Gesture.POINTING, 0.85
        # 엄지만 펴짐
        if thumb and not index and not middle and not ring and not pinky:
            return Gesture.THUMBS_UP, 0.8
        return Gesture.UNKNOWN, 0.5

    def _fingers_extended(self, lm: list[tuple[float, float, float]]) -> list[bool]:
        """각 손가락의 펴짐 여부를 반환합니다 (엄지, 검지, 중지, 약지, 소지)."""
        extended = []
        # 엄지: x 방향 비교 (오른손 기준)
        extended.append(lm[self._FINGER_TIPS[0]][0] > lm[self._FINGER_PIPS[0]][0])
        # 나머지 손가락: y 방향 비교 (tip이 pip보다 위)
        for tip, pip in zip(self._FINGER_TIPS[1:], self._FINGER_PIPS[1:]):
            extended.append(lm[tip][1] < lm[pip][1])
        return extended
