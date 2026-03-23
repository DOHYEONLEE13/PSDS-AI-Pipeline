"""src/threat_detection/detector.py — 제스처·인물 감지 기반 위협 판정."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from src.gesture_recognition.recognizer import Gesture, GestureResult
from src.threat_detection.approach_analyzer import ApproachAnalyzer, ApproachEvent
from src.threat_detection.yolo_detector import PersonDetection, YOLOPersonDetector


class ThreatLevel(Enum):
    """위협 수준 (0=정상, 1=의심, 2=위협, 3=긴급, 4=초긴급)."""

    NONE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class ThreatResult:
    """위협 판정 결과."""

    level: ThreatLevel
    score: float  # 0.0 ~ 1.0
    reasons: list[str] = field(default_factory=list)

    @property
    def is_threat(self) -> bool:
        return self.level not in (ThreatLevel.NONE, ThreatLevel.LOW)


# 위협 제스처 → 기본 점수 매핑
_THREAT_SCORE_MAP: dict[Gesture, float] = {
    Gesture.FIST: 0.7,
    Gesture.POINTING: 0.5,
    Gesture.UNKNOWN: 0.1,
    Gesture.OPEN_PALM: 0.0,
    Gesture.THUMBS_UP: 0.0,
    Gesture.THUMBS_DOWN: 0.0,
    Gesture.V_SIGN: 0.0,
}


class ThreatDetector:
    """제스처 결과를 바탕으로 위협 수준을 판정합니다."""

    def __init__(
        self,
        history_window: int = 10,
        high_threshold: float = 0.75,
        medium_threshold: float = 0.45,
        low_threshold: float = 0.2,
    ) -> None:
        self._window = history_window
        self._high = high_threshold
        self._medium = medium_threshold
        self._low = low_threshold
        self._score_history: list[float] = []

    def detect(self, gestures: list[GestureResult]) -> ThreatResult:
        """현재 프레임의 제스처 목록으로 위협 수준을 반환합니다."""
        frame_score = self._compute_frame_score(gestures)
        self._score_history.append(frame_score)
        if len(self._score_history) > self._window:
            self._score_history.pop(0)

        smoothed = float(np.mean(self._score_history))
        level, reasons = self._evaluate(smoothed, gestures)
        return ThreatResult(level=level, score=smoothed, reasons=reasons)

    def reset(self) -> None:
        """이력을 초기화합니다."""
        self._score_history.clear()

    def _compute_frame_score(self, gestures: list[GestureResult]) -> float:
        if not gestures:
            return 0.0
        scores = [
            _THREAT_SCORE_MAP.get(g.gesture, 0.0) * g.confidence
            for g in gestures
        ]
        return float(max(scores))

    def _evaluate(
        self, score: float, gestures: list[GestureResult]
    ) -> tuple[ThreatLevel, list[str]]:
        reasons: list[str] = []
        active = [
            g.gesture.name for g in gestures
            if _THREAT_SCORE_MAP.get(g.gesture, 0) > 0
        ]
        if active:
            reasons.append(f"위협 제스처 감지: {', '.join(active)}")

        if score >= self._high:
            return ThreatLevel.HIGH, reasons
        if score >= self._medium:
            return ThreatLevel.MEDIUM, reasons
        if score >= self._low:
            return ThreatLevel.LOW, reasons
        return ThreatLevel.NONE, reasons


class SceneThreatDetector:
    """YOLO 인물 감지 + 접근 분석 + 제스처를 결합한 종합 위협 감지기.

    두 가지 위협 신호를 결합합니다:

    1. **접근 위협** (approach_weight): 두 인물 사이 거리가 빠르게 줄어드는 경우.
       보호 대상을 향한 접근은 레벨을 추가 상향합니다.
    2. **제스처 위협** (gesture_weight): 주먹·포인팅 등 위협 제스처.

    Args:
        yolo_detector: 인물 감지기.
        approach_analyzer: 접근 속도 분석기.
        gesture_detector: 제스처 기반 위협 감지기. None이면 기본값 사용.
        approach_weight: 접근 점수 가중치 (기본 0.6).
        gesture_weight: 제스처 점수 가중치 (기본 0.4).
    """

    def __init__(
        self,
        yolo_detector: YOLOPersonDetector,
        approach_analyzer: ApproachAnalyzer,
        gesture_detector: ThreatDetector | None = None,
        approach_weight: float = 0.6,
        gesture_weight: float = 0.4,
    ) -> None:
        self._yolo = yolo_detector
        self._approach = approach_analyzer
        self._gesture = gesture_detector or ThreatDetector()
        self._approach_weight = approach_weight
        self._gesture_weight = gesture_weight

    def detect(
        self,
        frame: np.ndarray,
        gestures: list[GestureResult],
        protected_person_id: int | None = None,
        timestamp: float | None = None,
    ) -> tuple[ThreatResult, list[PersonDetection]]:
        """프레임을 분석하여 위협 결과와 인물 목록을 반환합니다.

        Args:
            frame: BGR 프레임.
            gestures: 현재 프레임의 GestureResult 목록.
            protected_person_id: 보호 대상 인물 ID. None이면 보호 대상 없음.
            timestamp: 현재 시각. None이면 time.monotonic() 사용.

        Returns:
            (ThreatResult, list[PersonDetection]) 튜플.
        """
        now = timestamp if timestamp is not None else time.monotonic()

        persons = self._yolo.detect(frame)
        events = self._approach.update(persons, now, protected_id=protected_person_id)
        approach_score = self._approach.threat_score(events)

        gesture_result = self._gesture.detect(gestures)
        combined_score = min(
            approach_score * self._approach_weight
            + gesture_result.score * self._gesture_weight,
            1.0,
        )

        reasons = list(gesture_result.reasons)
        for e in events:
            if e.speed >= self._approach.fast_threshold:
                note = " (보호 대상 향함)" if e.is_toward_protected else ""
                reasons.append(
                    f"빠른 접근: P{e.aggressor_id}→P{e.target_id}"
                    f" 속도={e.speed:.3f}{note}"
                )

        level = self._score_to_level(combined_score, events)
        return ThreatResult(level=level, score=combined_score, reasons=reasons), persons

    def reset(self) -> None:
        """모든 내부 상태를 초기화합니다."""
        self._yolo.reset()
        self._approach.reset()
        self._gesture.reset()

    def _score_to_level(
        self, score: float, events: list[ApproachEvent]
    ) -> ThreatLevel:
        max_protected_speed = max(
            (e.speed for e in events if e.is_toward_protected), default=0.0
        )
        threshold = self._approach.fast_threshold

        # 보호 대상을 향한 빠른 접근 → 레벨 강제 상향
        if max_protected_speed >= threshold * 3:
            return ThreatLevel.CRITICAL
        if max_protected_speed >= threshold * 2:
            return ThreatLevel.HIGH

        if score >= 0.75:
            return ThreatLevel.CRITICAL
        if score >= 0.5:
            return ThreatLevel.HIGH
        if score >= 0.25:
            return ThreatLevel.MEDIUM
        if score >= 0.1:
            return ThreatLevel.LOW
        return ThreatLevel.NONE
