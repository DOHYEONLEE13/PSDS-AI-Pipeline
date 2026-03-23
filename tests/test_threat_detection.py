"""tests/test_threat_detection.py — ThreatDetector 및 SceneThreatDetector 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.gesture_recognition.recognizer import Gesture
from src.threat_detection.approach_analyzer import ApproachAnalyzer
from src.threat_detection.detector import (
    SceneThreatDetector,
    ThreatDetector,
    ThreatLevel,
    ThreatResult,
)
from src.threat_detection.yolo_detector import PersonDetection
from tests.conftest import blank_frame, make_gesture_result, make_person, make_threat_result


class TestThreatResult:
    @pytest.mark.parametrize("level,expected", [
        (ThreatLevel.NONE, False),
        (ThreatLevel.LOW, False),
        (ThreatLevel.MEDIUM, True),
        (ThreatLevel.HIGH, True),
        (ThreatLevel.CRITICAL, True),
    ])
    def test_is_threat(self, level: ThreatLevel, expected: bool) -> None:
        result = make_threat_result(level=level)
        assert result.is_threat is expected


class TestThreatDetectorSingleFrame:
    def setup_method(self) -> None:
        self.detector = ThreatDetector(history_window=1)

    def test_no_gestures_returns_none_level(self) -> None:
        result = self.detector.detect([])
        assert result.level == ThreatLevel.NONE
        assert result.score == pytest.approx(0.0)

    def test_open_palm_returns_none_level(self) -> None:
        gestures = [make_gesture_result(Gesture.OPEN_PALM, confidence=1.0)]
        result = self.detector.detect(gestures)
        assert result.level == ThreatLevel.NONE

    def test_fist_high_confidence_returns_high_level(self) -> None:
        # FIST score=0.7, confidence=1.0 → smoothed=0.7 >= high_threshold=0.75? No.
        # Use window=1 and high threshold=0.6 to trigger HIGH
        detector = ThreatDetector(history_window=1, high_threshold=0.6)
        gestures = [make_gesture_result(Gesture.FIST, confidence=1.0)]
        result = detector.detect(gestures)
        assert result.level == ThreatLevel.HIGH

    def test_pointing_returns_medium_level(self) -> None:
        # POINTING score=0.5, confidence=1.0 → smoothed=0.5, medium_threshold=0.45
        gestures = [make_gesture_result(Gesture.POINTING, confidence=1.0)]
        result = self.detector.detect(gestures)
        assert result.level == ThreatLevel.MEDIUM

    def test_low_score_returns_low_level(self) -> None:
        # UNKNOWN score=0.1, confidence=1.0 → 0.1 < medium(0.45), >= low(0.2)? No, 0.1 < 0.2
        # Use low_threshold=0.05 to trigger LOW
        detector = ThreatDetector(history_window=1, low_threshold=0.05)
        gestures = [make_gesture_result(Gesture.UNKNOWN, confidence=1.0)]
        result = detector.detect(gestures)
        assert result.level == ThreatLevel.LOW

    def test_reasons_populated_for_threat_gestures(self) -> None:
        gestures = [make_gesture_result(Gesture.FIST, confidence=1.0)]
        result = self.detector.detect(gestures)
        assert len(result.reasons) > 0
        assert "FIST" in result.reasons[0]

    def test_no_reasons_for_safe_gestures(self) -> None:
        gestures = [make_gesture_result(Gesture.OPEN_PALM, confidence=1.0)]
        result = self.detector.detect(gestures)
        assert result.reasons == []


class TestThreatDetectorHistory:
    def test_score_smoothed_over_window(self) -> None:
        detector = ThreatDetector(history_window=4, high_threshold=0.6, medium_threshold=0.45)
        fist = [make_gesture_result(Gesture.FIST, confidence=1.0)]  # score 0.7
        safe: list = []

        # 1프레임만 위협 → 평균이 낮아 HIGH가 안 됨
        for _ in range(3):
            detector.detect(safe)
        result = detector.detect(fist)
        # 평균 = (0+0+0+0.7)/4 = 0.175 → LOW 미만 or LOW
        assert result.level in (ThreatLevel.NONE, ThreatLevel.LOW)

    def test_repeated_threats_raise_level(self) -> None:
        detector = ThreatDetector(history_window=3, high_threshold=0.6)
        fist = [make_gesture_result(Gesture.FIST, confidence=1.0)]
        for _ in range(3):
            result = detector.detect(fist)
        # 평균 ≈ 0.7 → HIGH
        assert result.level == ThreatLevel.HIGH

    def test_reset_clears_history(self) -> None:
        detector = ThreatDetector(history_window=3, high_threshold=0.6)
        fist = [make_gesture_result(Gesture.FIST, confidence=1.0)]
        for _ in range(3):
            detector.detect(fist)

        detector.reset()
        result = detector.detect([])
        assert result.score == pytest.approx(0.0)

    def test_window_caps_history_length(self) -> None:
        detector = ThreatDetector(history_window=3)
        for _ in range(10):
            detector.detect([])
        assert len(detector._score_history) <= 3


# ---------------------------------------------------------------------------
# SceneThreatDetector 헬퍼
# ---------------------------------------------------------------------------

def _make_scene_detector(
    persons: list[PersonDetection] | None = None,
    fast_threshold: float = 0.1,
) -> SceneThreatDetector:
    """YOLOPersonDetector를 mock으로 대체한 SceneThreatDetector를 반환합니다."""
    mock_yolo = MagicMock()
    mock_yolo.detect.return_value = persons or []
    mock_yolo.reset.return_value = None

    analyzer = ApproachAnalyzer(fast_threshold=fast_threshold)
    gesture_det = ThreatDetector(history_window=1)

    scene = SceneThreatDetector(
        yolo_detector=mock_yolo,
        approach_analyzer=analyzer,
        gesture_detector=gesture_det,
    )
    return scene


class TestSceneThreatDetectorBasic:
    def test_returns_tuple_of_result_and_persons(self) -> None:
        scene = _make_scene_detector(persons=[make_person(0, 0.3, 0.5)])
        result, persons = scene.detect(blank_frame(), [], timestamp=0.0)
        assert isinstance(result, ThreatResult)
        assert isinstance(persons, list)

    def test_no_persons_no_threat(self) -> None:
        scene = _make_scene_detector(persons=[])
        result, persons = scene.detect(blank_frame(), [], timestamp=0.0)
        assert result.level == ThreatLevel.NONE
        assert persons == []

    def test_single_person_no_threat(self) -> None:
        # 혼자 있으면 접근 위협 없음
        scene = _make_scene_detector(persons=[make_person(0, 0.5, 0.5)])
        # 두 프레임 넣어도 접근 대상 없음
        scene.detect(blank_frame(), [], timestamp=0.0)
        result, _ = scene.detect(blank_frame(), [], timestamp=1.0)
        assert result.level == ThreatLevel.NONE

    def test_yolo_detections_returned(self) -> None:
        persons = [make_person(0, 0.2, 0.5), make_person(1, 0.8, 0.5)]
        scene = _make_scene_detector(persons=persons)
        _, returned = scene.detect(blank_frame(), [], timestamp=0.0)
        assert len(returned) == 2

    def test_gesture_score_contributes(self) -> None:
        # FIST 제스처 → gesture_weight=0.4 * 0.7 = 0.28 → MEDIUM
        scene = _make_scene_detector(persons=[])
        fist = [make_gesture_result(Gesture.FIST, confidence=1.0)]
        result, _ = scene.detect(blank_frame(), fist, timestamp=0.0)
        assert result.level in (ThreatLevel.MEDIUM, ThreatLevel.HIGH)


class TestSceneThreatDetectorApproach:
    def _make_two_person_scene(
        self, fast_threshold: float = 0.1
    ) -> SceneThreatDetector:
        mock_yolo = MagicMock()
        mock_yolo.reset.return_value = None
        self._mock_yolo = mock_yolo
        analyzer = ApproachAnalyzer(fast_threshold=fast_threshold)
        return SceneThreatDetector(
            yolo_detector=mock_yolo,
            approach_analyzer=analyzer,
            gesture_detector=ThreatDetector(history_window=1),
        )

    def test_slow_approach_no_threat(self) -> None:
        scene = self._make_two_person_scene(fast_threshold=0.5)
        # 두 사람이 천천히 접근 (속도 ≪ threshold)
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.2, 0.5), make_person(1, 0.8, 0.5)
        ]
        scene.detect(blank_frame(), [], timestamp=0.0)
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.25, 0.5), make_person(1, 0.75, 0.5)
        ]
        result, _ = scene.detect(blank_frame(), [], timestamp=1.0)
        # speed = (0.6-0.5)/1.0 = 0.1 < threshold=0.5
        assert result.level in (ThreatLevel.NONE, ThreatLevel.LOW)

    def test_fast_approach_raises_level(self) -> None:
        scene = self._make_two_person_scene(fast_threshold=0.05)
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)
        ]
        scene.detect(blank_frame(), [], timestamp=0.0)
        # 0.5초 만에 0.4 이동 → speed ≈ 0.8/0.5 = 1.6 >> threshold
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)
        ]
        result, _ = scene.detect(blank_frame(), [], timestamp=0.5)
        assert result.level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)

    def test_fast_approach_toward_protected_is_high_or_critical(self) -> None:
        scene = self._make_two_person_scene(fast_threshold=0.05)
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)
        ]
        scene.detect(blank_frame(), [], timestamp=0.0, protected_person_id=1)
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)
        ]
        result, _ = scene.detect(
            blank_frame(), [], timestamp=0.5, protected_person_id=1
        )
        assert result.level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL)

    def test_approach_toward_protected_in_reasons(self) -> None:
        scene = self._make_two_person_scene(fast_threshold=0.05)
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)
        ]
        scene.detect(blank_frame(), [], timestamp=0.0, protected_person_id=1)
        self._mock_yolo.detect.return_value = [
            make_person(0, 0.35, 0.5), make_person(1, 0.65, 0.5)
        ]
        result, _ = scene.detect(
            blank_frame(), [], timestamp=0.5, protected_person_id=1
        )
        assert any("보호 대상" in r for r in result.reasons)


class TestSceneThreatDetectorReset:
    def test_reset_clears_approach_history(self) -> None:
        mock_yolo = MagicMock()
        mock_yolo.reset.return_value = None
        analyzer = ApproachAnalyzer(fast_threshold=0.05)
        scene = SceneThreatDetector(
            yolo_detector=mock_yolo,
            approach_analyzer=analyzer,
            gesture_detector=ThreatDetector(history_window=1),
        )

        mock_yolo.detect.return_value = [
            make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)
        ]
        scene.detect(blank_frame(), [], timestamp=0.0)
        scene.reset()

        # 리셋 후 첫 프레임 → 이력 없어서 접근 이벤트 없음
        mock_yolo.detect.return_value = [
            make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)
        ]
        result, _ = scene.detect(blank_frame(), [], timestamp=0.5)
        assert result.level == ThreatLevel.NONE
