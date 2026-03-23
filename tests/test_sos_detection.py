"""tests/test_sos_detection.py — RuleBasedSOSDetector 테스트."""
from __future__ import annotations

import pytest

from src.gesture_recognition.recognizer import (
    Gesture,
    GestureResult,
    RuleBasedSOSDetector,
    SOSDetectionResult,
)

# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def make_v_sign_result(confidence: float = 0.9) -> GestureResult:
    return GestureResult(gesture=Gesture.V_SIGN, confidence=confidence, handedness="Right")


def make_fist_result(confidence: float = 0.9) -> GestureResult:
    return GestureResult(gesture=Gesture.FIST, confidence=confidence, handedness="Right")


# ---------------------------------------------------------------------------
# SOSDetectionResult 필드
# ---------------------------------------------------------------------------

class TestSOSDetectionResult:
    def test_fields_accessible(self) -> None:
        result = SOSDetectionResult(
            is_detected=False,
            is_pending=True,
            confidence=0.9,
            held_duration=1.5,
        )
        assert result.is_detected is False
        assert result.is_pending is True
        assert result.confidence == pytest.approx(0.9)
        assert result.held_duration == pytest.approx(1.5)

    def test_detected_and_pending_mutually_exclusive(self) -> None:
        # 최종 확정 상태에서는 pending이 False
        result = SOSDetectionResult(
            is_detected=True, is_pending=False, confidence=0.95, held_duration=3.1
        )
        assert not result.is_pending

    def test_not_detected_not_pending_for_no_v_sign(self) -> None:
        result = SOSDetectionResult(
            is_detected=False, is_pending=False, confidence=0.0, held_duration=0.0
        )
        assert not result.is_detected
        assert not result.is_pending


# ---------------------------------------------------------------------------
# RuleBasedSOSDetector — 기본 동작
# ---------------------------------------------------------------------------

class TestRuleBasedSOSDetectorBasic:
    def setup_method(self) -> None:
        self.detector = RuleBasedSOSDetector(hold_seconds=3.0, confidence_threshold=0.85)

    def test_no_v_sign_returns_not_pending(self) -> None:
        result = self.detector.update([make_fist_result()], timestamp=0.0)
        assert result.is_pending is False
        assert result.is_detected is False

    def test_v_sign_below_threshold_not_pending(self) -> None:
        result = self.detector.update(
            [make_v_sign_result(confidence=0.5)], timestamp=0.0
        )
        assert result.is_pending is False
        assert result.is_detected is False

    def test_v_sign_above_threshold_is_pending(self) -> None:
        result = self.detector.update(
            [make_v_sign_result(confidence=0.9)], timestamp=0.0
        )
        assert result.is_pending is True
        assert result.is_detected is False

    def test_v_sign_exact_threshold_is_pending(self) -> None:
        result = self.detector.update(
            [make_v_sign_result(confidence=0.85)], timestamp=0.0
        )
        assert result.is_pending is True

    def test_confidence_reflected_in_result(self) -> None:
        result = self.detector.update(
            [make_v_sign_result(confidence=0.92)], timestamp=0.0
        )
        assert result.confidence == pytest.approx(0.92)

    def test_held_duration_zero_on_first_frame(self) -> None:
        result = self.detector.update(
            [make_v_sign_result()], timestamp=100.0
        )
        assert result.held_duration == pytest.approx(0.0)

    def test_empty_gesture_list_not_pending(self) -> None:
        result = self.detector.update([], timestamp=0.0)
        assert result.is_pending is False
        assert result.is_detected is False


# ---------------------------------------------------------------------------
# RuleBasedSOSDetector — 시간 누적
# ---------------------------------------------------------------------------

class TestRuleBasedSOSDetectorTiming:
    def setup_method(self) -> None:
        self.detector = RuleBasedSOSDetector(hold_seconds=3.0, confidence_threshold=0.85)

    def test_not_confirmed_before_hold_seconds(self) -> None:
        self.detector.update([make_v_sign_result()], timestamp=0.0)
        result = self.detector.update([make_v_sign_result()], timestamp=2.9)
        assert result.is_detected is False
        assert result.is_pending is True

    def test_confirmed_at_hold_seconds(self) -> None:
        self.detector.update([make_v_sign_result()], timestamp=0.0)
        result = self.detector.update([make_v_sign_result()], timestamp=3.0)
        assert result.is_detected is True
        assert result.is_pending is False

    def test_confirmed_after_hold_seconds(self) -> None:
        self.detector.update([make_v_sign_result()], timestamp=0.0)
        result = self.detector.update([make_v_sign_result()], timestamp=5.0)
        assert result.is_detected is True

    def test_held_duration_accumulates(self) -> None:
        self.detector.update([make_v_sign_result()], timestamp=10.0)
        result = self.detector.update([make_v_sign_result()], timestamp=11.5)
        assert result.held_duration == pytest.approx(1.5)

    def test_timer_resets_when_v_sign_breaks(self) -> None:
        # 2초 유지 후 끊김
        self.detector.update([make_v_sign_result()], timestamp=0.0)
        self.detector.update([make_v_sign_result()], timestamp=2.0)
        self.detector.update([make_fist_result()], timestamp=2.5)  # 끊김
        # 다시 V사인 시작
        self.detector.update([make_v_sign_result()], timestamp=3.0)
        result = self.detector.update([make_v_sign_result()], timestamp=5.9)
        # 리셋 후 2.9초 유지 → 미확정
        assert result.is_detected is False
        assert result.held_duration == pytest.approx(2.9)

    def test_timer_resets_and_confirms_after_full_hold(self) -> None:
        # 중간에 끊기고 다시 3초 유지
        self.detector.update([make_v_sign_result()], timestamp=0.0)
        self.detector.update([make_fist_result()], timestamp=1.0)   # 끊김
        self.detector.update([make_v_sign_result()], timestamp=2.0)  # 재시작
        result = self.detector.update([make_v_sign_result()], timestamp=5.0)
        assert result.is_detected is True
        assert result.held_duration == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# RuleBasedSOSDetector — reset
# ---------------------------------------------------------------------------

class TestRuleBasedSOSDetectorReset:
    def setup_method(self) -> None:
        self.detector = RuleBasedSOSDetector(hold_seconds=3.0, confidence_threshold=0.85)

    def test_reset_clears_timer(self) -> None:
        self.detector.update([make_v_sign_result()], timestamp=0.0)
        self.detector.reset()
        # 리셋 후 다시 시작 → 유지 시간 0
        result = self.detector.update([make_v_sign_result()], timestamp=0.5)
        assert result.held_duration == pytest.approx(0.0)

    def test_reset_before_confirmation_prevents_detection(self) -> None:
        self.detector.update([make_v_sign_result()], timestamp=0.0)
        self.detector.update([make_v_sign_result()], timestamp=2.9)
        self.detector.reset()
        result = self.detector.update([make_v_sign_result()], timestamp=3.0)
        # 리셋 후 재시작이므로 0.0초 유지
        assert result.is_detected is False

    def test_reset_on_clean_detector_is_safe(self) -> None:
        self.detector.reset()  # 예외 없이 동작해야 함


# ---------------------------------------------------------------------------
# RuleBasedSOSDetector — 다중 손
# ---------------------------------------------------------------------------

class TestRuleBasedSOSDetectorMultiHand:
    def setup_method(self) -> None:
        self.detector = RuleBasedSOSDetector(hold_seconds=3.0, confidence_threshold=0.85)

    def test_uses_highest_confidence_among_multiple_hands(self) -> None:
        low = GestureResult(gesture=Gesture.V_SIGN, confidence=0.87, handedness="Left")
        high = GestureResult(gesture=Gesture.V_SIGN, confidence=0.95, handedness="Right")
        result = self.detector.update([low, high], timestamp=0.0)
        assert result.confidence == pytest.approx(0.95)

    def test_v_sign_one_hand_fist_other_still_pending(self) -> None:
        v = make_v_sign_result(confidence=0.9)
        fist = make_fist_result()
        result = self.detector.update([v, fist], timestamp=0.0)
        assert result.is_pending is True

    def test_no_v_sign_in_multiple_hands_not_pending(self) -> None:
        result = self.detector.update(
            [make_fist_result(), make_fist_result()], timestamp=0.0
        )
        assert result.is_pending is False


# ---------------------------------------------------------------------------
# RuleBasedSOSDetector — 커스텀 파라미터
# ---------------------------------------------------------------------------

class TestRuleBasedSOSDetectorCustomParams:
    def test_custom_hold_seconds(self) -> None:
        detector = RuleBasedSOSDetector(hold_seconds=1.0, confidence_threshold=0.85)
        detector.update([make_v_sign_result()], timestamp=0.0)
        result = detector.update([make_v_sign_result()], timestamp=1.0)
        assert result.is_detected is True

    def test_custom_confidence_threshold(self) -> None:
        # threshold=0.95 → confidence=0.9 는 pending 아님
        detector = RuleBasedSOSDetector(hold_seconds=3.0, confidence_threshold=0.95)
        result = detector.update([make_v_sign_result(confidence=0.9)], timestamp=0.0)
        assert result.is_pending is False

    def test_custom_confidence_threshold_above(self) -> None:
        detector = RuleBasedSOSDetector(hold_seconds=3.0, confidence_threshold=0.95)
        result = detector.update([make_v_sign_result(confidence=0.96)], timestamp=0.0)
        assert result.is_pending is True
