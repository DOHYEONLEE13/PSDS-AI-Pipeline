"""tests/test_gesture_recognition.py — GestureRecognizer 기본 테스트."""
from __future__ import annotations

from src.gesture_recognition.recognizer import Gesture, GestureRecognizer
from src.hand_tracking.tracker import TrackingResult
from tests.conftest import make_hand, make_tracking_result


class TestGestureRecognizerClassify:
    def setup_method(self) -> None:
        self.recognizer = GestureRecognizer(confidence_threshold=0.0)

    def test_open_palm_all_extended(self) -> None:
        hand = make_hand(
            thumb_extended=True,
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
        )
        gesture, _ = self.recognizer._classify(hand)
        assert gesture == Gesture.OPEN_PALM

    def test_fist_none_extended(self) -> None:
        hand = make_hand(
            thumb_extended=False,
            index_extended=False,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=False,
        )
        gesture, _ = self.recognizer._classify(hand)
        assert gesture == Gesture.FIST

    def test_pointing_only_index_extended(self) -> None:
        hand = make_hand(
            thumb_extended=False,
            index_extended=True,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=False,
        )
        gesture, _ = self.recognizer._classify(hand)
        assert gesture == Gesture.POINTING

    def test_thumbs_up_only_thumb_extended(self) -> None:
        hand = make_hand(
            thumb_extended=True,
            index_extended=False,
            middle_extended=False,
            ring_extended=False,
            pinky_extended=False,
        )
        gesture, _ = self.recognizer._classify(hand)
        assert gesture == Gesture.THUMBS_UP

    def test_v_sign_index_and_middle_extended(self) -> None:
        # 검지 + 중지만 펴진 경우 → V_SIGN
        hand = make_hand(
            thumb_extended=False,
            index_extended=True,
            middle_extended=True,
            ring_extended=False,
            pinky_extended=False,
        )
        gesture, _ = self.recognizer._classify(hand)
        assert gesture == Gesture.V_SIGN

    def test_unknown_for_ambiguous_combo(self) -> None:
        # 검지 + 중지 + 소지 펴진 경우 → UNKNOWN
        hand = make_hand(
            thumb_extended=False,
            index_extended=True,
            middle_extended=True,
            ring_extended=False,
            pinky_extended=True,
        )
        gesture, _ = self.recognizer._classify(hand)
        assert gesture == Gesture.UNKNOWN


class TestGestureRecognizerRecognize:
    def test_recognize_returns_empty_for_no_hands(self) -> None:
        recognizer = GestureRecognizer()
        result = recognizer.recognize(TrackingResult(hands=[], frame_index=0))
        assert result == []

    def test_recognize_filters_below_threshold(self) -> None:
        recognizer = GestureRecognizer(confidence_threshold=0.99)
        hand = make_hand(
            index_extended=True,  # POINTING → confidence 0.85 < 0.99
        )
        result = recognizer.recognize(make_tracking_result(hand))
        assert result == []

    def test_recognize_returns_gesture_above_threshold(self) -> None:
        recognizer = GestureRecognizer(confidence_threshold=0.5)
        hand = make_hand(
            thumb_extended=True,
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
        )
        results = recognizer.recognize(make_tracking_result(hand))
        assert len(results) == 1
        assert results[0].gesture == Gesture.OPEN_PALM

    def test_recognize_multiple_hands(self) -> None:
        recognizer = GestureRecognizer(confidence_threshold=0.0)
        hand_a = make_hand(
            thumb_extended=True,
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
            handedness="Right",
        )
        hand_b = make_hand(handedness="Left")  # FIST
        results = recognizer.recognize(make_tracking_result(hand_a, hand_b))
        assert len(results) == 2
        gestures = {r.gesture for r in results}
        assert Gesture.OPEN_PALM in gestures
        assert Gesture.FIST in gestures

    def test_recognize_result_fields(self) -> None:
        recognizer = GestureRecognizer(confidence_threshold=0.0)
        hand = make_hand(handedness="Left")  # FIST
        results = recognizer.recognize(make_tracking_result(hand))
        assert results[0].handedness == "Left"
        assert isinstance(results[0].confidence, float)


class TestFingersExtended:
    def test_returns_five_booleans(self) -> None:
        recognizer = GestureRecognizer()
        hand = make_hand()
        flags = recognizer._fingers_extended(hand.landmarks)
        assert len(flags) == 5
        assert all(isinstance(f, bool) for f in flags)
