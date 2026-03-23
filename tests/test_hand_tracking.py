"""tests/test_hand_tracking.py — HandTracker 기본 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.hand_tracking.tracker import (
    HAND_CONNECTIONS,
    HandLandmarks,
    HandTracker,
    LandmarkKalmanFilter,
    TrackingResult,
)
from tests.conftest import blank_frame


# ---------------------------------------------------------------------------
# TrackingResult
# ---------------------------------------------------------------------------


class TestTrackingResult:
    def test_detected_false_when_no_hands(self) -> None:
        result = TrackingResult(hands=[], frame_index=0)
        assert result.detected is False

    def test_detected_true_when_hands_present(self) -> None:
        hand = HandLandmarks(
            landmarks=[(0.0, 0.0, 0.0)] * 21,
            handedness="Right",
            confidence=0.9,
        )
        result = TrackingResult(hands=[hand], frame_index=1)
        assert result.detected is True

    def test_frame_index_stored(self) -> None:
        result = TrackingResult(hands=[], frame_index=42)
        assert result.frame_index == 42


# ---------------------------------------------------------------------------
# HandLandmarks
# ---------------------------------------------------------------------------


class TestHandLandmarks:
    def test_fields_accessible(self) -> None:
        lm = HandLandmarks(
            landmarks=[(0.1, 0.2, 0.3)] * 21,
            handedness="Left",
            confidence=0.85,
        )
        assert lm.handedness == "Left"
        assert lm.confidence == pytest.approx(0.85)
        assert len(lm.landmarks) == 21


# ---------------------------------------------------------------------------
# HAND_CONNECTIONS
# ---------------------------------------------------------------------------


class TestHandConnections:
    def test_connections_nonempty(self) -> None:
        assert len(HAND_CONNECTIONS) > 0

    def test_all_indices_in_range(self) -> None:
        for a, b in HAND_CONNECTIONS:
            assert 0 <= a < 21
            assert 0 <= b < 21

    def test_connections_are_pairs(self) -> None:
        for conn in HAND_CONNECTIONS:
            assert len(conn) == 2


# ---------------------------------------------------------------------------
# LandmarkKalmanFilter
# ---------------------------------------------------------------------------


class TestLandmarkKalmanFilter:
    def test_update_returns_float_tuple(self) -> None:
        kf = LandmarkKalmanFilter()
        result = kf.update(0.5, 0.5, 0.0)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_first_update_initializes_near_input(self) -> None:
        kf = LandmarkKalmanFilter()
        x, y, z = kf.update(0.3, 0.6, 0.1)
        assert abs(x - 0.3) < 0.05
        assert abs(y - 0.6) < 0.05
        assert abs(z - 0.1) < 0.05

    def test_smoothing_reduces_noise(self) -> None:
        """일정한 참값에 노이즈를 더한 측정값을 반복 입력하면
        필터 출력이 원래 측정값보다 참값에 더 가까워야 한다."""
        rng = np.random.default_rng(seed=42)
        true_val = 0.5
        noise_std = 0.05
        kf = LandmarkKalmanFilter(process_noise=1e-3, measurement_noise=1e-2)

        errors_raw, errors_filtered = [], []
        for _ in range(20):
            noisy = true_val + rng.normal(0, noise_std)
            fx, _, _ = kf.update(noisy, 0.0, 0.0)
            errors_raw.append(abs(noisy - true_val))
            errors_filtered.append(abs(fx - true_val))

        assert np.mean(errors_filtered[-10:]) < np.mean(errors_raw[-10:])

    def test_reset_reinitializes_on_next_update(self) -> None:
        kf = LandmarkKalmanFilter()
        kf.update(0.1, 0.1, 0.1)
        assert kf._initialized is True

        kf.reset()
        assert kf._initialized is False

        x, y, z = kf.update(0.9, 0.9, 0.9)
        assert abs(x - 0.9) < 0.05
        assert abs(y - 0.9) < 0.05

    def test_multiple_updates_return_consistent_types(self) -> None:
        kf = LandmarkKalmanFilter()
        for i in range(5):
            result = kf.update(i * 0.1, i * 0.1, 0.0)
            assert isinstance(result, tuple)
            assert len(result) == 3


# ---------------------------------------------------------------------------
# HandTracker helpers
# ---------------------------------------------------------------------------


def _make_mp_result(
    label: str = "Right",
    score: float = 0.95,
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
) -> MagicMock:
    """MediaPipe HandLandmarkerResult 형태의 mock 객체를 생성한다."""
    lm_point = MagicMock()
    lm_point.x = x
    lm_point.y = y
    lm_point.z = z

    classification = MagicMock()
    classification.category_name = label
    classification.score = score

    mock_result = MagicMock()
    mock_result.hand_landmarks = [[lm_point] * 21]
    mock_result.handedness = [[classification]]
    return mock_result


def _empty_mp_result() -> MagicMock:
    mock_result = MagicMock()
    mock_result.hand_landmarks = None
    return mock_result


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------


class TestHandTracker:
    def test_process_raises_if_not_started(self) -> None:
        tracker = HandTracker()
        with pytest.raises(RuntimeError, match="start\\(\\)"):
            tracker.process(blank_frame())

    def test_start_stop(self) -> None:
        mock_landmarker = MagicMock()
        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker()
            tracker.start()
            assert tracker._landmarker is not None
            tracker.stop()
            assert tracker._landmarker is None
            mock_landmarker.close.assert_called_once()

    def test_stop_clears_kalman_filters(self) -> None:
        mock_landmarker = MagicMock()
        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker()
            tracker.start()
            tracker._kalman_filters["Right"] = []
            tracker.stop()
        assert tracker._kalman_filters == {}

    def test_process_returns_empty_result_when_no_detection(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.return_value = _empty_mp_result()

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker()
            tracker.start()
            result = tracker.process(blank_frame())

        assert isinstance(result, TrackingResult)
        assert not result.detected
        assert result.frame_index == 1

    def test_process_increments_frame_index(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.return_value = _empty_mp_result()

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker()
            tracker.start()
            tracker.process(blank_frame())
            tracker.process(blank_frame())
            result = tracker.process(blank_frame())

        assert result.frame_index == 3

    def test_timestamp_increments_per_frame(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.return_value = _empty_mp_result()

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker()
            tracker.start()
            tracker.process(blank_frame())
            tracker.process(blank_frame())

        calls = mock_landmarker.detect_for_video.call_args_list
        ts0 = calls[0][0][1]  # 두 번째 인자 = timestamp_ms
        ts1 = calls[1][0][1]
        assert ts1 - ts0 == HandTracker._MS_PER_FRAME

    def test_context_manager_starts_and_stops(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.return_value = _empty_mp_result()

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            with HandTracker() as tracker:
                assert tracker._landmarker is not None
            assert tracker._landmarker is None

    def test_process_parses_single_hand(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.return_value = _make_mp_result(
            label="Right", score=0.95, x=0.5, y=0.5, z=0.0
        )

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker(use_kalman=False)
            tracker.start()
            result = tracker.process(blank_frame())

        assert result.detected
        assert len(result.hands) == 1
        assert result.hands[0].handedness == "Right"
        assert result.hands[0].confidence == pytest.approx(0.95)
        assert len(result.hands[0].landmarks) == 21

    def test_kalman_filters_created_on_first_detection(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.return_value = _make_mp_result(label="Left")

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker(use_kalman=True)
            tracker.start()
            tracker.process(blank_frame())

        assert "Left" in tracker._kalman_filters
        assert len(tracker._kalman_filters["Left"]) == 21

    def test_kalman_filters_reset_when_hand_disappears(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.side_effect = [
            _make_mp_result(label="Right"),  # 첫 프레임: 감지됨
            _empty_mp_result(),              # 두 번째 프레임: 사라짐
        ]

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker(use_kalman=True)
            tracker.start()
            tracker.process(blank_frame())
            assert tracker._kalman_filters["Right"][0]._initialized is True

            tracker.process(blank_frame())
            assert tracker._kalman_filters["Right"][0]._initialized is False

    def test_process_with_kalman_disabled(self) -> None:
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.return_value = _make_mp_result(
            label="Right", x=0.3, y=0.7, z=0.0
        )

        with patch("src.hand_tracking.tracker.HandLandmarker") as mock_cls:
            mock_cls.create_from_options.return_value = mock_landmarker
            tracker = HandTracker(use_kalman=False)
            tracker.start()
            result = tracker.process(blank_frame())

        assert result.hands[0].landmarks[0] == pytest.approx((0.3, 0.7, 0.0))
        assert tracker._kalman_filters == {}


# ---------------------------------------------------------------------------
# HandTracker.draw
# ---------------------------------------------------------------------------


class TestHandTrackerDraw:
    def _make_hand(self) -> HandLandmarks:
        return HandLandmarks(
            landmarks=[(0.5, 0.5, 0.0)] * 21,
            handedness="Right",
            confidence=0.95,
        )

    def test_draw_returns_same_frame(self) -> None:
        tracker = HandTracker()
        frame = blank_frame(480, 640)
        result = TrackingResult(hands=[], frame_index=1)
        returned = tracker.draw(frame, result)
        assert returned is frame

    def test_draw_empty_result_does_not_raise(self) -> None:
        tracker = HandTracker()
        frame = blank_frame(480, 640)
        result = TrackingResult(hands=[], frame_index=1)
        tracker.draw(frame, result)

    def test_draw_with_hand_modifies_frame(self) -> None:
        tracker = HandTracker()
        frame = blank_frame(480, 640)
        original = frame.copy()
        result = TrackingResult(hands=[self._make_hand()], frame_index=1)
        tracker.draw(frame, result)
        assert not np.array_equal(frame, original)

    def test_draw_returns_ndarray(self) -> None:
        tracker = HandTracker()
        frame = blank_frame(480, 640)
        result = TrackingResult(hands=[self._make_hand()], frame_index=1)
        returned = tracker.draw(frame, result)
        assert isinstance(returned, np.ndarray)

    def test_draw_multiple_hands(self) -> None:
        tracker = HandTracker()
        frame = blank_frame(480, 640)
        hands = [
            HandLandmarks(landmarks=[(0.3, 0.4, 0.0)] * 21, handedness="Left", confidence=0.9),
            HandLandmarks(landmarks=[(0.7, 0.4, 0.0)] * 21, handedness="Right", confidence=0.88),
        ]
        result = TrackingResult(hands=hands, frame_index=1)
        tracker.draw(frame, result)
