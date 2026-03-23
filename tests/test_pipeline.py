"""tests/test_pipeline.py — Pipeline 통합 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.conftest import blank_frame, make_gesture_result, make_person
from src.gesture_recognition.recognizer import Gesture, SOSDetectionResult
from src.protection.protected_tracker import ProtectedPersonStatus
from src.threat_detection.detector import ThreatLevel, ThreatResult
from src.threat_detection.yolo_detector import PersonDetection


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------


def _make_mock_yolo(persons: list[PersonDetection] | None = None) -> MagicMock:
    """YOLOPersonDetector 목 객체를 반환합니다."""
    mock = MagicMock()
    mock.detect.return_value = persons or []
    mock._prev_centers = {}
    mock._next_id = 0
    return mock


@pytest.fixture
def pipeline():
    """실제 모델 파일 없이 Pipeline을 생성합니다."""
    from src.pipeline import Pipeline

    p = Pipeline(source=0, yolo_model=_make_mock_yolo())
    # HandTracker는 start() 시 모델 파일이 필요하므로 목으로 대체
    mock_hand = MagicMock()
    mock_hand.__enter__ = MagicMock(return_value=mock_hand)
    mock_hand.__exit__ = MagicMock(return_value=False)
    mock_hand.process.return_value = MagicMock(hands=[], frame_index=0)
    p._hand_tracker = mock_hand
    return p


# ---------------------------------------------------------------------------
# Pipeline 초기화
# ---------------------------------------------------------------------------


class TestPipelineInit:
    def test_default_state(self, pipeline) -> None:
        assert pipeline._fps == 30.0
        assert pipeline._yolo_skip == 1
        assert pipeline._yolo_frame_count == 0
        assert pipeline._last_persons == []
        assert not pipeline._protected_tracker.is_registered

    def test_webcam_source_detected(self) -> None:
        from src.pipeline import Pipeline

        p = Pipeline(source=0, yolo_model=_make_mock_yolo())
        assert p._is_webcam is True

    def test_video_source_detected(self) -> None:
        from src.pipeline import Pipeline

        p = Pipeline(source="video.mp4", yolo_model=_make_mock_yolo())
        assert p._is_webcam is False


# ---------------------------------------------------------------------------
# _update_yolo_skip
# ---------------------------------------------------------------------------


class TestUpdateYoloSkip:
    def test_high_fps_skip1(self, pipeline) -> None:
        pipeline._fps = 30.0
        pipeline._update_yolo_skip()
        assert pipeline._yolo_skip == 1

    def test_medium_fps_skip2(self, pipeline) -> None:
        pipeline._fps = 12.0
        pipeline._update_yolo_skip()
        assert pipeline._yolo_skip == 2

    def test_low_fps_skip3(self, pipeline) -> None:
        pipeline._fps = 8.0
        pipeline._update_yolo_skip()
        assert pipeline._yolo_skip == 3

    def test_boundary_fps_low(self, pipeline) -> None:
        """FPS == _FPS_LOW 경계값은 skip=3."""
        pipeline._fps = pipeline._FPS_LOW - 0.01
        pipeline._update_yolo_skip()
        assert pipeline._yolo_skip == 3

    def test_boundary_fps_med(self, pipeline) -> None:
        """FPS == _FPS_MED 경계값은 skip=2."""
        pipeline._fps = pipeline._FPS_MED - 0.01
        pipeline._update_yolo_skip()
        assert pipeline._yolo_skip == 2


# ---------------------------------------------------------------------------
# _find_protected_person_id
# ---------------------------------------------------------------------------


class TestFindProtectedPersonId:
    def _vsign(self, handedness: str = "Right"):
        return make_gesture_result(Gesture.V_SIGN, handedness=handedness)

    def test_wrist_inside_bbox_matched(self, pipeline) -> None:
        """손목이 바운딩 박스 안에 있으면 해당 person_id 반환."""
        pipeline._last_wrists = {"Right": (0.5, 0.5)}
        persons = [make_person(person_id=7, cx=0.5, cy=0.5)]
        result = pipeline._find_protected_person_id([self._vsign()], persons)
        assert result == 7

    def test_wrist_outside_all_bboxes_returns_first(self, pipeline) -> None:
        """손목이 박스 밖이면 첫 번째 인물 반환."""
        pipeline._last_wrists = {"Right": (0.1, 0.1)}
        persons = [
            make_person(person_id=3, cx=0.8, cy=0.8),
            make_person(person_id=5, cx=0.9, cy=0.9),
        ]
        result = pipeline._find_protected_person_id([self._vsign()], persons)
        assert result == 3

    def test_no_persons_returns_none(self, pipeline) -> None:
        pipeline._last_wrists = {"Right": (0.5, 0.5)}
        result = pipeline._find_protected_person_id([self._vsign()], [])
        assert result is None

    def test_no_v_sign_returns_first(self, pipeline) -> None:
        """V사인이 없어도 첫 번째 인물 반환."""
        pipeline._last_wrists = {}
        persons = [make_person(person_id=2, cx=0.5, cy=0.5)]
        gesture = make_gesture_result(Gesture.OPEN_PALM)
        result = pipeline._find_protected_person_id([gesture], persons)
        assert result == 2

    def test_no_wrist_in_last_wrists_fallback(self, pipeline) -> None:
        """handedness 불일치 시 첫 번째 인물 반환."""
        pipeline._last_wrists = {"Left": (0.5, 0.5)}
        persons = [make_person(person_id=1, cx=0.5, cy=0.5)]
        result = pipeline._find_protected_person_id([self._vsign("Right")], persons)
        assert result == 1

    def test_multiple_persons_wrist_selects_correct(self, pipeline) -> None:
        """여러 인물 중 손목이 포함된 인물 선택."""
        pipeline._last_wrists = {"Right": (0.9, 0.5)}
        persons = [
            make_person(person_id=0, cx=0.2, cy=0.5),
            make_person(person_id=1, cx=0.9, cy=0.5),
        ]
        result = pipeline._find_protected_person_id([self._vsign()], persons)
        assert result == 1


# ---------------------------------------------------------------------------
# _process_sos
# ---------------------------------------------------------------------------


class TestProcessSos:
    def test_no_gesture_returns_not_detected(self, pipeline) -> None:
        result = pipeline._process_sos([])
        assert not result.is_detected
        assert not result.is_pending

    def test_v_sign_below_threshold_not_pending(self, pipeline) -> None:
        gesture = make_gesture_result(Gesture.V_SIGN, confidence=0.5)
        result = pipeline._process_sos([gesture])
        assert not result.is_pending

    def test_non_v_sign_not_pending(self, pipeline) -> None:
        gesture = make_gesture_result(Gesture.FIST, confidence=0.9)
        result = pipeline._process_sos([gesture])
        assert not result.is_pending


# ---------------------------------------------------------------------------
# _process_registration
# ---------------------------------------------------------------------------


class TestProcessRegistration:
    def _detected_sos(self) -> SOSDetectionResult:
        return SOSDetectionResult(
            is_detected=True, is_pending=False, confidence=0.9, held_duration=3.0
        )

    def _not_detected_sos(self) -> SOSDetectionResult:
        return SOSDetectionResult(
            is_detected=False, is_pending=True, confidence=0.9, held_duration=1.0
        )

    def test_registers_on_confirmed_sos(self, pipeline) -> None:
        pipeline._last_wrists = {"Right": (0.5, 0.5)}
        persons = [make_person(person_id=2, cx=0.5, cy=0.5)]
        gesture = make_gesture_result(Gesture.V_SIGN)
        pipeline._process_registration(self._detected_sos(), [gesture], persons)
        assert pipeline._protected_tracker.is_registered
        assert pipeline._protected_tracker.protected_id == 2

    def test_does_not_register_if_already_registered(self, pipeline) -> None:
        pipeline._protected_tracker.register(99)
        pipeline._last_wrists = {"Right": (0.5, 0.5)}
        persons = [make_person(person_id=5, cx=0.5, cy=0.5)]
        gesture = make_gesture_result(Gesture.V_SIGN)
        pipeline._process_registration(self._detected_sos(), [gesture], persons)
        # 먼저 등록된 99번 유지
        assert pipeline._protected_tracker.protected_id == 99

    def test_does_not_register_on_pending_sos(self, pipeline) -> None:
        pipeline._last_wrists = {"Right": (0.5, 0.5)}
        persons = [make_person(person_id=1, cx=0.5, cy=0.5)]
        gesture = make_gesture_result(Gesture.V_SIGN)
        pipeline._process_registration(self._not_detected_sos(), [gesture], persons)
        assert not pipeline._protected_tracker.is_registered

    def test_no_registration_without_persons(self, pipeline) -> None:
        pipeline._last_wrists = {"Right": (0.5, 0.5)}
        gesture = make_gesture_result(Gesture.V_SIGN)
        pipeline._process_registration(self._detected_sos(), [gesture], [])
        assert not pipeline._protected_tracker.is_registered


# ---------------------------------------------------------------------------
# draw_overlay
# ---------------------------------------------------------------------------


class TestDrawOverlay:
    def _no_sos(self) -> SOSDetectionResult:
        return SOSDetectionResult(
            is_detected=False, is_pending=False, confidence=0.0, held_duration=0.0
        )

    def _pending_sos(self, held: float = 1.5) -> SOSDetectionResult:
        return SOSDetectionResult(
            is_detected=False, is_pending=True, confidence=0.9, held_duration=held
        )

    def _confirmed_sos(self) -> SOSDetectionResult:
        return SOSDetectionResult(
            is_detected=True, is_pending=False, confidence=0.9, held_duration=3.0
        )

    def _threat(self, level: ThreatLevel = ThreatLevel.NONE, score: float = 0.0) -> ThreatResult:
        return ThreatResult(level=level, score=score)

    def test_returns_frame_no_error_basic(self, pipeline) -> None:
        frame = blank_frame(480, 640)
        result = pipeline.draw_overlay(frame, self._no_sos(), [], None, self._threat())
        assert result is frame

    def test_draws_regular_person_box(self, pipeline) -> None:
        frame = blank_frame(480, 640)
        persons = [make_person(person_id=0, cx=0.5, cy=0.5)]
        result = pipeline.draw_overlay(frame, self._no_sos(), persons, None, self._threat())
        # 파란색 계열 픽셀이 그려졌는지 확인
        assert result.sum() > 0

    def test_draws_protected_person_green_box(self, pipeline) -> None:
        frame = blank_frame(480, 640)
        persons = [make_person(person_id=1, cx=0.5, cy=0.5)]
        status = ProtectedPersonStatus(
            person_id=1,
            is_in_frame=True,
            bbox=(0.4, 0.4, 0.6, 0.6),
            just_disappeared=False,
        )
        result = pipeline.draw_overlay(frame, self._no_sos(), persons, status, self._threat())
        assert result.sum() > 0

    def test_draws_disappear_warning_when_missing(self, pipeline) -> None:
        frame = blank_frame(480, 640)
        status = ProtectedPersonStatus(
            person_id=1,
            is_in_frame=False,
            bbox=(0.4, 0.4, 0.6, 0.6),
            just_disappeared=True,
        )
        result = pipeline.draw_overlay(frame, self._no_sos(), [], status, self._threat())
        assert result.sum() > 0

    def test_no_warning_when_protected_has_no_bbox(self, pipeline) -> None:
        """박스 없는 보호 대상 이탈 시 경고 없이도 정상 동작."""
        frame = blank_frame(480, 640)
        status = ProtectedPersonStatus(
            person_id=1,
            is_in_frame=False,
            bbox=None,
            just_disappeared=False,
        )
        result = pipeline.draw_overlay(frame, self._no_sos(), [], status, self._threat())
        assert result is frame

    def test_draws_sos_pending_bar(self, pipeline) -> None:
        frame = blank_frame(480, 640)
        result = pipeline.draw_overlay(frame, self._pending_sos(1.5), [], None, self._threat())
        assert result.sum() > 0

    def test_draws_sos_confirmed(self, pipeline) -> None:
        frame = blank_frame(480, 640)
        result = pipeline.draw_overlay(frame, self._confirmed_sos(), [], None, self._threat())
        assert result.sum() > 0

    def test_threat_gauge_full_score(self, pipeline) -> None:
        frame = blank_frame(480, 640)
        threat = self._threat(ThreatLevel.CRITICAL, score=1.0)
        result = pipeline.draw_overlay(frame, self._no_sos(), [], None, threat)
        assert result.sum() > 0

    def test_regular_persons_skip_protected_id(self, pipeline) -> None:
        """보호 대상과 동일한 ID 인물은 파란 박스 없이 초록 박스만."""
        frame = blank_frame(480, 640)
        persons = [make_person(person_id=3, cx=0.5, cy=0.5)]
        status = ProtectedPersonStatus(
            person_id=3,
            is_in_frame=True,
            bbox=(0.4, 0.4, 0.6, 0.6),
            just_disappeared=False,
        )
        # 예외 없이 실행되면 통과
        pipeline.draw_overlay(frame, self._no_sos(), persons, status, self._threat())


# ---------------------------------------------------------------------------
# _process_yolo (스킵 로직)
# ---------------------------------------------------------------------------


class TestProcessYolo:
    def test_runs_on_first_frame(self, pipeline) -> None:
        threat_out = ThreatResult(level=ThreatLevel.NONE, score=0.0)
        pipeline._scene_detector = MagicMock()
        pipeline._scene_detector.detect.return_value = (threat_out, [])
        pipeline._yolo_frame_count = 0
        pipeline._yolo_skip = 1

        persons, threat = pipeline._process_yolo(blank_frame(), [])
        pipeline._scene_detector.detect.assert_called_once()

    def test_skips_on_second_frame_when_skip2(self, pipeline) -> None:
        """FPS=12(skip=2) 일 때 홀수 번째 프레임은 이전 결과 재사용.

        _process_yolo 내부 순서:
        1. frame_count += 1  (0 → 1)
        2. _update_yolo_skip() → fps=12 이므로 skip=2
        3. 1 % 2 != 0 → 스킵
        """
        pipeline._scene_detector = MagicMock()
        pipeline._fps = 12.0             # _update_yolo_skip이 skip=2로 설정
        pipeline._yolo_frame_count = 0   # 증가 후 1 → 1%2=1 → 스킵

        prev_person = make_person(person_id=0, cx=0.5, cy=0.5)
        pipeline._last_persons = [prev_person]
        pipeline._last_threat = ThreatResult(level=ThreatLevel.NONE, score=0.0)

        persons, _ = pipeline._process_yolo(blank_frame(), [])
        pipeline._scene_detector.detect.assert_not_called()
        assert persons == [prev_person]


# ---------------------------------------------------------------------------
# run() 메인 루프 통합 테스트
# ---------------------------------------------------------------------------


class TestRun:
    def _frame(self) -> np.ndarray:
        return blank_frame(480, 640)

    def test_run_exits_on_read_false(self, pipeline) -> None:
        """VideoCapture.read() 가 False 를 반환하면 루프 종료."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, self._frame()), (False, None)]

        pipeline._scene_detector = MagicMock()
        pipeline._scene_detector.detect.return_value = (
            ThreatResult(level=ThreatLevel.NONE, score=0.0), []
        )

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("cv2.imshow"),
            patch("cv2.waitKey", return_value=0),
            patch("cv2.destroyAllWindows"),
        ):
            pipeline.run()  # 예외 없이 종료

    def test_run_exits_on_q_key(self, pipeline) -> None:
        """q 키 입력 시 루프 종료."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, self._frame())

        pipeline._scene_detector = MagicMock()
        pipeline._scene_detector.detect.return_value = (
            ThreatResult(level=ThreatLevel.NONE, score=0.0), []
        )

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("cv2.imshow"),
            patch("cv2.waitKey", return_value=ord("q")),
            patch("cv2.destroyAllWindows"),
        ):
            pipeline.run()

    def test_run_raises_on_invalid_source(self, pipeline) -> None:
        """영상 소스를 열 수 없으면 RuntimeError."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        with patch("cv2.VideoCapture", return_value=mock_cap):
            with pytest.raises(RuntimeError, match="영상 소스를 열 수 없습니다"):
                pipeline.run()

    def test_run_video_file_no_flip(self) -> None:
        """동영상 파일 입력 시 flip 미적용."""
        from src.pipeline import Pipeline

        p = Pipeline(source="video.mp4", yolo_model=_make_mock_yolo())
        mock_hand = MagicMock()
        mock_hand.__enter__ = MagicMock(return_value=mock_hand)
        mock_hand.__exit__ = MagicMock(return_value=False)
        mock_hand.process.return_value = MagicMock(hands=[], frame_index=0)
        p._hand_tracker = mock_hand

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, blank_frame(480, 640)), (False, None)]

        p._scene_detector = MagicMock()
        p._scene_detector.detect.return_value = (
            ThreatResult(level=ThreatLevel.NONE, score=0.0), []
        )

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("cv2.imshow"),
            patch("cv2.waitKey", return_value=0),
            patch("cv2.destroyAllWindows"),
            patch("cv2.flip") as mock_flip,
        ):
            p.run()
            mock_flip.assert_not_called()

    def test_hand_tracker_error_does_not_crash(self, pipeline) -> None:
        """HandTracker 오류 시 다른 모듈은 계속 동작."""
        pipeline._hand_tracker.process.side_effect = RuntimeError("모델 없음")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, self._frame()), (False, None)]

        pipeline._scene_detector = MagicMock()
        pipeline._scene_detector.detect.return_value = (
            ThreatResult(level=ThreatLevel.NONE, score=0.0), []
        )

        with (
            patch("cv2.VideoCapture", return_value=mock_cap),
            patch("cv2.imshow"),
            patch("cv2.waitKey", return_value=0),
            patch("cv2.destroyAllWindows"),
        ):
            pipeline.run()  # 예외 없이 완료


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_webcam_default(self) -> None:
        """--input 없으면 웹캠(0) 사용."""
        from src.pipeline import main

        mock_pipeline = MagicMock()
        with (
            patch("src.pipeline.Pipeline", return_value=mock_pipeline) as MockPipeline,
            patch("sys.argv", ["pipeline"]),
        ):
            main()
            args, kwargs = MockPipeline.call_args
            assert kwargs.get("source", args[0] if args else None) == 0

    def test_main_video_file(self) -> None:
        """--input video.mp4 이면 파일 경로 사용."""
        from src.pipeline import main

        mock_pipeline = MagicMock()
        with (
            patch("src.pipeline.Pipeline", return_value=mock_pipeline) as MockPipeline,
            patch("sys.argv", ["pipeline", "--input", "video.mp4"]),
        ):
            main()
            args, kwargs = MockPipeline.call_args
            source = kwargs.get("source", args[0] if args else None)
            assert source == "video.mp4"
