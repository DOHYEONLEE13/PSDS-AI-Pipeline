"""tests/test_yolo_detector.py — YOLOPersonDetector 테스트."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.threat_detection.yolo_detector import PersonDetection, YOLOPersonDetector
from tests.conftest import blank_frame, make_person

# ---------------------------------------------------------------------------
# Mock YOLO 헬퍼
# ---------------------------------------------------------------------------

class _MockBox:
    """ultralytics Box 최소 mock."""

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        conf: float = 0.9,
        cls_id: int = 0,
    ) -> None:
        self.xyxy = [torch.tensor([x1, y1, x2, y2], dtype=torch.float32)]
        self.conf = [torch.tensor(conf, dtype=torch.float32)]
        self.cls = [torch.tensor(cls_id, dtype=torch.float32)]


class _MockResult:
    def __init__(self, boxes: list[_MockBox]) -> None:
        self.boxes = boxes


class _MockYOLO:
    """호출할 때마다 정해진 결과를 순서대로 반환하는 mock."""

    def __init__(self, call_results: list[list[_MockBox]]) -> None:
        self._queue = list(call_results)

    def __call__(self, frame: np.ndarray, **kwargs: object) -> list[_MockResult]:
        boxes = self._queue.pop(0) if self._queue else []
        return [_MockResult(boxes)]


def _pixel_box(
    cx: float, cy: float, frame_w: int = 64, frame_h: int = 64,
    half_pw: int = 5, half_ph: int = 10, conf: float = 0.9, cls_id: int = 0,
) -> _MockBox:
    """픽셀 좌표 기반 바운딩 박스 mock을 생성합니다."""
    x1 = cx * frame_w - half_pw
    y1 = cy * frame_h - half_ph
    x2 = cx * frame_w + half_pw
    y2 = cy * frame_h + half_ph
    return _MockBox(x1, y1, x2, y2, conf=conf, cls_id=cls_id)


# ---------------------------------------------------------------------------
# PersonDetection
# ---------------------------------------------------------------------------

class TestPersonDetection:
    def test_center_computed_correctly(self) -> None:
        det = make_person(0, 0.3, 0.4)
        cx, cy = det.center
        assert cx == pytest.approx(0.3, abs=1e-3)
        assert cy == pytest.approx(0.4, abs=1e-3)

    def test_fields_accessible(self) -> None:
        det = PersonDetection(person_id=2, bbox=(0.1, 0.2, 0.5, 0.6), confidence=0.85)
        assert det.person_id == 2
        assert det.bbox == (0.1, 0.2, 0.5, 0.6)
        assert det.confidence == pytest.approx(0.85)

    def test_center_of_full_frame(self) -> None:
        det = PersonDetection(person_id=0, bbox=(0.0, 0.0, 1.0, 1.0), confidence=1.0)
        assert det.center == pytest.approx((0.5, 0.5))


# ---------------------------------------------------------------------------
# YOLOPersonDetector — detect
# ---------------------------------------------------------------------------

class TestYOLOPersonDetectorDetect:
    def test_no_detection_returns_empty(self) -> None:
        mock = _MockYOLO([[]])
        det = YOLOPersonDetector(model=mock)
        result = det.detect(blank_frame())
        assert result == []

    def test_single_person_id_zero(self) -> None:
        frame = blank_frame(64, 64)
        mock = _MockYOLO([[_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)]])
        det = YOLOPersonDetector(model=mock)
        persons = det.detect(frame)
        assert len(persons) == 1
        assert persons[0].person_id == 0

    def test_multiple_persons_unique_ids(self) -> None:
        frame = blank_frame(64, 64)
        boxes = [
            _pixel_box(0.2, 0.5, frame_w=64, frame_h=64),
            _pixel_box(0.8, 0.5, frame_w=64, frame_h=64),
        ]
        mock = _MockYOLO([boxes])
        det = YOLOPersonDetector(model=mock)
        persons = det.detect(frame)
        ids = [p.person_id for p in persons]
        assert len(set(ids)) == 2  # 고유 ID

    def test_non_person_class_filtered(self) -> None:
        frame = blank_frame(64, 64)
        # cls_id=1 (차량) → 필터됨
        mock = _MockYOLO([[_pixel_box(0.5, 0.5, frame_w=64, frame_h=64, cls_id=1)]])
        det = YOLOPersonDetector(model=mock)
        persons = det.detect(frame)
        assert persons == []

    def test_bbox_normalized(self) -> None:
        frame = blank_frame(64, 64)
        mock = _MockYOLO([[_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)]])
        det = YOLOPersonDetector(model=mock)
        persons = det.detect(frame)
        x1, y1, x2, y2 = persons[0].bbox
        assert 0.0 <= x1 < x2 <= 1.0
        assert 0.0 <= y1 < y2 <= 1.0

    def test_confidence_stored(self) -> None:
        frame = blank_frame(64, 64)
        mock = _MockYOLO([[_pixel_box(0.5, 0.5, conf=0.77, frame_w=64, frame_h=64)]])
        det = YOLOPersonDetector(model=mock)
        persons = det.detect(frame)
        assert persons[0].confidence == pytest.approx(0.77, abs=1e-3)


# ---------------------------------------------------------------------------
# YOLOPersonDetector — ID 추적 (프레임 간 일관성)
# ---------------------------------------------------------------------------

class TestYOLOPersonDetectorIDTracking:
    def test_same_person_keeps_id_across_frames(self) -> None:
        frame = blank_frame(64, 64)
        # 프레임1: P0 at (0.5, 0.5)
        # 프레임2: P0 이 조금 이동 (0.52, 0.5) → 같은 ID 유지
        mock = _MockYOLO([
            [_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)],
            [_pixel_box(0.52, 0.5, frame_w=64, frame_h=64)],
        ])
        det = YOLOPersonDetector(model=mock, match_threshold=0.3)
        p1 = det.detect(frame)
        p2 = det.detect(frame)
        assert p1[0].person_id == p2[0].person_id

    def test_new_person_gets_new_id(self) -> None:
        frame = blank_frame(64, 64)
        # 프레임1: P0 at (0.2, 0.5)
        # 프레임2: P0 유지 + P1 새로 등장 (0.8, 0.5) → 서로 다른 ID
        mock = _MockYOLO([
            [_pixel_box(0.2, 0.5, frame_w=64, frame_h=64)],
            [
                _pixel_box(0.2, 0.5, frame_w=64, frame_h=64),
                _pixel_box(0.8, 0.5, frame_w=64, frame_h=64),
            ],
        ])
        det = YOLOPersonDetector(model=mock, match_threshold=0.3)
        p1 = det.detect(frame)
        p2 = det.detect(frame)
        ids_frame2 = {p.person_id for p in p2}
        ids_frame1 = {p.person_id for p in p1}
        # 첫 프레임 ID가 두 번째 프레임에도 있어야 함
        assert ids_frame1.issubset(ids_frame2)
        # 두 번째 프레임에 새 ID 추가
        assert len(ids_frame2) == 2

    def test_disappearing_person_id_not_reused_immediately(self) -> None:
        frame = blank_frame(64, 64)
        mock = _MockYOLO([
            [_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)],  # P0
            [],                                                  # P0 사라짐
            [_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)],  # 재등장
        ])
        det = YOLOPersonDetector(model=mock, match_threshold=0.3)
        p1 = det.detect(frame)
        det.detect(frame)  # 사라짐
        p3 = det.detect(frame)
        # 이전 이력이 없으므로 새 ID 부여
        assert p3[0].person_id != p1[0].person_id


# ---------------------------------------------------------------------------
# YOLOPersonDetector — draw
# ---------------------------------------------------------------------------

class TestYOLOPersonDetectorDraw:
    def test_draw_returns_ndarray(self) -> None:
        det = YOLOPersonDetector(model=_MockYOLO([]))
        frame = blank_frame(128, 128)
        persons = [make_person(0, 0.5, 0.5)]
        out = det.draw(frame, persons)
        assert isinstance(out, np.ndarray)

    def test_draw_does_not_modify_original(self) -> None:
        det = YOLOPersonDetector(model=_MockYOLO([]))
        frame = blank_frame(128, 128)
        original = frame.copy()
        persons = [make_person(0, 0.5, 0.5)]
        det.draw(frame, persons)
        np.testing.assert_array_equal(frame, original)

    def test_draw_same_shape(self) -> None:
        det = YOLOPersonDetector(model=_MockYOLO([]))
        frame = blank_frame(128, 64)
        persons = [make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)]
        out = det.draw(frame, persons)
        assert out.shape == frame.shape

    def test_draw_empty_detections(self) -> None:
        det = YOLOPersonDetector(model=_MockYOLO([]))
        frame = blank_frame()
        out = det.draw(frame, [])
        np.testing.assert_array_equal(out, frame)


# ---------------------------------------------------------------------------
# YOLOPersonDetector — reset
# ---------------------------------------------------------------------------

class TestYOLOPersonDetectorReset:
    def test_reset_clears_id_state(self) -> None:
        frame = blank_frame(64, 64)
        mock = _MockYOLO([
            [_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)],
            [_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)],
        ])
        det = YOLOPersonDetector(model=mock, match_threshold=0.3)
        p1 = det.detect(frame)
        det.reset()
        p2 = det.detect(frame)
        # 리셋 후 새로 시작이므로 ID 0부터 재할당
        assert p2[0].person_id == 0
        assert p1[0].person_id == p2[0].person_id  # 둘 다 0

    def test_reset_next_id_to_zero(self) -> None:
        frame = blank_frame(64, 64)
        boxes_2 = [
            _pixel_box(0.2, 0.5, frame_w=64, frame_h=64),
            _pixel_box(0.8, 0.5, frame_w=64, frame_h=64),
        ]
        mock = _MockYOLO([boxes_2, [_pixel_box(0.5, 0.5, frame_w=64, frame_h=64)]])
        det = YOLOPersonDetector(model=mock)
        det.detect(frame)   # P0, P1 등록
        det.reset()
        p = det.detect(frame)
        assert p[0].person_id == 0  # 리셋 후 0부터 재시작
