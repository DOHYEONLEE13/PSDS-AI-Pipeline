"""tests/test_protected_tracker.py — ProtectedPersonTracker 테스트."""
from __future__ import annotations

import numpy as np

from src.protection.protected_tracker import ProtectedPersonStatus, ProtectedPersonTracker
from tests.conftest import blank_frame, make_person

# ---------------------------------------------------------------------------
# ProtectedPersonStatus
# ---------------------------------------------------------------------------

class TestProtectedPersonStatus:
    def test_fields_accessible(self) -> None:
        s = ProtectedPersonStatus(
            person_id=2,
            is_in_frame=True,
            bbox=(0.1, 0.2, 0.5, 0.7),
            just_disappeared=False,
        )
        assert s.person_id == 2
        assert s.is_in_frame is True
        assert s.bbox == (0.1, 0.2, 0.5, 0.7)
        assert s.just_disappeared is False

    def test_bbox_can_be_none(self) -> None:
        s = ProtectedPersonStatus(
            person_id=0, is_in_frame=False, bbox=None, just_disappeared=False
        )
        assert s.bbox is None

    def test_just_disappeared_true(self) -> None:
        s = ProtectedPersonStatus(
            person_id=1, is_in_frame=False, bbox=None, just_disappeared=True
        )
        assert s.just_disappeared is True


# ---------------------------------------------------------------------------
# ProtectedPersonTracker — 속성 / 등록 / 해제
# ---------------------------------------------------------------------------

class TestProtectedPersonTrackerRegistration:
    def setup_method(self) -> None:
        self.tracker = ProtectedPersonTracker()

    def test_initial_not_registered(self) -> None:
        assert not self.tracker.is_registered
        assert self.tracker.protected_id is None

    def test_register_sets_id(self) -> None:
        self.tracker.register(3)
        assert self.tracker.is_registered
        assert self.tracker.protected_id == 3

    def test_register_overwrites_previous(self) -> None:
        self.tracker.register(1)
        self.tracker.register(5)
        assert self.tracker.protected_id == 5

    def test_clear_removes_registration(self) -> None:
        self.tracker.register(2)
        self.tracker.clear()
        assert not self.tracker.is_registered
        assert self.tracker.protected_id is None

    def test_clear_without_registration_is_safe(self) -> None:
        self.tracker.clear()  # 예외 없어야 함

    def test_update_returns_none_when_not_registered(self) -> None:
        status = self.tracker.update([make_person(0, 0.5, 0.5)])
        assert status is None


# ---------------------------------------------------------------------------
# ProtectedPersonTracker — update: 화면 내 / 외
# ---------------------------------------------------------------------------

class TestProtectedPersonTrackerUpdate:
    def setup_method(self) -> None:
        self.tracker = ProtectedPersonTracker()
        self.tracker.register(1)

    def test_protected_in_frame(self) -> None:
        status = self.tracker.update([make_person(1, 0.5, 0.5)])
        assert status is not None
        assert status.person_id == 1
        assert status.is_in_frame is True

    def test_protected_not_in_frame_first_frame(self) -> None:
        # 처음부터 없음 → just_disappeared=False (이전에 없었으므로)
        status = self.tracker.update([make_person(0, 0.5, 0.5)])  # 다른 사람만 있음
        assert status is not None
        assert status.is_in_frame is False
        assert status.just_disappeared is False

    def test_bbox_updated_when_in_frame(self) -> None:
        person = make_person(1, 0.4, 0.6)
        status = self.tracker.update([person])
        assert status.bbox == person.bbox

    def test_bbox_preserved_when_not_in_frame(self) -> None:
        # 첫 프레임: 보호 대상 있음 → bbox 저장
        first = make_person(1, 0.3, 0.4)
        self.tracker.update([first])
        # 두 번째 프레임: 사라짐 → 이전 bbox 유지
        status = self.tracker.update([make_person(0, 0.9, 0.9)])
        assert status.bbox == first.bbox

    def test_bbox_none_if_never_seen(self) -> None:
        # 등록 후 한 번도 감지 안 됨
        status = self.tracker.update([])
        assert status.bbox is None

    def test_multiple_persons_only_protected_tracked(self) -> None:
        p0 = make_person(0, 0.2, 0.5)
        p1 = make_person(1, 0.7, 0.5)  # 보호 대상
        status = self.tracker.update([p0, p1])
        assert status.is_in_frame is True
        assert status.person_id == 1

    def test_empty_frame_not_in_frame(self) -> None:
        status = self.tracker.update([])
        assert status.is_in_frame is False

    def test_update_after_clear_returns_none(self) -> None:
        self.tracker.clear()
        status = self.tracker.update([make_person(1, 0.5, 0.5)])
        assert status is None


# ---------------------------------------------------------------------------
# ProtectedPersonTracker — just_disappeared
# ---------------------------------------------------------------------------

class TestProtectedPersonTrackerDisappearance:
    def setup_method(self) -> None:
        self.tracker = ProtectedPersonTracker()
        self.tracker.register(1)

    def test_just_disappeared_false_when_in_frame(self) -> None:
        # 두 프레임 연속 있음
        self.tracker.update([make_person(1, 0.5, 0.5)])
        status = self.tracker.update([make_person(1, 0.5, 0.5)])
        assert status.just_disappeared is False

    def test_just_disappeared_true_on_first_missing_frame(self) -> None:
        # 첫 프레임: 있음
        self.tracker.update([make_person(1, 0.5, 0.5)])
        # 두 번째 프레임: 사라짐
        status = self.tracker.update([make_person(0, 0.5, 0.5)])
        assert status.just_disappeared is True

    def test_just_disappeared_false_on_second_missing_frame(self) -> None:
        # 첫 프레임: 있음
        self.tracker.update([make_person(1, 0.5, 0.5)])
        # 두 번째: 사라짐 (just_disappeared=True)
        self.tracker.update([make_person(0, 0.5, 0.5)])
        # 세 번째: 여전히 없음 → just_disappeared는 다시 False
        status = self.tracker.update([make_person(0, 0.5, 0.5)])
        assert status.just_disappeared is False

    def test_just_disappeared_false_after_reappear(self) -> None:
        self.tracker.update([make_person(1, 0.5, 0.5)])
        self.tracker.update([])  # 사라짐
        status = self.tracker.update([make_person(1, 0.5, 0.5)])  # 재등장
        assert status.is_in_frame is True
        assert status.just_disappeared is False

    def test_register_resets_disappearance_state(self) -> None:
        # 첫 등록 후 프레임에서 사라짐
        self.tracker.update([make_person(1, 0.5, 0.5)])
        self.tracker.update([])
        # 새 ID로 재등록 → just_disappeared 초기화
        self.tracker.register(2)
        status = self.tracker.update([])
        assert status.just_disappeared is False


# ---------------------------------------------------------------------------
# ProtectedPersonTracker — draw
# ---------------------------------------------------------------------------

class TestProtectedPersonTrackerDraw:
    def setup_method(self) -> None:
        self.tracker = ProtectedPersonTracker()
        self.tracker.register(0)

    def test_draw_returns_ndarray(self) -> None:
        status = ProtectedPersonStatus(
            person_id=0, is_in_frame=True,
            bbox=(0.2, 0.2, 0.8, 0.8), just_disappeared=False,
        )
        out = self.tracker.draw(blank_frame(64, 64), status)
        assert isinstance(out, np.ndarray)

    def test_draw_does_not_modify_original(self) -> None:
        frame = blank_frame(64, 64)
        original = frame.copy()
        status = ProtectedPersonStatus(
            person_id=0, is_in_frame=True,
            bbox=(0.1, 0.1, 0.9, 0.9), just_disappeared=False,
        )
        self.tracker.draw(frame, status)
        np.testing.assert_array_equal(frame, original)

    def test_draw_same_shape(self) -> None:
        frame = blank_frame(128, 64)
        status = ProtectedPersonStatus(
            person_id=0, is_in_frame=True,
            bbox=(0.1, 0.1, 0.9, 0.9), just_disappeared=False,
        )
        out = self.tracker.draw(frame, status)
        assert out.shape == frame.shape

    def test_draw_modifies_frame_when_in_frame(self) -> None:
        frame = blank_frame(64, 64)
        status = ProtectedPersonStatus(
            person_id=0, is_in_frame=True,
            bbox=(0.1, 0.1, 0.9, 0.9), just_disappeared=False,
        )
        out = self.tracker.draw(frame, status)
        # 초록색 픽셀이 생겨야 함 (원본은 전부 0)
        assert not np.array_equal(out, frame)

    def test_draw_unchanged_when_not_in_frame(self) -> None:
        frame = blank_frame(64, 64)
        status = ProtectedPersonStatus(
            person_id=0, is_in_frame=False,
            bbox=None, just_disappeared=True,
        )
        out = self.tracker.draw(frame, status)
        np.testing.assert_array_equal(out, frame)

    def test_draw_unchanged_when_bbox_none(self) -> None:
        frame = blank_frame(64, 64)
        status = ProtectedPersonStatus(
            person_id=0, is_in_frame=True,  # in_frame=True지만 bbox=None
            bbox=None, just_disappeared=False,
        )
        out = self.tracker.draw(frame, status)
        np.testing.assert_array_equal(out, frame)


# ---------------------------------------------------------------------------
# 통합: SOS 확정 → 등록 → 추적 → 이탈 시나리오
# ---------------------------------------------------------------------------

class TestProtectedPersonTrackerIntegration:
    def test_full_sos_to_disappearance_scenario(self) -> None:
        """SOS 확정 → 추적 → 이탈 전체 흐름."""
        tracker = ProtectedPersonTracker()

        # 1) SOS 확정: P2를 보호 대상으로 등록
        tracker.register(2)

        # 2) 다음 프레임: P2가 화면에 있음
        persons = [make_person(0, 0.2, 0.5), make_person(2, 0.5, 0.5)]
        status = tracker.update(persons)
        assert status.is_in_frame is True
        assert status.just_disappeared is False

        # 3) V사인 중단 후 여러 프레임: 여전히 P2 추적 중
        status = tracker.update([make_person(0, 0.2, 0.5), make_person(2, 0.55, 0.5)])
        assert status.is_in_frame is True

        # 4) P2가 화면 밖으로 나감
        status = tracker.update([make_person(0, 0.2, 0.5)])
        assert status.is_in_frame is False
        assert status.just_disappeared is True

        # 5) 다음 프레임도 없음 → just_disappeared=False (중복 경고 방지)
        status = tracker.update([make_person(0, 0.2, 0.5)])
        assert status.just_disappeared is False

    def test_reregister_after_disappearance(self) -> None:
        """이탈 후 다른 사람을 새 보호 대상으로 등록."""
        tracker = ProtectedPersonTracker()
        tracker.register(1)
        tracker.update([make_person(1, 0.5, 0.5)])
        tracker.update([])  # 이탈

        # 새 보호 대상 등록
        tracker.register(3)
        status = tracker.update([make_person(3, 0.6, 0.4)])
        assert status.person_id == 3
        assert status.is_in_frame is True
        assert status.just_disappeared is False
