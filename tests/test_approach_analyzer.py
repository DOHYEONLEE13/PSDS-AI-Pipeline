"""tests/test_approach_analyzer.py — ApproachAnalyzer 테스트."""
from __future__ import annotations

import pytest

from src.threat_detection.approach_analyzer import ApproachAnalyzer, ApproachEvent
from tests.conftest import make_person

# ---------------------------------------------------------------------------
# ApproachEvent
# ---------------------------------------------------------------------------

class TestApproachEvent:
    def test_fields_accessible(self) -> None:
        e = ApproachEvent(aggressor_id=0, target_id=1, speed=0.3, is_toward_protected=True)
        assert e.aggressor_id == 0
        assert e.target_id == 1
        assert e.speed == pytest.approx(0.3)
        assert e.is_toward_protected is True

    def test_not_protected_by_default(self) -> None:
        e = ApproachEvent(aggressor_id=0, target_id=1, speed=0.1, is_toward_protected=False)
        assert not e.is_toward_protected


# ---------------------------------------------------------------------------
# ApproachAnalyzer — 기본 동작
# ---------------------------------------------------------------------------

class TestApproachAnalyzerBasic:
    def setup_method(self) -> None:
        self.analyzer = ApproachAnalyzer(fast_threshold=0.1)

    def test_single_person_no_events(self) -> None:
        self.analyzer.update([make_person(0, 0.5, 0.5)], timestamp=0.0)
        events = self.analyzer.update([make_person(0, 0.5, 0.5)], timestamp=1.0)
        assert events == []

    def test_empty_persons_no_events(self) -> None:
        events = self.analyzer.update([], timestamp=0.0)
        assert events == []

    def test_first_frame_two_persons_no_events(self) -> None:
        # 이력 1개만 있으면 속도 계산 불가
        events = self.analyzer.update(
            [make_person(0, 0.2, 0.5), make_person(1, 0.8, 0.5)], timestamp=0.0
        )
        assert events == []

    def test_two_persons_approaching_returns_event(self) -> None:
        # 프레임1: 0.2, 0.8 → 거리 0.6
        # 프레임2: 0.35, 0.65 → 거리 0.3, dt=1.0 → speed=0.3
        self.analyzer.update(
            [make_person(0, 0.2, 0.5), make_person(1, 0.8, 0.5)], timestamp=0.0
        )
        events = self.analyzer.update(
            [make_person(0, 0.35, 0.5), make_person(1, 0.65, 0.5)], timestamp=1.0
        )
        assert len(events) == 1
        assert events[0].speed == pytest.approx(0.3, abs=1e-3)

    def test_two_persons_moving_away_no_event(self) -> None:
        # 거리가 증가 → 음수 speed → 이벤트 없음
        self.analyzer.update(
            [make_person(0, 0.4, 0.5), make_person(1, 0.6, 0.5)], timestamp=0.0
        )
        events = self.analyzer.update(
            [make_person(0, 0.2, 0.5), make_person(1, 0.8, 0.5)], timestamp=1.0
        )
        assert events == []

    def test_stationary_persons_no_event(self) -> None:
        pos = [make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)]
        self.analyzer.update(pos, timestamp=0.0)
        events = self.analyzer.update(pos, timestamp=1.0)
        assert events == []

    def test_speed_proportional_to_delta(self) -> None:
        # dt=2.0 → speed = 동일 이동 / 2.0
        a = ApproachAnalyzer(fast_threshold=0.05)
        a.update([make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)], timestamp=0.0)
        events = a.update(
            [make_person(0, 0.2, 0.5), make_person(1, 0.8, 0.5)], timestamp=2.0
        )
        # Person 0: prev=(0.1,0.5) → curr=(0.2,0.5)
        # Person 1: prev=(0.9,0.5) → curr=(0.8,0.5)
        # dist_prev=0.8, dist_curr=0.6, dt=2.0 → speed=(0.8-0.6)/2.0=0.1
        assert len(events) == 1
        assert events[0].speed == pytest.approx(0.1, abs=0.01)


# ---------------------------------------------------------------------------
# ApproachAnalyzer — 보호 대상
# ---------------------------------------------------------------------------

class TestApproachAnalyzerProtected:
    def setup_method(self) -> None:
        self.analyzer = ApproachAnalyzer(fast_threshold=0.05)

    def test_event_toward_protected_flagged(self) -> None:
        self.analyzer.update(
            [make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)],
            timestamp=0.0, protected_id=1,
        )
        events = self.analyzer.update(
            [make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)],
            timestamp=1.0, protected_id=1,
        )
        assert len(events) == 1
        assert events[0].is_toward_protected is True
        assert events[0].target_id == 1
        assert events[0].aggressor_id == 0

    def test_event_not_flagged_when_no_protected(self) -> None:
        self.analyzer.update(
            [make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)],
            timestamp=0.0,
        )
        events = self.analyzer.update(
            [make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)],
            timestamp=1.0,
        )
        assert len(events) == 1
        assert events[0].is_toward_protected is False

    def test_protected_as_aggressor_correctly_assigned(self) -> None:
        # protected=0 이 P1 쪽으로 접근 → 그래도 P0이 aggressor, P1이 target
        self.analyzer.update(
            [make_person(0, 0.9, 0.5), make_person(1, 0.1, 0.5)],
            timestamp=0.0, protected_id=0,
        )
        events = self.analyzer.update(
            [make_person(0, 0.7, 0.5), make_person(1, 0.1, 0.5)],
            timestamp=1.0, protected_id=0,
        )
        # P0이 P1 쪽으로 이동하더라도 P0이 보호 대상이므로
        # P1이 aggressor, P0이 target 으로 간주
        if events:
            assert events[0].target_id == 0
            assert events[0].is_toward_protected is True

    def test_three_persons_only_approach_to_protected_flagged(self) -> None:
        # P0(공격자)이 P2(보호 대상)에게 접근, P1은 정지
        self.analyzer.update(
            [make_person(0, 0.1, 0.5), make_person(1, 0.5, 0.5), make_person(2, 0.9, 0.5)],
            timestamp=0.0, protected_id=2,
        )
        events = self.analyzer.update(
            [make_person(0, 0.3, 0.5), make_person(1, 0.5, 0.5), make_person(2, 0.9, 0.5)],
            timestamp=1.0, protected_id=2,
        )
        protected_events = [e for e in events if e.is_toward_protected]
        assert len(protected_events) >= 1
        assert all(e.target_id == 2 for e in protected_events)


# ---------------------------------------------------------------------------
# ApproachAnalyzer — 위협 점수
# ---------------------------------------------------------------------------

class TestApproachAnalyzerThreatScore:
    def test_empty_events_score_zero(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1)
        assert a.threat_score([]) == pytest.approx(0.0)

    def test_below_threshold_score_zero(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.5)
        events = [ApproachEvent(0, 1, speed=0.1, is_toward_protected=False)]
        assert a.threat_score(events) == pytest.approx(0.0)

    def test_above_threshold_positive_score(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1)
        events = [ApproachEvent(0, 1, speed=0.3, is_toward_protected=False)]
        score = a.threat_score(events)
        assert score > 0.0

    def test_score_capped_at_one(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1)
        events = [ApproachEvent(0, 1, speed=999.0, is_toward_protected=False)]
        assert a.threat_score(events) <= 1.0

    def test_protected_approach_score_higher(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1)
        speed = 0.3
        normal = [ApproachEvent(0, 1, speed=speed, is_toward_protected=False)]
        protected = [ApproachEvent(0, 1, speed=speed, is_toward_protected=True)]
        assert a.threat_score(protected) > a.threat_score(normal)

    def test_multiple_events_max_score(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1)
        events = [
            ApproachEvent(0, 1, speed=0.2, is_toward_protected=False),
            ApproachEvent(0, 2, speed=0.8, is_toward_protected=False),
        ]
        score = a.threat_score(events)
        single = a.threat_score([ApproachEvent(0, 2, speed=0.8, is_toward_protected=False)])
        assert score == pytest.approx(single)


# ---------------------------------------------------------------------------
# ApproachAnalyzer — 이력 제거 및 reset
# ---------------------------------------------------------------------------

class TestApproachAnalyzerHistory:
    def test_disappeared_person_history_removed(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1)
        a.update([make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)], timestamp=0.0)
        # P1이 사라짐
        a.update([make_person(0, 0.3, 0.5)], timestamp=1.0)
        assert 1 not in a._history

    def test_reset_clears_all_history(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1)
        a.update([make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)], timestamp=0.0)
        a.reset()
        assert a._history == {}

    def test_reset_then_no_events_on_next_frame(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.05)
        a.update([make_person(0, 0.1, 0.5), make_person(1, 0.9, 0.5)], timestamp=0.0)
        a.reset()
        # 리셋 후 첫 프레임 → 이력 없어서 이벤트 없음
        events = a.update(
            [make_person(0, 0.3, 0.5), make_person(1, 0.7, 0.5)], timestamp=1.0
        )
        assert events == []

    def test_history_len_caps_entries(self) -> None:
        a = ApproachAnalyzer(fast_threshold=0.1, history_len=3)
        for i in range(10):
            a.update([make_person(0, 0.5, 0.5)], timestamp=float(i))
        assert len(a._history[0]) <= 3
