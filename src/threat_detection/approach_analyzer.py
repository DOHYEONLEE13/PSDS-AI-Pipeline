"""src/threat_detection/approach_analyzer.py — 인물 간 접근 속도 분석."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from src.threat_detection.yolo_detector import PersonDetection


@dataclass
class ApproachEvent:
    """두 인물 사이 접근 이벤트.

    Attributes:
        aggressor_id: 접근하는 쪽 인물 ID.
        target_id: 접근 당하는 쪽 인물 ID.
        speed: 접근 속도 (정규화 단위/초). 양수 = 거리 감소 중.
        is_toward_protected: target이 보호 대상이면 True.
    """

    aggressor_id: int
    target_id: int
    speed: float
    is_toward_protected: bool


class ApproachAnalyzer:
    """인물들의 위치 이력을 추적하여 빠른 접근을 감지합니다.

    두 인물 사이 거리가 ``fast_threshold`` 이상의 속도로 줄어들면
    ApproachEvent를 생성합니다. 보호 대상을 향한 접근은 위협 점수에
    1.5배 가중치를 적용합니다.

    Args:
        fast_threshold: 위협으로 간주하는 최소 접근 속도 (정규화 단위/초).
        history_len: 각 인물의 위치 이력 최대 길이.
    """

    def __init__(
        self,
        fast_threshold: float = 0.1,
        history_len: int = 5,
    ) -> None:
        self.fast_threshold = fast_threshold
        self._history_len = history_len
        # person_id -> deque[(timestamp, (cx, cy))]
        self._history: dict[int, deque[tuple[float, tuple[float, float]]]] = {}

    def update(
        self,
        persons: list[PersonDetection],
        timestamp: float,
        protected_id: int | None = None,
    ) -> list[ApproachEvent]:
        """현재 프레임 위치를 업데이트하고 접근 이벤트 목록을 반환합니다.

        Args:
            persons: 현재 프레임의 PersonDetection 목록.
            timestamp: 현재 시각 (monotonic 기준).
            protected_id: 보호 대상 인물 ID. None이면 보호 대상 없음.

        Returns:
            이번 프레임에서 감지된 ApproachEvent 목록.
        """
        current_ids = {d.person_id for d in persons}

        # 더 이상 화면에 없는 인물 이력 제거
        for stale in [pid for pid in self._history if pid not in current_ids]:
            del self._history[stale]

        for det in persons:
            pid = det.person_id
            if pid not in self._history:
                self._history[pid] = deque(maxlen=self._history_len)
            self._history[pid].append((timestamp, det.center))

        return self._compute_events(sorted(current_ids), protected_id)

    def threat_score(self, events: list[ApproachEvent]) -> float:
        """접근 이벤트 목록에서 위협 점수(0.0~1.0)를 계산합니다.

        ``fast_threshold`` 미만의 접근 속도는 무시합니다.
        보호 대상을 향한 접근은 1.5배 가중치를 적용합니다.

        Args:
            events: update()가 반환한 ApproachEvent 목록.

        Returns:
            0.0 ~ 1.0 범위의 위협 점수.
        """
        scores: list[float] = []
        for e in events:
            if e.speed < self.fast_threshold:
                continue
            base = min(e.speed / (self.fast_threshold * 5.0), 1.0)
            if e.is_toward_protected:
                base = min(base * 1.5, 1.0)
            scores.append(base)
        return max(scores, default=0.0)

    def reset(self) -> None:
        """위치 이력을 초기화합니다."""
        self._history.clear()

    def _compute_events(
        self,
        ids: list[int],
        protected_id: int | None,
    ) -> list[ApproachEvent]:
        events: list[ApproachEvent] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                speed = self._approach_speed(id_a, id_b)
                if speed <= 0.0:
                    continue
                # 보호 대상이 있으면 해당 방향으로 aggressor/target 결정
                if protected_id is not None and id_b == protected_id:
                    aggressor, target = id_a, id_b
                elif protected_id is not None and id_a == protected_id:
                    aggressor, target = id_b, id_a
                else:
                    aggressor, target = id_a, id_b
                events.append(ApproachEvent(
                    aggressor_id=aggressor,
                    target_id=target,
                    speed=speed,
                    is_toward_protected=(protected_id is not None and target == protected_id),
                ))
        return events

    def _approach_speed(self, id_a: int, id_b: int) -> float:
        """두 인물 사이 접근 속도를 반환합니다. 양수 = 거리 줄어드는 중."""
        hist_a = self._history.get(id_a)
        hist_b = self._history.get(id_b)
        if not hist_a or not hist_b or len(hist_a) < 2 or len(hist_b) < 2:
            return 0.0

        t_prev_a, c_prev_a = hist_a[-2]
        t_curr_a, c_curr_a = hist_a[-1]
        t_prev_b, c_prev_b = hist_b[-2]
        t_curr_b, c_curr_b = hist_b[-1]

        t_prev = (t_prev_a + t_prev_b) / 2.0
        t_curr = (t_curr_a + t_curr_b) / 2.0
        dt = t_curr - t_prev
        if dt <= 0.0:
            return 0.0

        prev_dist = math.dist(c_prev_a, c_prev_b)
        curr_dist = math.dist(c_curr_a, c_curr_b)
        return (prev_dist - curr_dist) / dt
