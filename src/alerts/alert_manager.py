"""src/alerts/alert_manager.py — 위협 레벨 3 경찰 신고 시뮬레이션.

위협 레벨이 HIGH 또는 CRITICAL 로 상승할 때 경찰 신고를 시뮬레이션합니다.
콘솔에 경고를 출력하고 alerts/ 폴더에 JSON 증거 파일을 저장합니다.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.threat_detection.detector import ThreatLevel, ThreatResult

logger = logging.getLogger(__name__)

# 이 레벨 이상이면 경찰 신고 시뮬레이션
_ALERT_LEVELS = {ThreatLevel.HIGH, ThreatLevel.CRITICAL}


class AlertManager:
    """위협 레벨 3 이상 시 경찰 신고 시뮬레이션.

    동일 레벨이 연속되면 중복 신고하지 않습니다.
    레벨이 더 높아질 때마다 새로 신고합니다.

    Args:
        alerts_dir: JSON 파일 저장 폴더.
    """

    def __init__(self, alerts_dir: str | Path = "alerts") -> None:
        self._dir = Path(alerts_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._last_alerted_level: ThreatLevel = ThreatLevel.NONE

    def handle_threat(self, threat: ThreatResult) -> bool:
        """위협 결과를 처리합니다.

        레벨이 HIGH/CRITICAL 이고 이전보다 높아졌을 때만 신고합니다.

        Args:
            threat: 현재 위협 결과.

        Returns:
            신고가 실행됐으면 True.
        """
        if threat.level not in _ALERT_LEVELS:
            # 위협 해제 시 이전 레벨 초기화
            if self._last_alerted_level in _ALERT_LEVELS:
                self._last_alerted_level = ThreatLevel.NONE
            return False

        # 같은 레벨이면 중복 신고 방지
        if threat.level == self._last_alerted_level:
            return False

        self._last_alerted_level = threat.level
        self._simulate_police_call(threat)
        return True

    def list_alerts(self) -> list[Path]:
        """저장된 알림 JSON 파일 목록을 시간 내림차순으로 반환합니다."""
        return sorted(
            self._dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _simulate_police_call(self, threat: ThreatResult) -> None:
        """경찰 신고 시뮬레이션: 콘솔 출력 + JSON 파일 저장."""
        ts = datetime.now()
        reasons_str = ", ".join(threat.reasons) if threat.reasons else "없음"

        print(f"\n{'=' * 52}")
        print("경찰 신고 시뮬레이션")
        print(f"  시각     : {ts.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  위협 레벨: {threat.level.name} (score={threat.score:.2f})")
        print(f"  사유     : {reasons_str}")
        print(f"{'=' * 52}\n")

        filename = ts.strftime("%Y-%m-%d_%H-%M-%S-%f") + f"_{threat.level.name}.json"
        filepath = self._dir / filename
        payload = {
            "timestamp": ts.isoformat(),
            "level": threat.level.name,
            "score": threat.score,
            "reasons": threat.reasons,
            "action": "경찰 신고 시뮬레이션",
        }
        try:
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            logger.info("알림 JSON 저장: %s", filepath.name)
        except OSError as exc:
            logger.error("알림 JSON 저장 실패: %s", exc)
