"""tests/test_alert_manager.py — AlertManager 단위 테스트."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.alerts.alert_manager import AlertManager
from src.threat_detection.detector import ThreatLevel, ThreatResult


def make_threat(level: ThreatLevel, score: float = 0.8) -> ThreatResult:
    return ThreatResult(level=level, score=score, reasons=["테스트 사유"])


@pytest.fixture()
def manager(tmp_path: Path) -> AlertManager:
    return AlertManager(alerts_dir=tmp_path)


class TestAlertManagerInit:
    def test_alerts_dir_created(self, tmp_path: Path) -> None:
        d = tmp_path / "alerts"
        AlertManager(alerts_dir=d)
        assert d.exists()

    def test_list_alerts_empty_initially(self, manager: AlertManager) -> None:
        assert manager.list_alerts() == []


class TestHandleThreat:
    def test_no_alert_on_none(self, manager: AlertManager) -> None:
        result = manager.handle_threat(make_threat(ThreatLevel.NONE))
        assert result is False
        assert manager.list_alerts() == []

    def test_no_alert_on_low(self, manager: AlertManager) -> None:
        result = manager.handle_threat(make_threat(ThreatLevel.LOW))
        assert result is False

    def test_no_alert_on_medium(self, manager: AlertManager) -> None:
        result = manager.handle_threat(make_threat(ThreatLevel.MEDIUM))
        assert result is False

    def test_alert_on_high(self, manager: AlertManager, capsys) -> None:
        result = manager.handle_threat(make_threat(ThreatLevel.HIGH))
        assert result is True
        captured = capsys.readouterr()
        assert "경찰 신고 시뮬레이션" in captured.out

    def test_alert_on_critical(self, manager: AlertManager, capsys) -> None:
        result = manager.handle_threat(make_threat(ThreatLevel.CRITICAL))
        assert result is True
        captured = capsys.readouterr()
        assert "경찰 신고 시뮬레이션" in captured.out

    def test_json_saved_on_high(self, manager: AlertManager) -> None:
        manager.handle_threat(make_threat(ThreatLevel.HIGH))
        alerts = manager.list_alerts()
        assert len(alerts) == 1
        data = json.loads(alerts[0].read_text(encoding="utf-8"))
        assert data["level"] == "HIGH"
        assert data["action"] == "경찰 신고 시뮬레이션"

    def test_json_saved_on_critical(self, manager: AlertManager) -> None:
        manager.handle_threat(make_threat(ThreatLevel.CRITICAL))
        alerts = manager.list_alerts()
        assert len(alerts) == 1
        data = json.loads(alerts[0].read_text(encoding="utf-8"))
        assert data["level"] == "CRITICAL"


class TestDuplicatePrevention:
    def test_no_duplicate_on_same_level(self, manager: AlertManager) -> None:
        manager.handle_threat(make_threat(ThreatLevel.HIGH))
        manager.handle_threat(make_threat(ThreatLevel.HIGH))
        assert len(manager.list_alerts()) == 1

    def test_new_alert_on_escalation(self, manager: AlertManager) -> None:
        manager.handle_threat(make_threat(ThreatLevel.HIGH))
        manager.handle_threat(make_threat(ThreatLevel.CRITICAL))
        assert len(manager.list_alerts()) == 2

    def test_alert_again_after_level_reset(self, manager: AlertManager) -> None:
        """위협 해제 후 다시 HIGH 가 되면 재신고해야 합니다."""
        manager.handle_threat(make_threat(ThreatLevel.HIGH))
        manager.handle_threat(make_threat(ThreatLevel.NONE))   # 해제
        manager.handle_threat(make_threat(ThreatLevel.HIGH))   # 재발
        assert len(manager.list_alerts()) == 2


class TestJsonContent:
    def test_json_contains_reasons(self, manager: AlertManager) -> None:
        manager.handle_threat(
            ThreatResult(level=ThreatLevel.CRITICAL, score=0.9, reasons=["빠른 접근", "주먹"])
        )
        data = json.loads(manager.list_alerts()[0].read_text(encoding="utf-8"))
        assert "빠른 접근" in data["reasons"]
        assert "주먹" in data["reasons"]

    def test_json_contains_score(self, manager: AlertManager) -> None:
        manager.handle_threat(make_threat(ThreatLevel.HIGH, score=0.77))
        data = json.loads(manager.list_alerts()[0].read_text(encoding="utf-8"))
        assert abs(data["score"] - 0.77) < 0.01

    def test_json_filename_contains_level(self, manager: AlertManager) -> None:
        manager.handle_threat(make_threat(ThreatLevel.CRITICAL))
        filename = manager.list_alerts()[0].name
        assert "CRITICAL" in filename
