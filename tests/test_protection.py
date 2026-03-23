"""tests/test_protection.py — Protector 기본 테스트."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.protection.protector import (
    ProtectionAction,
    ProtectionResponse,
    Protector,
)
from src.threat_detection.detector import ThreatLevel
from tests.conftest import make_threat_result


class TestProtectorDefaultPolicy:
    def setup_method(self) -> None:
        self.protector = Protector()

    @pytest.mark.parametrize("level,expected_action", [
        (ThreatLevel.NONE, ProtectionAction.NONE),
        (ThreatLevel.LOW, ProtectionAction.LOG),
        (ThreatLevel.MEDIUM, ProtectionAction.ALERT),
        (ThreatLevel.HIGH, ProtectionAction.BLOCK),
        (ThreatLevel.CRITICAL, ProtectionAction.EMERGENCY),
    ])
    def test_default_policy_mapping(
        self, level: ThreatLevel, expected_action: ProtectionAction
    ) -> None:
        threat = make_threat_result(level=level, score=0.5)
        response = self.protector.respond(threat)
        assert response.action == expected_action

    def test_respond_returns_protection_response(self) -> None:
        threat = make_threat_result(level=ThreatLevel.NONE)
        response = self.protector.respond(threat)
        assert isinstance(response, ProtectionResponse)
        assert response.threat is threat

    def test_response_message_contains_level_and_action(self) -> None:
        threat = make_threat_result(level=ThreatLevel.HIGH, score=0.8)
        response = self.protector.respond(threat)
        assert "HIGH" in response.message
        assert "BLOCK" in response.message

    def test_response_message_contains_score(self) -> None:
        threat = make_threat_result(level=ThreatLevel.MEDIUM, score=0.55)
        response = self.protector.respond(threat)
        assert "0.55" in response.message

    def test_response_message_contains_reasons(self) -> None:
        threat = make_threat_result(
            level=ThreatLevel.MEDIUM, reasons=["위협 제스처 감지: FIST"]
        )
        response = self.protector.respond(threat)
        assert "FIST" in response.message

    def test_response_message_shows_없음_when_no_reasons(self) -> None:
        threat = make_threat_result(level=ThreatLevel.NONE, reasons=[])
        response = self.protector.respond(threat)
        assert "없음" in response.message


class TestProtectorCallbacks:
    def test_on_alert_called_for_medium(self) -> None:
        callback = MagicMock()
        protector = Protector(on_alert=callback)
        threat = make_threat_result(level=ThreatLevel.MEDIUM)
        protector.respond(threat)
        callback.assert_called_once()
        assert callback.call_args[0][0].action == ProtectionAction.ALERT

    def test_on_block_called_for_high(self) -> None:
        callback = MagicMock()
        protector = Protector(on_block=callback)
        threat = make_threat_result(level=ThreatLevel.HIGH)
        protector.respond(threat)
        callback.assert_called_once()

    def test_on_block_called_for_critical(self) -> None:
        callback = MagicMock()
        protector = Protector(on_block=callback)
        threat = make_threat_result(level=ThreatLevel.CRITICAL)
        protector.respond(threat)
        callback.assert_called_once()

    def test_on_alert_not_called_for_none(self) -> None:
        callback = MagicMock()
        protector = Protector(on_alert=callback)
        protector.respond(make_threat_result(level=ThreatLevel.NONE))
        callback.assert_not_called()

    def test_callbacks_not_required(self) -> None:
        """콜백 없이도 예외 없이 동작해야 한다."""
        protector = Protector()
        protector.respond(make_threat_result(level=ThreatLevel.HIGH))


class TestProtectorSetPolicy:
    def test_set_policy_overrides_default(self) -> None:
        protector = Protector()
        protector.set_policy(ThreatLevel.LOW, ProtectionAction.BLOCK)
        response = protector.respond(make_threat_result(level=ThreatLevel.LOW))
        assert response.action == ProtectionAction.BLOCK

    def test_custom_policy_at_init(self) -> None:
        policy = {ThreatLevel.NONE: ProtectionAction.ALERT}
        protector = Protector(policy=policy)
        response = protector.respond(make_threat_result(level=ThreatLevel.NONE))
        assert response.action == ProtectionAction.ALERT
