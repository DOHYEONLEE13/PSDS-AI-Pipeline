from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from src.threat_detection.detector import ThreatLevel, ThreatResult

logger = logging.getLogger(__name__)


class ProtectionAction(Enum):
    NONE = auto()
    LOG = auto()
    ALERT = auto()
    BLOCK = auto()
    EMERGENCY = auto()


@dataclass
class ProtectionResponse:
    action: ProtectionAction
    threat: ThreatResult
    message: str


# 위협 수준 → 기본 액션 매핑
_DEFAULT_POLICY: dict[ThreatLevel, ProtectionAction] = {
    ThreatLevel.NONE: ProtectionAction.NONE,
    ThreatLevel.LOW: ProtectionAction.LOG,
    ThreatLevel.MEDIUM: ProtectionAction.ALERT,
    ThreatLevel.HIGH: ProtectionAction.BLOCK,
    ThreatLevel.CRITICAL: ProtectionAction.EMERGENCY,
}


class Protector:
    """위협 감지 결과에 따라 보호 액션을 실행합니다."""

    def __init__(
        self,
        policy: dict[ThreatLevel, ProtectionAction] | None = None,
        on_alert: Callable[[ProtectionResponse], None] | None = None,
        on_block: Callable[[ProtectionResponse], None] | None = None,
    ) -> None:
        self._policy = policy or _DEFAULT_POLICY.copy()
        self._on_alert = on_alert
        self._on_block = on_block

    def respond(self, threat: ThreatResult) -> ProtectionResponse:
        """위협 결과를 받아 보호 액션을 결정하고 실행합니다."""
        action = self._policy.get(threat.level, ProtectionAction.NONE)
        message = self._build_message(action, threat)
        response = ProtectionResponse(action=action, threat=threat, message=message)
        self._dispatch(response)
        return response

    def set_policy(self, level: ThreatLevel, action: ProtectionAction) -> None:
        """특정 위협 수준의 정책을 변경합니다."""
        self._policy[level] = action

    def _dispatch(self, response: ProtectionResponse) -> None:
        action = response.action
        if action == ProtectionAction.LOG:
            logger.info("[PROTECTION] %s", response.message)
        elif action == ProtectionAction.ALERT:
            logger.warning("[ALERT] %s", response.message)
            if self._on_alert:
                self._on_alert(response)
        elif action in (ProtectionAction.BLOCK, ProtectionAction.EMERGENCY):
            logger.error("[BLOCK] %s", response.message)
            if self._on_block:
                self._on_block(response)

    @staticmethod
    def _build_message(action: ProtectionAction, threat: ThreatResult) -> str:
        reasons = "; ".join(threat.reasons) if threat.reasons else "없음"
        return (
            f"액션={action.name} | 위협수준={threat.level.name} "
            f"| 점수={threat.score:.2f} | 사유={reasons}"
        )
