"""src/api/state.py — 파이프라인 ↔ API 서버 공유 상태.

pipeline.py (동기 루프) 와 server.py (asyncio) 가 함께 접근하므로
threading.Lock 으로 스레드 안전성을 보장합니다.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PipelineStatus:
    """파이프라인 현재 상태 스냅샷."""

    threat_level: str = "NONE"
    threat_score: float = 0.0
    is_recording: bool = False
    protected_person_id: int | None = None
    is_protected_in_frame: bool = False
    sos_detected: bool = False
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class StatusBroadcaster:
    """파이프라인 상태를 스레드 안전하게 관리합니다.

    pipeline.py 는 ``update_sync()`` 로 상태를 갱신하고,
    FastAPI 핸들러는 ``get_status()`` 로 최신 상태를 읽습니다.
    """

    def __init__(self) -> None:
        self._status = PipelineStatus()
        self._lock = threading.Lock()

    def get_status(self) -> dict[str, Any]:
        """현재 상태 딕셔너리를 반환합니다 (스레드 안전)."""
        with self._lock:
            s = self._status
            return {
                "threat_level": s.threat_level,
                "threat_score": s.threat_score,
                "is_recording": s.is_recording,
                "protected_person_id": s.protected_person_id,
                "is_protected_in_frame": s.is_protected_in_frame,
                "sos_detected": s.sos_detected,
                "updated_at": s.updated_at,
            }

    def update_sync(self, **kwargs: Any) -> None:
        """동기 컨텍스트에서 상태를 갱신합니다 (스레드 안전).

        Args:
            **kwargs: PipelineStatus 필드명과 값.
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._status, key):
                    setattr(self._status, key, value)
            self._status.updated_at = datetime.now().isoformat()


# 모듈 레벨 싱글톤 — pipeline.py 와 server.py 가 공유
broadcaster: StatusBroadcaster = StatusBroadcaster()
