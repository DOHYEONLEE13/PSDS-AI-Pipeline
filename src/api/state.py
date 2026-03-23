"""src/api/state.py — 파이프라인 ↔ API 서버 공유 상태.

pipeline.py (동기 루프) 와 server.py (asyncio) 가 함께 접근하므로
threading.Lock 으로 스레드 안전성을 보장합니다.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_MAX_ALERT_HISTORY = 50  # 최대 알림 이력 건수

_DEFAULT_SETTINGS: dict[str, Any] = {
    "sos_hold_seconds": 3.0,
    "yolo_confidence": 0.5,
    "threat_medium_threshold": 0.45,
    "threat_high_threshold": 0.75,
    "record_trigger_level": "MEDIUM",
}


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
    # 추가 필드
    fps: float = 0.0
    inference_times: dict[str, float] = field(default_factory=dict)
    sos_pending_duration: float = 0.0
    sos_hold_seconds: float = 3.0
    protected_track_start: str | None = None


class StatusBroadcaster:
    """파이프라인 상태를 스레드 안전하게 관리합니다.

    pipeline.py 는 ``update_sync()`` 로 상태를 갱신하고,
    FastAPI 핸들러는 ``get_status()`` 로 최신 상태를 읽습니다.
    """

    def __init__(self) -> None:
        self._status = PipelineStatus()
        self._lock = threading.Lock()
        self._frame_jpg: bytes | None = None
        self._alert_history: list[dict[str, Any]] = []
        self._settings: dict[str, Any] = _DEFAULT_SETTINGS.copy()

    # ------------------------------------------------------------------
    # 상태 접근
    # ------------------------------------------------------------------

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
                "fps": s.fps,
                "inference_times": dict(s.inference_times),
                "sos_pending_duration": s.sos_pending_duration,
                "sos_hold_seconds": s.sos_hold_seconds,
                "protected_track_start": s.protected_track_start,
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

    # ------------------------------------------------------------------
    # 최신 프레임 JPEG
    # ------------------------------------------------------------------

    def get_frame_jpg(self) -> bytes | None:
        """최신 처리 프레임 JPEG 바이트를 반환합니다."""
        with self._lock:
            return self._frame_jpg

    def set_frame_jpg(self, data: bytes) -> None:
        """최신 프레임 JPEG 를 갱신합니다."""
        with self._lock:
            self._frame_jpg = data

    # ------------------------------------------------------------------
    # 알림 이력
    # ------------------------------------------------------------------

    def add_alert(self, alert: dict[str, Any]) -> None:
        """알림 이력에 새 항목을 추가합니다 (최대 50건 유지)."""
        with self._lock:
            self._alert_history.append(alert)
            if len(self._alert_history) > _MAX_ALERT_HISTORY:
                self._alert_history.pop(0)

    def get_alerts(self) -> list[dict[str, Any]]:
        """알림 이력 목록을 반환합니다 (최신순)."""
        with self._lock:
            return list(reversed(self._alert_history))

    def clear_alerts(self) -> None:
        """알림 이력을 전부 삭제합니다."""
        with self._lock:
            self._alert_history.clear()

    # ------------------------------------------------------------------
    # 설정
    # ------------------------------------------------------------------

    def get_settings(self) -> dict[str, Any]:
        """현재 설정을 반환합니다."""
        with self._lock:
            return dict(self._settings)

    def update_settings(self, **kwargs: Any) -> None:
        """설정을 갱신합니다. 알 수 없는 키는 무시합니다."""
        with self._lock:
            for key, value in kwargs.items():
                if key in self._settings:
                    self._settings[key] = value


# 모듈 레벨 싱글톤 — pipeline.py 와 server.py 가 공유
broadcaster: StatusBroadcaster = StatusBroadcaster()
