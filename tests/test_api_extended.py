"""tests/test_api_extended.py — 확장된 FastAPI 엔드포인트 테스트.

/frame, /alerts, DELETE /recordings/{filename},
GET /settings, PUT /settings 를 검증합니다.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.state import broadcaster


@pytest.fixture(autouse=True)
def reset_state() -> None:
    """각 테스트 전에 broadcaster 상태를 초기화합니다."""
    broadcaster.update_sync(
        threat_level="NONE",
        threat_score=0.0,
        is_recording=False,
        fps=0.0,
        inference_times={},
        sos_pending_duration=0.0,
        sos_detected=False,
        protected_person_id=None,
        is_protected_in_frame=False,
        protected_track_start=None,
    )
    broadcaster.clear_alerts()
    # settings 초기화
    broadcaster.update_settings(
        sos_hold_seconds=3.0,
        yolo_confidence=0.5,
        threat_medium_threshold=0.45,
        threat_high_threshold=0.75,
        record_trigger_level="MEDIUM",
    )


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ------------------------------------------------------------------
# GET /status — 새 필드 확인
# ------------------------------------------------------------------


class TestStatusExtended:
    def test_status_includes_fps(self, client: TestClient) -> None:
        broadcaster.update_sync(fps=24.5)
        data = client.get("/status").json()
        assert data["fps"] == pytest.approx(24.5)

    def test_status_includes_inference_times(self, client: TestClient) -> None:
        broadcaster.update_sync(inference_times={"손 추적": 5.2, "YOLO+위협": 30.1})
        data = client.get("/status").json()
        assert "손 추적" in data["inference_times"]

    def test_status_includes_sos_fields(self, client: TestClient) -> None:
        broadcaster.update_sync(sos_pending_duration=1.5, sos_hold_seconds=3.0)
        data = client.get("/status").json()
        assert data["sos_pending_duration"] == pytest.approx(1.5)
        assert data["sos_hold_seconds"] == pytest.approx(3.0)

    def test_status_includes_protected_track_start(self, client: TestClient) -> None:
        broadcaster.update_sync(protected_track_start="2026-03-23T10:00:00")
        data = client.get("/status").json()
        assert data["protected_track_start"] == "2026-03-23T10:00:00"


# ------------------------------------------------------------------
# GET /frame
# ------------------------------------------------------------------


class TestGetFrame:
    def test_returns_204_when_no_frame(self, client: TestClient) -> None:
        res = client.get("/frame")
        assert res.status_code == 204

    def test_returns_jpeg_when_frame_exists(self, client: TestClient) -> None:
        broadcaster.set_frame_jpg(b"\xff\xd8\xff fake jpeg data")
        res = client.get("/frame")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"
        assert res.content == b"\xff\xd8\xff fake jpeg data"

    def test_frame_content_type_is_jpeg(self, client: TestClient) -> None:
        broadcaster.set_frame_jpg(b"fake")
        res = client.get("/frame")
        assert "image/jpeg" in res.headers["content-type"]

    def test_frame_updates_after_set(self, client: TestClient) -> None:
        broadcaster.set_frame_jpg(b"frame1")
        assert client.get("/frame").content == b"frame1"
        broadcaster.set_frame_jpg(b"frame2")
        assert client.get("/frame").content == b"frame2"


# ------------------------------------------------------------------
# DELETE /recordings/{filename}
# ------------------------------------------------------------------


class TestDeleteRecording:
    def test_returns_404_for_missing_file(self, client: TestClient) -> None:
        res = client.delete("/recordings/nonexistent.mp4")
        assert res.status_code == 404

    def test_deletes_existing_file(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        filepath = tmp_path / "2026-01-01_00-00-01.mp4"
        filepath.write_bytes(b"fake")
        res = client.delete("/recordings/2026-01-01_00-00-01.mp4")
        assert res.status_code == 200
        assert not filepath.exists()

    def test_response_contains_deleted_filename(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        (tmp_path / "test.mp4").write_bytes(b"x")
        data = client.delete("/recordings/test.mp4").json()
        assert data["deleted"] == "test.mp4"

    def test_path_traversal_blocked(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        res = client.delete("/recordings/../../etc/passwd")
        assert res.status_code == 404


# ------------------------------------------------------------------
# GET /alerts
# ------------------------------------------------------------------


class TestGetAlerts:
    def test_returns_empty_initially(self, client: TestClient) -> None:
        data = client.get("/alerts").json()
        assert data["count"] == 0
        assert data["alerts"] == []

    def test_returns_added_alerts(self, client: TestClient) -> None:
        broadcaster.add_alert({"level": "HIGH", "timestamp": "2026-01-01T00:00:00"})
        broadcaster.add_alert({"level": "CRITICAL", "timestamp": "2026-01-01T00:00:01"})
        data = client.get("/alerts").json()
        assert data["count"] == 2

    def test_alerts_returned_newest_first(self, client: TestClient) -> None:
        broadcaster.add_alert({"order": 1})
        broadcaster.add_alert({"order": 2})
        alerts = client.get("/alerts").json()["alerts"]
        assert alerts[0]["order"] == 2  # 최신이 먼저

    def test_max_50_alerts(self, client: TestClient) -> None:
        for i in range(55):
            broadcaster.add_alert({"i": i})
        data = client.get("/alerts").json()
        assert data["count"] <= 50


# ------------------------------------------------------------------
# GET /settings
# ------------------------------------------------------------------


class TestGetSettings:
    def test_returns_200(self, client: TestClient) -> None:
        assert client.get("/settings").status_code == 200

    def test_has_default_fields(self, client: TestClient) -> None:
        data = client.get("/settings").json()
        for key in (
            "sos_hold_seconds",
            "yolo_confidence",
            "threat_medium_threshold",
            "threat_high_threshold",
            "record_trigger_level",
        ):
            assert key in data

    def test_default_sos_hold(self, client: TestClient) -> None:
        data = client.get("/settings").json()
        assert data["sos_hold_seconds"] == pytest.approx(3.0)


# ------------------------------------------------------------------
# PUT /settings
# ------------------------------------------------------------------


class TestPutSettings:
    def test_update_single_field(self, client: TestClient) -> None:
        res = client.put("/settings", json={"sos_hold_seconds": 5.0})
        assert res.status_code == 200
        assert res.json()["sos_hold_seconds"] == pytest.approx(5.0)

    def test_update_multiple_fields(self, client: TestClient) -> None:
        res = client.put(
            "/settings",
            json={"yolo_confidence": 0.7, "record_trigger_level": "HIGH"},
        )
        data = res.json()
        assert data["yolo_confidence"] == pytest.approx(0.7)
        assert data["record_trigger_level"] == "HIGH"

    def test_unknown_key_is_ignored(self, client: TestClient) -> None:
        res = client.put("/settings", json={"unknown_key": "value"})
        assert res.status_code == 200
        assert "unknown_key" not in res.json()

    def test_persisted_after_update(self, client: TestClient) -> None:
        client.put("/settings", json={"sos_hold_seconds": 7.0})
        data = client.get("/settings").json()
        assert data["sos_hold_seconds"] == pytest.approx(7.0)


# ------------------------------------------------------------------
# StatusBroadcaster 추가 기능
# ------------------------------------------------------------------


class TestBroadcasterExtended:
    def test_set_and_get_frame_jpg(self) -> None:
        broadcaster.set_frame_jpg(b"test_frame")
        assert broadcaster.get_frame_jpg() == b"test_frame"

    def test_get_frame_jpg_none_initially(self) -> None:
        # reset 픽스처가 clear하지 않으므로 직접 None 설정
        broadcaster._frame_jpg = None  # noqa: SLF001
        assert broadcaster.get_frame_jpg() is None

    def test_add_and_get_alerts(self) -> None:
        broadcaster.add_alert({"level": "HIGH"})
        alerts = broadcaster.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]["level"] == "HIGH"

    def test_clear_alerts(self) -> None:
        broadcaster.add_alert({"level": "HIGH"})
        broadcaster.clear_alerts()
        assert broadcaster.get_alerts() == []

    def test_alert_history_max_50(self) -> None:
        for i in range(60):
            broadcaster.add_alert({"i": i})
        assert len(broadcaster.get_alerts()) == 50

    def test_get_settings_returns_copy(self) -> None:
        s1 = broadcaster.get_settings()
        s1["sos_hold_seconds"] = 999.0
        s2 = broadcaster.get_settings()
        assert s2["sos_hold_seconds"] != 999.0

    def test_update_settings_ignores_unknown(self) -> None:
        broadcaster.update_settings(nonexistent=42)
        assert "nonexistent" not in broadcaster.get_settings()
