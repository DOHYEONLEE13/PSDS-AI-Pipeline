"""tests/test_api.py — FastAPI 서버 단위 테스트."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.api.state import StatusBroadcaster, broadcaster


@pytest.fixture(autouse=True)
def reset_broadcaster() -> None:
    """각 테스트 전에 broadcaster 상태를 초기화합니다."""
    broadcaster.update_sync(
        threat_level="NONE",
        threat_score=0.0,
        is_recording=False,
        protected_person_id=None,
        is_protected_in_frame=False,
        sos_detected=False,
    )


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ------------------------------------------------------------------
# StatusBroadcaster 단위 테스트
# ------------------------------------------------------------------


class TestStatusBroadcaster:
    def test_initial_state(self) -> None:
        b = StatusBroadcaster()
        s = b.get_status()
        assert s["threat_level"] == "NONE"
        assert s["threat_score"] == 0.0
        assert s["is_recording"] is False
        assert s["sos_detected"] is False

    def test_update_sync_changes_state(self) -> None:
        b = StatusBroadcaster()
        b.update_sync(threat_level="HIGH", threat_score=0.9)
        s = b.get_status()
        assert s["threat_level"] == "HIGH"
        assert s["threat_score"] == pytest.approx(0.9)

    def test_update_sync_ignores_unknown_keys(self) -> None:
        b = StatusBroadcaster()
        b.update_sync(nonexistent_key="value")  # 오류 없이 무시
        s = b.get_status()
        assert "nonexistent_key" not in s

    def test_updated_at_changes_on_update(self) -> None:
        b = StatusBroadcaster()
        t1 = b.get_status()["updated_at"]
        import time
        time.sleep(0.01)
        b.update_sync(threat_level="LOW")
        t2 = b.get_status()["updated_at"]
        assert t2 >= t1

    def test_thread_safety(self) -> None:
        """여러 스레드에서 동시에 업데이트해도 충돌이 없어야 합니다."""
        import threading
        b = StatusBroadcaster()
        errors: list[Exception] = []

        def update_many() -> None:
            try:
                for _ in range(100):
                    b.update_sync(threat_score=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ------------------------------------------------------------------
# GET /status
# ------------------------------------------------------------------


class TestGetStatus:
    def test_returns_200(self, client: TestClient) -> None:
        res = client.get("/status")
        assert res.status_code == 200

    def test_response_has_required_fields(self, client: TestClient) -> None:
        res = client.get("/status")
        data = res.json()
        for field in ("threat_level", "threat_score", "is_recording",
                      "protected_person_id", "is_protected_in_frame",
                      "sos_detected", "updated_at"):
            assert field in data

    def test_reflects_broadcaster_state(self, client: TestClient) -> None:
        broadcaster.update_sync(threat_level="CRITICAL", threat_score=0.95)
        res = client.get("/status")
        data = res.json()
        assert data["threat_level"] == "CRITICAL"
        assert data["threat_score"] == pytest.approx(0.95)

    def test_sos_detected_true(self, client: TestClient) -> None:
        broadcaster.update_sync(sos_detected=True)
        data = client.get("/status").json()
        assert data["sos_detected"] is True

    def test_protected_person_id(self, client: TestClient) -> None:
        broadcaster.update_sync(protected_person_id=3)
        data = client.get("/status").json()
        assert data["protected_person_id"] == 3


# ------------------------------------------------------------------
# GET /recordings
# ------------------------------------------------------------------


class TestGetRecordings:
    def test_returns_200(self, client: TestClient) -> None:
        res = client.get("/recordings")
        assert res.status_code == 200

    def test_empty_list_when_no_files(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        res = client.get("/recordings")
        data = res.json()
        assert data["count"] == 0
        assert data["recordings"] == []

    def test_lists_mp4_files(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        (tmp_path / "2026-01-01_00-00-01.mp4").write_bytes(b"fake")
        (tmp_path / "2026-01-01_00-00-02.mp4").write_bytes(b"fake")
        res = client.get("/recordings")
        data = res.json()
        assert data["count"] == 2

    def test_recording_entry_has_required_fields(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        (tmp_path / "2026-01-01_00-00-01.mp4").write_bytes(b"x" * 100)
        res = client.get("/recordings")
        entry = res.json()["recordings"][0]
        assert "filename" in entry
        assert "size_bytes" in entry
        assert "created_at" in entry
        assert entry["size_bytes"] == 100


# ------------------------------------------------------------------
# GET /recordings/{filename}
# ------------------------------------------------------------------


class TestDownloadRecording:
    def test_returns_404_for_missing_file(self, client: TestClient) -> None:
        res = client.get("/recordings/nonexistent.mp4")
        assert res.status_code == 404

    def test_returns_file_when_exists(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        filepath = tmp_path / "2026-01-01_00-00-01.mp4"
        filepath.write_bytes(b"fake video content")
        res = client.get("/recordings/2026-01-01_00-00-01.mp4")
        assert res.status_code == 200
        assert res.content == b"fake video content"

    def test_path_traversal_blocked(
        self, client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """경로 탈출 시도는 404 여야 합니다."""
        monkeypatch.setattr("src.api.server.RECORDINGS_DIR", tmp_path)
        res = client.get("/recordings/../../etc/passwd")
        assert res.status_code == 404


# ------------------------------------------------------------------
# WebSocket /ws/live
# ------------------------------------------------------------------


class TestWebSocketLive:
    def test_connects_successfully(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/live") as ws:
            data = ws.receive_json()
        assert "threat_level" in data

    def test_receives_current_state(self, client: TestClient) -> None:
        broadcaster.update_sync(threat_level="HIGH", threat_score=0.88)
        with client.websocket_connect("/ws/live") as ws:
            data = ws.receive_json()
        assert data["threat_level"] == "HIGH"
        assert data["threat_score"] == pytest.approx(0.88)

    def test_multiple_clients_independent(self, client: TestClient) -> None:
        """여러 클라이언트가 동시에 접속해도 독립적으로 동작해야 합니다."""
        broadcaster.update_sync(threat_level="MEDIUM")
        with (
            client.websocket_connect("/ws/live") as ws1,
            client.websocket_connect("/ws/live") as ws2,
        ):
            d1 = ws1.receive_json()
            d2 = ws2.receive_json()
        assert d1["threat_level"] == "MEDIUM"
        assert d2["threat_level"] == "MEDIUM"
