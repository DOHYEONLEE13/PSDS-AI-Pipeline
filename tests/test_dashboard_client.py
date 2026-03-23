"""tests/test_dashboard_client.py — API 클라이언트 단위 테스트.

httpx 호출을 mock 하여 서버 없이 테스트합니다.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from src.dashboard import api_client


def _mock_response(json_data=None, content=None, status_code=200):
    """httpx.Response 모의 객체를 생성합니다."""
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = status_code
    if json_data is not None:
        mock.json.return_value = json_data
    if content is not None:
        mock.content = content
    mock.raise_for_status.return_value = None
    return mock


def _mock_error():
    """연결 오류 mock 을 생성합니다."""
    return httpx.ConnectError("연결 실패")


# ------------------------------------------------------------------
# fetch_status
# ------------------------------------------------------------------


class TestFetchStatus:
    def test_returns_dict_on_success(self) -> None:
        payload = {"threat_level": "NONE", "threat_score": 0.0}
        with patch("httpx.get", return_value=_mock_response(json_data=payload)):
            result = api_client.fetch_status()
        assert result == payload

    def test_returns_none_on_connection_error(self) -> None:
        with patch("httpx.get", side_effect=_mock_error()):
            result = api_client.fetch_status()
        assert result is None

    def test_returns_none_on_timeout(self) -> None:
        with patch("httpx.get", side_effect=httpx.TimeoutException("timeout")):
            result = api_client.fetch_status()
        assert result is None

    def test_returns_none_on_http_error(self) -> None:
        mock = _mock_response(status_code=500)
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "서버 오류", request=MagicMock(), response=mock
        )
        with patch("httpx.get", return_value=mock):
            result = api_client.fetch_status()
        assert result is None


# ------------------------------------------------------------------
# fetch_recordings
# ------------------------------------------------------------------


class TestFetchRecordings:
    def test_returns_list_on_success(self) -> None:
        payload = {
            "count": 2,
            "recordings": [
                {"filename": "a.mp4", "size_bytes": 100, "created_at": "2026-01-01T00:00:00"},
                {"filename": "b.mp4", "size_bytes": 200, "created_at": "2026-01-01T00:01:00"},
            ],
        }
        with patch("httpx.get", return_value=_mock_response(json_data=payload)):
            result = api_client.fetch_recordings()
        assert len(result) == 2
        assert result[0]["filename"] == "a.mp4"

    def test_returns_empty_list_on_error(self) -> None:
        with patch("httpx.get", side_effect=_mock_error()):
            result = api_client.fetch_recordings()
        assert result == []

    def test_returns_empty_list_when_key_missing(self) -> None:
        with patch("httpx.get", return_value=_mock_response(json_data={"count": 0})):
            result = api_client.fetch_recordings()
        assert result == []


# ------------------------------------------------------------------
# fetch_frame
# ------------------------------------------------------------------


class TestFetchFrame:
    def test_returns_bytes_on_success(self) -> None:
        with patch("httpx.get", return_value=_mock_response(content=b"jpeg_data", status_code=200)):
            result = api_client.fetch_frame()
        assert result == b"jpeg_data"

    def test_returns_none_on_204(self) -> None:
        mock = MagicMock(spec=httpx.Response)
        mock.status_code = 204
        with patch("httpx.get", return_value=mock):
            result = api_client.fetch_frame()
        assert result is None

    def test_returns_none_on_error(self) -> None:
        with patch("httpx.get", side_effect=_mock_error()):
            result = api_client.fetch_frame()
        assert result is None


# ------------------------------------------------------------------
# fetch_alerts
# ------------------------------------------------------------------


class TestFetchAlerts:
    def test_returns_list_on_success(self) -> None:
        payload = {"count": 1, "alerts": [{"level": "HIGH", "timestamp": "2026-01-01"}]}
        with patch("httpx.get", return_value=_mock_response(json_data=payload)):
            result = api_client.fetch_alerts()
        assert len(result) == 1
        assert result[0]["level"] == "HIGH"

    def test_returns_empty_on_error(self) -> None:
        with patch("httpx.get", side_effect=_mock_error()):
            result = api_client.fetch_alerts()
        assert result == []


# ------------------------------------------------------------------
# fetch_settings
# ------------------------------------------------------------------


class TestFetchSettings:
    def test_returns_dict_on_success(self) -> None:
        payload = {"sos_hold_seconds": 3.0, "yolo_confidence": 0.5}
        with patch("httpx.get", return_value=_mock_response(json_data=payload)):
            result = api_client.fetch_settings()
        assert result == payload

    def test_returns_none_on_error(self) -> None:
        with patch("httpx.get", side_effect=_mock_error()):
            result = api_client.fetch_settings()
        assert result is None


# ------------------------------------------------------------------
# update_settings
# ------------------------------------------------------------------


class TestUpdateSettings:
    def test_returns_true_on_success(self) -> None:
        with patch("httpx.put", return_value=_mock_response(json_data={})):
            result = api_client.update_settings({"sos_hold_seconds": 5.0})
        assert result is True

    def test_returns_false_on_error(self) -> None:
        with patch("httpx.put", side_effect=_mock_error()):
            result = api_client.update_settings({"sos_hold_seconds": 5.0})
        assert result is False

    def test_returns_false_on_http_error(self) -> None:
        mock = _mock_response(status_code=422)
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "오류", request=MagicMock(), response=mock
        )
        with patch("httpx.put", return_value=mock):
            result = api_client.update_settings({"bad": "data"})
        assert result is False


# ------------------------------------------------------------------
# delete_recording
# ------------------------------------------------------------------


class TestDeleteRecording:
    def test_returns_true_on_success(self) -> None:
        with patch("httpx.delete", return_value=_mock_response(json_data={"deleted": "a.mp4"})):
            result = api_client.delete_recording("a.mp4")
        assert result is True

    def test_returns_false_on_404(self) -> None:
        mock = _mock_response(status_code=404)
        mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "not found", request=MagicMock(), response=mock
        )
        with patch("httpx.delete", return_value=mock):
            result = api_client.delete_recording("missing.mp4")
        assert result is False

    def test_returns_false_on_error(self) -> None:
        with patch("httpx.delete", side_effect=_mock_error()):
            result = api_client.delete_recording("a.mp4")
        assert result is False


# ------------------------------------------------------------------
# API_BASE 환경변수
# ------------------------------------------------------------------


class TestApiBase:
    def test_default_base_url(self) -> None:
        import importlib
        import os

        env_backup = os.environ.pop("PSDS_API_URL", None)
        try:
            importlib.reload(api_client)
            assert api_client.API_BASE == "http://localhost:8000"
        finally:
            if env_backup:
                os.environ["PSDS_API_URL"] = env_backup
            importlib.reload(api_client)

    def test_custom_base_url(self) -> None:
        import importlib
        import os

        os.environ["PSDS_API_URL"] = "http://192.168.1.100:9000"
        try:
            importlib.reload(api_client)
            assert api_client.API_BASE == "http://192.168.1.100:9000"
        finally:
            del os.environ["PSDS_API_URL"]
            importlib.reload(api_client)
