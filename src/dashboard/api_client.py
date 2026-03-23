"""src/dashboard/api_client.py — PSDS FastAPI 서버 HTTP 클라이언트.

환경변수 PSDS_API_URL 로 서버 주소를 지정할 수 있습니다 (기본: http://localhost:8000).

Usage::

    from src.dashboard import api_client

    status = api_client.fetch_status()   # None 이면 서버 미연결
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# 환경변수로 주소 변경 가능
API_BASE: str = os.environ.get("PSDS_API_URL", "http://localhost:8000")
_TIMEOUT: float = 2.0  # 초


def _get(path: str) -> httpx.Response | None:
    """GET 요청을 보내고 실패 시 None 을 반환합니다."""
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=_TIMEOUT)
        r.raise_for_status()
        return r
    except Exception as exc:
        logger.debug("API GET %s 실패: %s", path, exc)
        return None


def fetch_status() -> dict[str, Any] | None:
    """현재 파이프라인 상태를 반환합니다.

    Returns:
        상태 딕셔너리. 서버 미연결 시 None.
    """
    r = _get("/status")
    return r.json() if r else None


def fetch_recordings() -> list[dict[str, Any]]:
    """저장된 영상 목록을 반환합니다.

    Returns:
        녹화 파일 정보 리스트. 서버 미연결 시 빈 리스트.
    """
    r = _get("/recordings")
    if r is None:
        return []
    return r.json().get("recordings", [])


def fetch_frame() -> bytes | None:
    """최신 AI 처리 프레임 JPEG 를 반환합니다.

    Returns:
        JPEG 바이트. 프레임 없거나 서버 미연결 시 None.
    """
    try:
        r = httpx.get(f"{API_BASE}/frame", timeout=_TIMEOUT)
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.content
    except Exception as exc:
        logger.debug("frame 조회 실패: %s", exc)
        return None


def fetch_alerts() -> list[dict[str, Any]]:
    """알림 이력을 반환합니다.

    Returns:
        알림 딕셔너리 리스트. 서버 미연결 시 빈 리스트.
    """
    r = _get("/alerts")
    if r is None:
        return []
    return r.json().get("alerts", [])


def fetch_settings() -> dict[str, Any] | None:
    """현재 파이프라인 설정을 반환합니다.

    Returns:
        설정 딕셔너리. 서버 미연결 시 None.
    """
    r = _get("/settings")
    return r.json() if r else None


def update_settings(settings: dict[str, Any]) -> bool:
    """파이프라인 설정을 갱신합니다.

    Args:
        settings: 갱신할 설정 딕셔너리.

    Returns:
        성공 여부.
    """
    try:
        r = httpx.put(f"{API_BASE}/settings", json=settings, timeout=_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.debug("settings 갱신 실패: %s", exc)
        return False


def delete_recording(filename: str) -> bool:
    """녹화 파일을 삭제합니다.

    Args:
        filename: 삭제할 파일명.

    Returns:
        성공 여부.
    """
    try:
        r = httpx.delete(f"{API_BASE}/recordings/{filename}", timeout=_TIMEOUT)
        r.raise_for_status()
        return True
    except Exception as exc:
        logger.debug("영상 삭제 실패 (%s): %s", filename, exc)
        return False
