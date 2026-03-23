"""src/api/server.py — PSDS FastAPI 상태 API + WebSocket 스트리밍.

엔드포인트:
    GET    /status                  현재 위협 레벨, 보호 대상 상태, SOS 여부
    GET    /frame                   최신 처리 프레임 JPEG (없으면 204)
    GET    /recordings              저장된 영상 목록
    GET    /recordings/{filename}   영상 파일 다운로드
    DELETE /recordings/{filename}   영상 파일 삭제
    GET    /alerts                  알림 이력
    GET    /settings                현재 파이프라인 설정
    PUT    /settings                설정 갱신
    WS     /ws/live                 실시간 상태 스트리밍 (1초 간격 JSON)

Usage (독립 실행)::

    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from src.api.state import broadcaster

logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path("recordings")

app = FastAPI(title="PSDS API", version="1.0.0")


# ------------------------------------------------------------------
# REST 엔드포인트
# ------------------------------------------------------------------


@app.get("/status", summary="현재 파이프라인 상태")
async def get_status() -> dict[str, Any]:
    """현재 위협 레벨, 보호 대상 상태, SOS 감지 여부를 반환합니다."""
    return broadcaster.get_status()


@app.get("/frame", summary="최신 처리 프레임 JPEG")
async def get_frame() -> Response:
    """파이프라인이 출력한 최신 AI 오버레이 프레임을 JPEG 로 반환합니다.

    파이프라인이 실행 중이지 않으면 204 No Content 를 반환합니다.
    """
    data = broadcaster.get_frame_jpg()
    if data is None:
        return Response(status_code=204)
    return Response(content=data, media_type="image/jpeg")


@app.get("/recordings", summary="저장된 영상 목록")
async def list_recordings() -> dict[str, Any]:
    """recordings/ 폴더의 MP4 파일 목록을 반환합니다."""
    RECORDINGS_DIR.mkdir(exist_ok=True)
    files = sorted(
        RECORDINGS_DIR.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return {
        "count": len(files),
        "recordings": [
            {
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in files
        ],
    }


@app.get("/recordings/{filename}", summary="영상 파일 다운로드")
async def download_recording(filename: str) -> FileResponse:
    """지정한 영상 파일을 다운로드합니다.

    Args:
        filename: recordings/ 폴더 내 파일명 (예: 2026-03-24_14-30-22.mp4).

    Raises:
        HTTPException: 파일이 없으면 404.
    """
    safe_name = Path(filename).name
    filepath = RECORDINGS_DIR / safe_name
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(str(filepath), media_type="video/mp4", filename=safe_name)


@app.delete("/recordings/{filename}", summary="영상 파일 삭제")
async def delete_recording(filename: str) -> dict[str, str]:
    """지정한 영상 파일을 삭제합니다.

    Args:
        filename: 삭제할 파일명.

    Raises:
        HTTPException: 파일이 없으면 404.
    """
    safe_name = Path(filename).name
    filepath = RECORDINGS_DIR / safe_name
    if not filepath.exists() or not filepath.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    filepath.unlink()
    logger.info("영상 삭제: %s", safe_name)
    return {"deleted": safe_name}


@app.get("/alerts", summary="알림 이력")
async def get_alerts() -> dict[str, Any]:
    """경찰 신고 시뮬레이션 등 알림 이력을 반환합니다."""
    alerts = broadcaster.get_alerts()
    return {"count": len(alerts), "alerts": alerts}


@app.get("/settings", summary="현재 파이프라인 설정")
async def get_settings() -> dict[str, Any]:
    """현재 파이프라인 설정값을 반환합니다."""
    return broadcaster.get_settings()


@app.put("/settings", summary="파이프라인 설정 갱신")
async def update_settings(body: dict[str, Any]) -> dict[str, Any]:
    """파이프라인 설정을 갱신합니다.

    Args:
        body: 갱신할 설정 키-값 딕셔너리.

    Returns:
        갱신 후 전체 설정.
    """
    broadcaster.update_settings(**body)
    return broadcaster.get_settings()


# ------------------------------------------------------------------
# WebSocket 엔드포인트
# ------------------------------------------------------------------


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """실시간 파이프라인 상태를 1초 간격으로 JSON 스트리밍합니다.

    여러 클라이언트가 동시에 접속해도 독립적으로 처리됩니다.
    클라이언트 연결이 끊기면 자동으로 정리됩니다.
    """
    await websocket.accept()
    logger.info("WebSocket 클라이언트 접속")
    try:
        while True:
            await asyncio.sleep(1.0)
            payload = broadcaster.get_status()
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info("WebSocket 클라이언트 연결 종료")
    except Exception as exc:
        logger.warning("WebSocket 오류: %s", exc)
