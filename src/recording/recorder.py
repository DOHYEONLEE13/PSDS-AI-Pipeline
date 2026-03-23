"""src/recording/recorder.py — 위협 감지 시 자동 영상 저장.

위협 레벨 MEDIUM 이상이면 녹화를 시작하고, NONE 으로 돌아온 후
STOP_DELAY_SECONDS 초 뒤 자동 정지합니다.
recordings/ 폴더 총 용량이 MAX_STORAGE_BYTES 를 초과하면
가장 오래된 파일을 자동 삭제합니다.
"""
from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.threat_detection.detector import ThreatLevel

logger = logging.getLogger(__name__)

# 이 레벨 이상이면 녹화 시작
_RECORD_LEVELS = {ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL}


class VideoRecorder:
    """위협 레벨 연동 자동 영상 저장기.

    Args:
        recordings_dir: 영상 저장 폴더.
        fps: 녹화 FPS.
        max_storage_bytes: 최대 저장 용량 (기본 1 GB).
        stop_delay: 위협 해제 후 추가 녹화 시간 (초, 기본 10).
    """

    MAX_STORAGE_BYTES: int = 1 * 1024 * 1024 * 1024  # 1 GB
    STOP_DELAY_SECONDS: float = 10.0
    _MIN_FREE_BYTES: int = 100 * 1024 * 1024  # 100 MB

    def __init__(
        self,
        recordings_dir: str | Path = "recordings",
        fps: float = 30.0,
        max_storage_bytes: int = MAX_STORAGE_BYTES,
        stop_delay: float = STOP_DELAY_SECONDS,
    ) -> None:
        self._dir = Path(recordings_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._max_bytes = max_storage_bytes
        self._stop_delay = stop_delay

        self._writer: cv2.VideoWriter | None = None
        self._current_file: Path | None = None
        self._is_recording: bool = False
        self._stop_pending_since: float | None = None  # 위협 해제 시각

    # ------------------------------------------------------------------
    # 공개 속성
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """현재 녹화 중 여부."""
        return self._is_recording

    @property
    def current_file(self) -> Path | None:
        """현재 녹화 중인 파일 경로. 녹화 중이 아니면 None."""
        return self._current_file

    # ------------------------------------------------------------------
    # 메인 인터페이스
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, threat_level: ThreatLevel) -> None:
        """프레임과 위협 레벨을 받아 녹화 상태를 갱신합니다.

        - 위협 레벨 MEDIUM 이상 → 녹화 시작
        - 위협 레벨 NONE/LOW → STOP_DELAY 후 녹화 정지
        - 녹화 중이면 현재 프레임을 파일에 씁니다.

        Args:
            frame: 현재 BGR 프레임.
            threat_level: 현재 위협 레벨.
        """
        should_record = threat_level in _RECORD_LEVELS

        if should_record:
            self._stop_pending_since = None  # 위협 재개 → 딜레이 리셋
            if not self._is_recording:
                self._start_recording(frame)
        else:
            if self._is_recording:
                now = time.monotonic()
                if self._stop_pending_since is None:
                    self._stop_pending_since = now
                elif now - self._stop_pending_since >= self._stop_delay:
                    self._stop_recording()

        if self._is_recording and self._writer is not None:
            try:
                self._writer.write(frame)
            except OSError as exc:
                logger.error("디스크 오류 — 녹화 강제 중지: %s", exc)
                self._stop_recording()

    def draw_rec_indicator(self, frame: np.ndarray) -> None:
        """녹화 중일 때 빨간 원과 'REC' 텍스트를 프레임에 그립니다.

        Args:
            frame: 그릴 대상 BGR 프레임 (in-place 수정).
        """
        if not self._is_recording:
            return
        cv2.circle(frame, (28, 28), 10, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC",
            (44, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )

    def stop(self) -> None:
        """녹화를 즉시 정지합니다."""
        if self._is_recording:
            self._stop_recording()

    def list_recordings(self) -> list[Path]:
        """저장된 영상 파일 목록을 수정 시각 내림차순으로 반환합니다."""
        return sorted(
            self._dir.glob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def total_size_bytes(self) -> int:
        """recordings 폴더의 총 용량 (바이트)."""
        return sum(p.stat().st_size for p in self._dir.glob("*.mp4"))

    # ------------------------------------------------------------------
    # 내부 메서드
    # ------------------------------------------------------------------

    def _start_recording(self, frame: np.ndarray) -> None:
        """VideoWriter 를 생성해 녹화를 시작합니다."""
        if not self._check_disk_space():
            return
        self._enforce_storage_limit()

        h, w = frame.shape[:2]
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = self._dir / f"{ts}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, self._fps, (w, h))
        if not writer.isOpened():
            logger.error("VideoWriter 열기 실패: %s", filepath)
            return

        self._writer = writer
        self._current_file = filepath
        self._is_recording = True
        logger.info("녹화 시작: %s", filepath.name)

    def _stop_recording(self) -> None:
        """VideoWriter 를 닫고 녹화를 종료합니다."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._is_recording = False
        self._stop_pending_since = None
        if self._current_file:
            logger.info("녹화 정지: %s", self._current_file.name)
        self._current_file = None

    def _enforce_storage_limit(self) -> None:
        """총 용량이 max_storage_bytes 를 초과하면 가장 오래된 파일부터 삭제합니다."""
        files = sorted(self._dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while total > self._max_bytes and files:
            oldest = files.pop(0)
            size = oldest.stat().st_size
            oldest.unlink()
            total -= size
            logger.warning("저장 용량 초과 — 삭제: %s", oldest.name)

    def _check_disk_space(self) -> bool:
        """디스크 여유 공간을 확인합니다.

        Returns:
            여유 공간이 충분하면 True. 부족하면 경고 로그를 남기고 False.
        """
        try:
            usage = shutil.disk_usage(self._dir)
        except OSError:
            return True  # 확인 불가 시 허용
        if usage.free < self._MIN_FREE_BYTES:
            logger.warning(
                "디스크 여유 공간 부족 (%.0f MB) — 녹화 중지",
                usage.free / 1_000_000,
            )
            return False
        return True
