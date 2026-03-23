"""tests/test_recorder.py — VideoRecorder 단위 테스트."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.recording.recorder import VideoRecorder
from src.threat_detection.detector import ThreatLevel


@pytest.fixture()
def recorder(tmp_path: Path) -> VideoRecorder:
    """임시 폴더를 사용하는 VideoRecorder 픽스처."""
    return VideoRecorder(recordings_dir=tmp_path, fps=30.0, stop_delay=10.0)


def blank_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestVideoRecorderInit:
    def test_recordings_dir_is_created(self, tmp_path: Path) -> None:
        d = tmp_path / "recs"
        VideoRecorder(recordings_dir=d)
        assert d.exists()

    def test_initial_state(self, recorder: VideoRecorder) -> None:
        assert not recorder.is_recording
        assert recorder.current_file is None

    def test_total_size_zero_initially(self, recorder: VideoRecorder) -> None:
        assert recorder.total_size_bytes() == 0

    def test_list_recordings_empty_initially(self, recorder: VideoRecorder) -> None:
        assert recorder.list_recordings() == []


class TestRecordingStartStop:
    def test_starts_recording_on_medium_threat(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.MEDIUM)
        assert recorder.is_recording

    def test_starts_recording_on_high_threat(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.HIGH)
        assert recorder.is_recording

    def test_starts_recording_on_critical_threat(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.CRITICAL)
        assert recorder.is_recording

    def test_does_not_start_on_none_threat(
        self, recorder: VideoRecorder
    ) -> None:
        recorder.update(blank_frame(), ThreatLevel.NONE)
        assert not recorder.is_recording

    def test_does_not_start_on_low_threat(
        self, recorder: VideoRecorder
    ) -> None:
        recorder.update(blank_frame(), ThreatLevel.LOW)
        assert not recorder.is_recording

    def test_stop_called_immediately_stops(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.HIGH)
            assert recorder.is_recording
            recorder.stop()
        assert not recorder.is_recording

    def test_stop_delay_not_yet_elapsed(
        self, recorder: VideoRecorder
    ) -> None:
        """위협 해제 직후에는 아직 녹화 중이어야 합니다."""
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.HIGH)
            recorder.update(blank_frame(), ThreatLevel.NONE)  # 해제
        # 딜레이 미경과 → 여전히 녹화 중
        assert recorder.is_recording

    def test_stop_after_delay_elapsed(
        self, recorder: VideoRecorder
    ) -> None:
        """딜레이가 경과하면 녹화가 멈춰야 합니다."""
        fast = VideoRecorder(recordings_dir=recorder._dir, fps=30.0, stop_delay=0.01)
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            fast.update(blank_frame(), ThreatLevel.HIGH)
            fast.update(blank_frame(), ThreatLevel.NONE)  # 위협 해제
            time.sleep(0.05)
            fast.update(blank_frame(), ThreatLevel.NONE)  # 딜레이 경과 후 호출
        assert not fast.is_recording

    def test_threat_resumes_resets_stop_timer(
        self, recorder: VideoRecorder
    ) -> None:
        """위협이 해제됐다가 다시 발생하면 타이머가 리셋돼야 합니다."""
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.HIGH)
            recorder.update(blank_frame(), ThreatLevel.NONE)  # 딜레이 시작
            recorder.update(blank_frame(), ThreatLevel.HIGH)  # 위협 재개
        assert recorder._stop_pending_since is None  # 타이머 리셋


class TestRecordingCurrentFile:
    def test_current_file_set_when_recording(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.HIGH)
        assert recorder.current_file is not None
        assert recorder.current_file.suffix == ".mp4"

    def test_current_file_cleared_after_stop(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(blank_frame(), ThreatLevel.HIGH)
            recorder.stop()
        assert recorder.current_file is None

    def test_video_writer_called_with_correct_size(
        self, recorder: VideoRecorder
    ) -> None:
        frame = blank_frame(h=480, w=640)
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(frame, ThreatLevel.HIGH)
        # VideoWriter(path, fourcc, fps, (w, h)) 형식 확인
        args = mock_vw.call_args[0]
        assert args[3] == (640, 480)


class TestRecDrawIndicator:
    def test_no_draw_when_not_recording(self, recorder: VideoRecorder) -> None:
        frame = blank_frame()
        recorder.draw_rec_indicator(frame)
        # 프레임이 변경되지 않아야 함 (모두 0)
        assert frame.sum() == 0

    def test_draws_when_recording(self, recorder: VideoRecorder) -> None:
        frame = blank_frame(h=100, w=100)
        with patch("cv2.VideoWriter") as mock_vw:
            mock_vw.return_value.isOpened.return_value = True
            recorder.update(frame, ThreatLevel.HIGH)
        recorder.draw_rec_indicator(frame)
        assert frame.sum() > 0  # 픽셀이 그려짐


class TestStorageLimit:
    def test_enforce_storage_limit_deletes_oldest(
        self, tmp_path: Path
    ) -> None:
        """총 용량 초과 시 가장 오래된 파일이 삭제돼야 합니다."""
        rec = VideoRecorder(
            recordings_dir=tmp_path,
            max_storage_bytes=100,  # 100 바이트 제한
        )
        # 파일 2개 생성
        f1 = tmp_path / "2026-01-01_00-00-01.mp4"
        f2 = tmp_path / "2026-01-01_00-00-02.mp4"
        f1.write_bytes(b"x" * 60)
        f2.write_bytes(b"x" * 60)
        # f1이 더 오래됐도록 mtime 조정
        import os
        os.utime(f1, (1000.0, 1000.0))
        os.utime(f2, (2000.0, 2000.0))

        rec._enforce_storage_limit()

        assert not f1.exists()
        assert f2.exists()

    def test_no_deletion_within_limit(self, tmp_path: Path) -> None:
        rec = VideoRecorder(recordings_dir=tmp_path, max_storage_bytes=200)
        f = tmp_path / "2026-01-01_00-00-01.mp4"
        f.write_bytes(b"x" * 50)
        rec._enforce_storage_limit()
        assert f.exists()


class TestDiskSpaceCheck:
    def test_returns_false_when_disk_full(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(free=1024)  # 1 KB
            assert recorder._check_disk_space() is False

    def test_returns_true_when_disk_ok(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(free=500 * 1024 * 1024)
            assert recorder._check_disk_space() is True

    def test_no_recording_when_disk_full(
        self, recorder: VideoRecorder
    ) -> None:
        with patch("shutil.disk_usage") as mock_du:
            mock_du.return_value = MagicMock(free=1024)
            recorder.update(blank_frame(), ThreatLevel.HIGH)
        assert not recorder.is_recording
