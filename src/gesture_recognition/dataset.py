"""src/gesture_recognition/dataset.py — 모션 데이터 수집·저장 도구."""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator

import numpy as np


class MotionLabel(str, Enum):
    """녹화할 모션 레이블."""

    NORMAL = "normal"
    SOS = "sos"


@dataclass
class MotionSample:
    """단일 모션 샘플 (30프레임 시퀀스 + 레이블).

    Attributes:
        sequence: (30, 63) float32 배열 — 30프레임 × 21관절 × 3좌표(x,y,z).
        label: 모션 레이블.
        recorded_at: 녹화 시각 (Unix timestamp).
    """

    sequence: np.ndarray  # shape (30, 63)
    label: MotionLabel
    recorded_at: float = field(default_factory=time.time)

    def to_flat(self) -> np.ndarray:
        """(30, 63) → (1890,) 1D 배열로 변환합니다."""
        return self.sequence.flatten()


class SequenceBuffer:
    """슬라이딩 윈도우 방식으로 프레임을 누적하는 원형 버퍼.

    Args:
        seq_len: 윈도우 크기 (프레임 수, 기본값 30).
        num_landmarks: 랜드마크 수 (기본값 21).
        coords_per_landmark: 좌표 차원 수 (기본값 3, x/y/z).
    """

    def __init__(
        self,
        seq_len: int = 30,
        num_landmarks: int = 21,
        coords_per_landmark: int = 3,
    ) -> None:
        self._seq_len = seq_len
        self._feature_size = num_landmarks * coords_per_landmark
        self._buf: deque[np.ndarray] = deque(maxlen=seq_len)

    @property
    def seq_len(self) -> int:
        """윈도우 크기."""
        return self._seq_len

    @property
    def is_full(self) -> bool:
        """버퍼가 seq_len 프레임으로 가득 찼는지 여부."""
        return len(self._buf) == self._seq_len

    @property
    def fill_ratio(self) -> float:
        """현재 채워진 비율 (0.0 ~ 1.0)."""
        return len(self._buf) / self._seq_len

    def push(self, landmarks: list[tuple[float, float, float]]) -> None:
        """한 프레임의 랜드마크를 버퍼에 추가합니다.

        Args:
            landmarks: 21개 (x, y, z) 랜드마크 리스트.
        """
        flat = np.array([coord for lm in landmarks for coord in lm], dtype=np.float32)
        self._buf.append(flat)

    def get_sequence(self) -> np.ndarray | None:
        """버퍼가 가득 찬 경우 (seq_len, feature_size) 배열을 반환합니다.

        Returns:
            (30, 63) float32 배열. 버퍼 미충족 시 None.
        """
        if not self.is_full:
            return None
        return np.array(list(self._buf), dtype=np.float32)

    def clear(self) -> None:
        """버퍼를 초기화합니다."""
        self._buf.clear()


class DatasetCollector:
    """웹캠 프레임에서 손 모션을 녹화하고 레이블링하는 데이터 수집 도구.

    HandTracker와 연동하여 실시간으로 데이터를 수집합니다.
    저장 포맷: ``{label}_{timestamp_ms}.npy`` (시퀀스) + ``.json`` (메타데이터).

    Args:
        save_dir: 데이터셋 저장 디렉터리.
        seq_len: 시퀀스 길이 (프레임 수, 기본값 30).
        label: 현재 녹화할 레이블 (기본값 NORMAL).

    Examples:
        >>> collector = DatasetCollector("data/raw", label=MotionLabel.SOS)
        >>> collector.push_frame(landmarks)   # HandTracker 랜드마크
        >>> if collector.is_sample_ready:
        ...     sample = collector.capture_sample()
        ...     collector.save_sample(sample)
    """

    def __init__(
        self,
        save_dir: str | Path = Path("data/raw"),
        seq_len: int = 30,
        label: MotionLabel = MotionLabel.NORMAL,
    ) -> None:
        self._save_dir = Path(save_dir)
        self._label = label
        self._buffer = SequenceBuffer(seq_len=seq_len)
        self._samples: list[MotionSample] = []

    @property
    def label(self) -> MotionLabel:
        """현재 녹화 레이블."""
        return self._label

    @label.setter
    def label(self, value: MotionLabel) -> None:
        self._label = value

    @property
    def sample_count(self) -> int:
        """누적 수집된 샘플 수."""
        return len(self._samples)

    @property
    def is_sample_ready(self) -> bool:
        """현재 버퍼로 샘플을 생성할 수 있는지 여부."""
        return self._buffer.is_full

    @property
    def fill_ratio(self) -> float:
        """버퍼 채워진 비율 (0.0 ~ 1.0)."""
        return self._buffer.fill_ratio

    def push_frame(self, landmarks: list[tuple[float, float, float]]) -> None:
        """한 프레임의 랜드마크를 버퍼에 추가합니다.

        Args:
            landmarks: 21개 (x, y, z) 랜드마크 리스트.
        """
        self._buffer.push(landmarks)

    def capture_sample(self) -> MotionSample | None:
        """현재 버퍼에서 샘플을 생성하고 내부 목록에 추가합니다.

        버퍼를 capture 후 초기화하므로 다음 샘플을 위해 다시 채워야 합니다.

        Returns:
            MotionSample 인스턴스. 버퍼 미충족 시 None.
        """
        seq = self._buffer.get_sequence()
        if seq is None:
            return None
        sample = MotionSample(sequence=seq, label=self._label)
        self._samples.append(sample)
        self._buffer.clear()
        return sample

    def save_sample(self, sample: MotionSample) -> Path:
        """샘플을 디스크에 저장합니다.

        Args:
            sample: 저장할 MotionSample.

        Returns:
            저장된 ``.npy`` 파일 경로.
        """
        self._save_dir.mkdir(parents=True, exist_ok=True)
        ts_ms = int(sample.recorded_at * 1000)
        stem = f"{sample.label.value}_{ts_ms}"
        npy_path = self._save_dir / f"{stem}.npy"
        json_path = self._save_dir / f"{stem}.json"
        np.save(str(npy_path), sample.sequence)
        meta = {"label": sample.label.value, "recorded_at": sample.recorded_at}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        return npy_path

    def save_all(self) -> list[Path]:
        """누적된 모든 샘플을 디스크에 저장합니다.

        Returns:
            저장된 파일 경로 목록.
        """
        return [self.save_sample(s) for s in self._samples]

    @staticmethod
    def load_samples(data_dir: str | Path) -> list[MotionSample]:
        """디렉터리에서 ``.npy`` 파일을 읽어 MotionSample 리스트를 반환합니다.

        Args:
            data_dir: 데이터가 저장된 디렉터리 경로.

        Returns:
            로드된 MotionSample 리스트.
        """
        data_dir = Path(data_dir)
        samples: list[MotionSample] = []
        for npy_path in sorted(data_dir.glob("*.npy")):
            json_path = npy_path.with_suffix(".json")
            label = MotionLabel.NORMAL
            recorded_at = 0.0
            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    meta = json.load(f)
                label = MotionLabel(meta.get("label", "normal"))
                recorded_at = float(meta.get("recorded_at", 0.0))
            seq = np.load(str(npy_path))
            samples.append(MotionSample(sequence=seq, label=label, recorded_at=recorded_at))
        return samples

    def iter_dataset(self) -> Iterator[tuple[np.ndarray, int]]:
        """수집된 샘플을 ``(sequence, class_index)`` 튜플로 yield합니다.

        Yields:
            (30×63 float32 배열, 클래스 정수 인덱스) 튜플.
            NORMAL → 0, SOS → 1.
        """
        label_to_idx: dict[MotionLabel, int] = {MotionLabel.NORMAL: 0, MotionLabel.SOS: 1}
        for sample in self._samples:
            yield sample.sequence, label_to_idx[sample.label]
