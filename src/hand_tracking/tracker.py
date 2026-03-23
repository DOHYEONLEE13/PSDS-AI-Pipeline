from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from filterpy.kalman import KalmanFilter as _KF
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
)

# MediaPipe 손 모델 21개 관절 연결 쌍 (랜드마크 인덱스)
HAND_CONNECTIONS: frozenset[tuple[int, int]] = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 4),         # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),         # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),    # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),  # 약지
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # 소지 + 손바닥
])

_DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")


@dataclass
class HandLandmarks:
    """단일 손의 랜드마크 결과."""

    landmarks: list[tuple[float, float, float]]  # (x, y, z) normalized
    handedness: str  # "Left" | "Right"
    confidence: float


@dataclass
class TrackingResult:
    """한 프레임의 추적 결과."""

    hands: list[HandLandmarks] = field(default_factory=list)
    frame_index: int = 0

    @property
    def detected(self) -> bool:
        return len(self.hands) > 0


class LandmarkKalmanFilter:
    """단일 랜드마크 (x, y, z) 에 대한 칼만 필터.

    상수 속도 모델(constant-velocity): state = [x, vx, y, vy, z, vz].
    측정값은 위치만 관측 (x, y, z).

    Args:
        process_noise: 프로세스 노이즈 공분산 스칼라 (Q 대각 성분).
        measurement_noise: 측정 노이즈 공분산 스칼라 (R 대각 성분).
    """

    def __init__(
        self,
        process_noise: float = 1e-2,
        measurement_noise: float = 1e-1,
    ) -> None:
        self._kf = _KF(dim_x=6, dim_z=3)
        dt = 1.0
        # 상태 전이 행렬: 등속 운동 모델
        self._kf.F = np.array(
            [
                [1, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=float,
        )
        # 관측 행렬: 위치만 측정
        self._kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=float,
        )
        self._kf.R = np.eye(3) * measurement_noise
        self._kf.Q = np.eye(6) * process_noise
        self._kf.P = np.eye(6) * 1.0
        self._initialized = False

    def update(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """측정값을 입력받아 필터링된 (x, y, z) 를 반환합니다.

        Args:
            x: 정규화된 x 좌표.
            y: 정규화된 y 좌표.
            z: 정규화된 z 좌표(깊이).

        Returns:
            필터링된 (x, y, z) 튜플.
        """
        if not self._initialized:
            self._kf.x = np.array([[x], [0.0], [y], [0.0], [z], [0.0]])
            self._initialized = True
        self._kf.predict()
        self._kf.update(np.array([[x], [y], [z]]))
        state = self._kf.x.flatten()
        return float(state[0]), float(state[2]), float(state[4])

    def reset(self) -> None:
        """필터 상태를 초기화합니다. 손이 사라졌다가 재등장할 때 호출합니다."""
        self._initialized = False


class HandTracker:
    """MediaPipe HandLandmarker Tasks API 기반 실시간 손 추적기.

    웹캠 BGR 프레임을 입력받아 손의 21개 관절 좌표를 추출합니다.
    칼만 필터 옵션으로 시계열 노이즈를 제거합니다.

    Args:
        model_path: hand_landmarker.task 모델 파일 경로.
        max_num_hands: 동시에 추적할 최대 손 개수.
        min_detection_confidence: 검출 최소 신뢰도.
        min_tracking_confidence: 추적 최소 신뢰도.
        use_kalman: True이면 랜드마크에 칼만 필터를 적용합니다.
        kalman_process_noise: 칼만 필터 프로세스 노이즈 분산.
        kalman_measurement_noise: 칼만 필터 측정 노이즈 분산.

    Examples:
        >>> with HandTracker("models/hand_landmarker.task") as tracker:
        ...     result = tracker.process(frame)
        ...     tracker.draw(frame, result)
    """

    _LANDMARK_COLOR: tuple[int, int, int] = (0, 255, 0)
    _CONNECTION_COLOR: tuple[int, int, int] = (255, 255, 255)
    _LABEL_COLOR: tuple[int, int, int] = (0, 255, 0)
    _LANDMARK_RADIUS: int = 4
    _CONNECTION_THICKNESS: int = 2
    _MS_PER_FRAME: int = 33  # ~30fps 기준 프레임당 타임스탬프 간격

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        use_kalman: bool = True,
        kalman_process_noise: float = 1e-2,
        kalman_measurement_noise: float = 1e-1,
    ) -> None:
        self._model_path = Path(model_path)
        self._max_num_hands = max_num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._use_kalman = use_kalman
        self._kalman_process_noise = kalman_process_noise
        self._kalman_measurement_noise = kalman_measurement_noise
        self._landmarker: HandLandmarker | None = None
        self._frame_index = 0
        self._timestamp_ms = 0
        # handedness("Left"/"Right") → 21개 LandmarkKalmanFilter
        self._kalman_filters: dict[str, list[LandmarkKalmanFilter]] = {}

    def start(self) -> None:
        """HandLandmarker 세션을 초기화합니다."""
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=self._max_num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_tracking_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_index = 0
        self._timestamp_ms = 0
        self._kalman_filters.clear()

    def stop(self) -> None:
        """리소스를 해제합니다."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._kalman_filters.clear()

    def _get_kalman_filters(self, handedness: str) -> list[LandmarkKalmanFilter]:
        """handedness별 21개 칼만 필터를 반환합니다. 없으면 생성합니다.

        Args:
            handedness: "Left" 또는 "Right".

        Returns:
            21개 LandmarkKalmanFilter 리스트.
        """
        if handedness not in self._kalman_filters:
            self._kalman_filters[handedness] = [
                LandmarkKalmanFilter(
                    process_noise=self._kalman_process_noise,
                    measurement_noise=self._kalman_measurement_noise,
                )
                for _ in range(21)
            ]
        return self._kalman_filters[handedness]

    def _apply_kalman(
        self,
        landmarks: list[tuple[float, float, float]],
        handedness: str,
    ) -> list[tuple[float, float, float]]:
        """랜드마크 리스트에 칼만 필터를 적용합니다.

        Args:
            landmarks: 21개 (x, y, z) 정규화 좌표 리스트.
            handedness: "Left" 또는 "Right".

        Returns:
            필터링된 21개 (x, y, z) 리스트.
        """
        filters = self._get_kalman_filters(handedness)
        return [f.update(x, y, z) for f, (x, y, z) in zip(filters, landmarks)]

    def process(self, frame: np.ndarray) -> TrackingResult:
        """BGR 프레임을 입력받아 손 랜드마크를 반환합니다.

        Args:
            frame: OpenCV BGR 형식의 입력 프레임.

        Returns:
            감지된 손 랜드마크 목록을 담은 TrackingResult.

        Raises:
            RuntimeError: start()를 먼저 호출하지 않은 경우.
        """
        if self._landmarker is None:
            raise RuntimeError("HandTracker가 시작되지 않았습니다. start()를 먼저 호출하세요.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        raw = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        self._frame_index += 1
        self._timestamp_ms += self._MS_PER_FRAME

        hands: list[HandLandmarks] = []
        detected_handedness: set[str] = set()

        if raw.hand_landmarks:
            for lm_list, handedness_list in zip(raw.hand_landmarks, raw.handedness):
                classification = handedness_list[0]
                label = classification.category_name
                detected_handedness.add(label)

                landmarks: list[tuple[float, float, float]] = [
                    (lm.x, lm.y, lm.z) for lm in lm_list
                ]

                if self._use_kalman:
                    landmarks = self._apply_kalman(landmarks, label)

                hands.append(
                    HandLandmarks(
                        landmarks=landmarks,
                        handedness=label,
                        confidence=classification.score,
                    )
                )

        # 이번 프레임에서 감지되지 않은 손의 칼만 필터를 리셋
        # → 재등장 시 이전 상태로 튀는 현상 방지
        for key in list(self._kalman_filters.keys()):
            if key not in detected_handedness:
                for kf in self._kalman_filters[key]:
                    kf.reset()

        return TrackingResult(hands=hands, frame_index=self._frame_index)

    def draw(self, frame: np.ndarray, result: TrackingResult) -> np.ndarray:
        """프레임에 랜드마크 포인트와 연결선을 시각화합니다.

        Args:
            frame: 시각화할 BGR 프레임 (in-place 수정됩니다).
            result: process()가 반환한 TrackingResult.

        Returns:
            시각화가 추가된 프레임 (입력과 동일한 객체).
        """
        h, w = frame.shape[:2]

        for hand in result.hands:
            lm = hand.landmarks

            # 연결선
            for start_idx, end_idx in HAND_CONNECTIONS:
                x1, y1 = int(lm[start_idx][0] * w), int(lm[start_idx][1] * h)
                x2, y2 = int(lm[end_idx][0] * w), int(lm[end_idx][1] * h)
                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    self._CONNECTION_COLOR,
                    self._CONNECTION_THICKNESS,
                )

            # 관절 포인트
            for x_n, y_n, _ in lm:
                cx, cy = int(x_n * w), int(y_n * h)
                cv2.circle(frame, (cx, cy), self._LANDMARK_RADIUS, self._LANDMARK_COLOR, -1)

            # 손목 위에 레이블 표시
            wx = int(lm[0][0] * w)
            wy = max(int(lm[0][1] * h) - 12, 0)
            label_text = f"{hand.handedness} {hand.confidence:.2f}"
            cv2.putText(
                frame,
                label_text,
                (wx, wy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self._LABEL_COLOR,
                2,
                cv2.LINE_AA,
            )

        return frame

    def __enter__(self) -> HandTracker:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    model = _Path("models/hand_landmarker.task")
    if not model.exists():
        print(
            "모델 파일이 없습니다. 아래 명령으로 다운로드하세요:\n"
            "  mkdir -p models && "
            "wget -q -O models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없습니다.")

    print("PSDS Hand Tracking 데모 시작 — 종료: q")

    with HandTracker(model_path=model, use_kalman=True) as tracker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임 읽기 실패 — 종료합니다.")
                break

            frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
            result = tracker.process(frame)
            tracker.draw(frame, result)

            status = f"Hands: {len(result.hands)}  Frame: {result.frame_index}"
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("PSDS Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
