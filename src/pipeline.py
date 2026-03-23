"""src/pipeline.py — 전체 AI 파이프라인 통합.

웹캠 또는 동영상 파일을 입력받아 손 추적, SOS 감지, 인물 감지,
위협 레벨 판단을 순차 실행하고 결과를 화면에 표시합니다.

Usage::

    # 웹캠
    python -m src.pipeline

    # 동영상 파일
    python -m src.pipeline --input video.mp4
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.alerts.alert_manager import AlertManager
from src.api.state import broadcaster  # noqa: E402
from src.gesture_recognition.recognizer import (
    GestureRecognizer,
    GestureResult,
    RuleBasedSOSDetector,
    SOSDetectionResult,
)
from src.hand_tracking.tracker import HandTracker, TrackingResult
from src.protection.protected_tracker import ProtectedPersonStatus, ProtectedPersonTracker
from src.protection.protector import Protector
from src.recording.recorder import VideoRecorder
from src.threat_detection.approach_analyzer import ApproachAnalyzer
from src.threat_detection.detector import SceneThreatDetector, ThreatLevel, ThreatResult
from src.threat_detection.yolo_detector import PersonDetection, YOLOPersonDetector

logger = logging.getLogger(__name__)

# 위협 레벨별 게이지 색상 (BGR)
_THREAT_COLORS: dict[ThreatLevel, tuple[int, int, int]] = {
    ThreatLevel.NONE: (0, 200, 0),
    ThreatLevel.LOW: (0, 200, 200),
    ThreatLevel.MEDIUM: (0, 165, 255),
    ThreatLevel.HIGH: (0, 0, 255),
    ThreatLevel.CRITICAL: (0, 0, 180),
}

_THREAT_LABELS: dict[ThreatLevel, str] = {
    ThreatLevel.NONE: "정상",
    ThreatLevel.LOW: "의심",
    ThreatLevel.MEDIUM: "위협",
    ThreatLevel.HIGH: "고위협",
    ThreatLevel.CRITICAL: "긴급",
}

_EMPTY_SOS = SOSDetectionResult(
    is_detected=False, is_pending=False, confidence=0.0, held_duration=0.0
)


class Pipeline:
    """PSDS AI 인식 파이프라인.

    웹캠 또는 동영상 파일에서 프레임을 읽어 손 추적 → SOS 감지 →
    인물 감지 → 위협 판단 → 보호 액션을 순차 실행합니다.

    Args:
        source: 웹캠 인덱스(int) 또는 동영상 파일 경로(str/Path).
        hand_model_path: MediaPipe 손 랜드마크 모델 파일 경로.
        yolo_model: YOLOv8 호환 모델 객체. None 이면 ``yolov8n.pt`` 자동 로드.
    """

    _SOS_HOLD_SECONDS: float = 3.0
    _FPS_LOW: float = 10.0   # 이하 → YOLO 3프레임에 1회
    _FPS_MED: float = 15.0   # 이하 → YOLO 2프레임에 1회

    def __init__(
        self,
        source: int | str | Path = 0,
        hand_model_path: str | Path = "models/hand_landmarker.task",
        yolo_model: object | None = None,
        recorder: VideoRecorder | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self._source = source
        self._is_webcam = isinstance(source, int)

        self._hand_tracker = HandTracker(
            model_path=hand_model_path,
            max_num_hands=4,
            use_kalman=True,
        )
        self._gesture_recognizer = GestureRecognizer()
        self._sos_detector = RuleBasedSOSDetector(hold_seconds=self._SOS_HOLD_SECONDS)
        self._protected_tracker = ProtectedPersonTracker()

        yolo = yolo_model if yolo_model is not None else YOLOPersonDetector()
        self._scene_detector = SceneThreatDetector(
            yolo_detector=yolo,
            approach_analyzer=ApproachAnalyzer(),
        )
        self._protector = Protector()

        self._recorder: VideoRecorder | None = recorder
        self._alert_manager: AlertManager | None = alert_manager

        # 런타임 상태
        self._fps: float = 30.0
        self._yolo_frame_count: int = 0
        self._yolo_skip: int = 1
        self._last_persons: list[PersonDetection] = []
        self._last_threat: ThreatResult = ThreatResult(level=ThreatLevel.NONE, score=0.0)
        self._last_wrists: dict[str, tuple[float, float]] = {}
        self._last_hand_centers: dict[str, tuple[float, float]] = {}  # 손 21개 관절 평균
        self._protection_start_time: str | None = None  # 보호 대상 등록 시각

    # ------------------------------------------------------------------
    # 내부 로직
    # ------------------------------------------------------------------

    def _update_yolo_skip(self) -> None:
        """현재 FPS에 따라 YOLO 추론 스킵 간격을 조정합니다.

        FPS 가 낮을수록 YOLO 추론을 덜 자주 실행해 성능을 확보합니다.
        """
        if self._fps < self._FPS_LOW:
            self._yolo_skip = 3
        elif self._fps < self._FPS_MED:
            self._yolo_skip = 2
        else:
            self._yolo_skip = 1

    def _find_protected_person_id(
        self,
        gesture_results: list[GestureResult],
        persons: list[PersonDetection],
    ) -> int | None:
        """V사인 손 위치를 YOLO 바운딩 박스에 매칭해 보호 대상 ID를 반환합니다.

        손 21개 관절 평균 좌표(``_last_hand_centers``)를 우선 사용하고,
        없으면 손목 좌표(``_last_wrists``)로 폴백합니다.

        1차: 손 좌표가 bbox 안에 있는 인물 선택(여유 마진 5%).
        2차: 박스 중심까지 거리가 가장 짧은 인물 선택.
        인물이 없으면 None 을 반환합니다.

        Args:
            gesture_results: 현재 프레임의 GestureResult 목록.
            persons: 현재 프레임의 PersonDetection 목록.

        Returns:
            보호 대상으로 지정할 person_id. 인물 없으면 None.
        """
        if not persons:
            return None

        v_signs = [r for r in gesture_results if r.gesture.name == "V_SIGN"]
        if not v_signs:
            return persons[0].person_id

        for v in v_signs:
            hand_pos = self._last_hand_centers.get(v.handedness) or self._last_wrists.get(
                v.handedness
            )
            if hand_pos is None:
                continue
            hx, hy = hand_pos
            m = 0.05

            # 1차: 손 좌표가 bbox(+마진) 안에 있는 인물
            for person in persons:
                x1, y1, x2, y2 = person.bbox
                if (x1 - m) <= hx <= (x2 + m) and (y1 - m) <= hy <= (y2 + m):
                    return person.person_id

            # 2차: bbox 중심까지 거리가 가장 짧은 인물
            return min(
                persons,
                key=lambda p: (
                    (hx - (p.bbox[0] + p.bbox[2]) / 2) ** 2
                    + (hy - (p.bbox[1] + p.bbox[3]) / 2) ** 2
                ),
            ).person_id

        return persons[0].person_id

    # ------------------------------------------------------------------
    # 프레임별 처리 단계 (각각 try-except 로 격리)
    # ------------------------------------------------------------------

    def _process_hand_tracking(self, frame: np.ndarray) -> TrackingResult:
        """손 관절 추적 단계."""
        try:
            result = self._hand_tracker.process(frame)
            self._hand_tracker.draw(frame, result)
            self._last_wrists = {
                hand.handedness: (hand.landmarks[0][0], hand.landmarks[0][1])
                for hand in result.hands
                if hand.landmarks
            }
            self._last_hand_centers = {
                hand.handedness: (
                    sum(lm[0] for lm in hand.landmarks) / len(hand.landmarks),
                    sum(lm[1] for lm in hand.landmarks) / len(hand.landmarks),
                )
                for hand in result.hands
                if hand.landmarks
            }
            return result
        except Exception:
            logger.exception("HandTracker 오류")
            self._last_wrists = {}
            self._last_hand_centers = {}
            return TrackingResult(hands=[], frame_index=0)

    def _process_gesture(self, tracking_result: TrackingResult) -> list[GestureResult]:
        """제스처 인식 단계."""
        try:
            return self._gesture_recognizer.recognize(tracking_result)
        except Exception:
            logger.exception("GestureRecognizer 오류")
            return []

    def _process_sos(self, gesture_results: list[GestureResult]) -> SOSDetectionResult:
        """SOS 감지 단계."""
        try:
            return self._sos_detector.update(gesture_results)
        except Exception:
            logger.exception("SOSDetector 오류")
            return _EMPTY_SOS

    def _process_yolo(
        self,
        frame: np.ndarray,
        gesture_results: list[GestureResult],
    ) -> tuple[list[PersonDetection], ThreatResult]:
        """YOLO 인물 감지 + 위협 판단 단계.

        FPS 가 낮으면 이전 결과를 재사용해 성능을 확보합니다.
        """
        self._yolo_frame_count += 1
        self._update_yolo_skip()

        if self._yolo_frame_count % self._yolo_skip != 0:
            return self._last_persons, self._last_threat

        try:
            threat, persons = self._scene_detector.detect(
                frame,
                gesture_results,
                protected_person_id=self._protected_tracker.protected_id,
                timestamp=time.monotonic(),
            )
            self._last_persons = persons
            self._last_threat = threat
            return persons, threat
        except Exception:
            logger.exception("SceneThreatDetector 오류")
            return self._last_persons, self._last_threat

    def _process_registration(
        self,
        sos_result: SOSDetectionResult,
        gesture_results: list[GestureResult],
        persons: list[PersonDetection],
    ) -> None:
        """SOS 확정 시 보호 대상 등록 단계.

        이미 등록된 보호 대상이 있으면 무시합니다.
        여러 명이 동시에 SOS 해도 먼저 확정된 한 명만 등록합니다.
        """
        if not sos_result.is_detected or self._protected_tracker.is_registered:
            return
        try:
            person_id = self._find_protected_person_id(gesture_results, persons)
            if person_id is not None:
                self._protected_tracker.register(person_id)
                self._sos_detector.reset()
                from datetime import datetime as _dt
                self._protection_start_time = _dt.now().isoformat()
                print(f"[PSDS] SOS 확정 → person_id={person_id} 보호 대상 등록")  # noqa: T201
                logger.info("보호 대상 등록: P%d", person_id)
        except Exception:
            logger.exception("보호 대상 등록 오류")

    def _process_protected_tracking(
        self,
        persons: list[PersonDetection],
    ) -> ProtectedPersonStatus | None:
        """보호 대상 위치 추적 단계."""
        try:
            return self._protected_tracker.update(persons)
        except Exception:
            logger.exception("보호 대상 추적 오류")
            return None

    # ------------------------------------------------------------------
    # 시각화
    # ------------------------------------------------------------------

    def draw_overlay(
        self,
        frame: np.ndarray,
        sos_result: SOSDetectionResult,
        persons: list[PersonDetection],
        protected_status: ProtectedPersonStatus | None,
        threat: ThreatResult,
    ) -> np.ndarray:
        """프레임에 모든 HUD 오버레이를 그립니다.

        Args:
            frame: 그릴 대상 BGR 프레임 (in-place 수정).
            sos_result: SOS 감지 결과.
            persons: YOLO 인물 감지 결과.
            protected_status: 보호 대상 추적 상태.
            threat: 위협 레벨 판정 결과.

        Returns:
            오버레이가 추가된 프레임 (입력과 동일한 객체).
        """
        h, w = frame.shape[:2]
        protected_id = protected_status.person_id if protected_status else None

        # 1. 일반 인물 박스 (파란색)
        for person in persons:
            if person.person_id == protected_id:
                continue
            x1, y1, x2, y2 = person.bbox
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 100, 0), 2)
            cv2.putText(
                frame, f"P{person.person_id}",
                (px1, max(py1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1,
            )

        # 2. 보호 대상 박스 (초록색, 이탈 시 어두운 초록)
        if protected_status and protected_status.bbox is not None:
            x1, y1, x2, y2 = protected_status.bbox
            px1, py1 = int(x1 * w), int(y1 * h)
            px2, py2 = int(x2 * w), int(y2 * h)
            box_color = (0, 255, 0) if protected_status.is_in_frame else (0, 100, 0)
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 3)
            label = f"보호대상 P{protected_status.person_id}"
            if not protected_status.is_in_frame:
                label += " (이탈)"
            cv2.putText(
                frame, label, (px1, max(py1 - 8, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2,
            )

        # 3. 위협 레벨 게이지 (화면 상단)
        self._draw_threat_gauge(frame, threat, y=10, w=w)

        # 4. SOS 진행바 (게이지 아래)
        self._draw_sos_bar(frame, sos_result, y=48, w=w)

        # 5. 보호 대상 이탈 경고 (화면 중앙 오버레이)
        if protected_status and not protected_status.is_in_frame and protected_status.bbox:
            self._draw_disappear_warning(frame, w, h)

        # 6. FPS 표시 (우하단)
        cv2.putText(
            frame, f"FPS: {self._fps:.1f}",
            (w - 120, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA,
        )

        return frame

    def _draw_threat_gauge(
        self,
        frame: np.ndarray,
        threat: ThreatResult,
        y: int,
        w: int,
    ) -> None:
        """화면 상단 위협 레벨 게이지."""
        x1, y1, x2, y2 = 10, y, w - 10, y + 30
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        filled = int((x2 - x1) * threat.score)
        if filled > 0:
            color = _THREAT_COLORS.get(threat.level, (0, 200, 0))
            cv2.rectangle(frame, (x1, y1), (x1 + filled, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        label = _THREAT_LABELS.get(threat.level, "")
        cv2.putText(
            frame, f"위협 레벨: {label} ({threat.score:.2f})",
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def _draw_sos_bar(
        self,
        frame: np.ndarray,
        sos_result: SOSDetectionResult,
        y: int,
        w: int,
    ) -> None:
        """SOS 감지 진행바."""
        if sos_result.is_detected:
            cv2.putText(
                frame, "[SOS 확정!]",
                (12, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA,
            )
        elif sos_result.is_pending:
            ratio = min(sos_result.held_duration / self._SOS_HOLD_SECONDS, 1.0)
            bar_w = int((w - 20) * ratio)
            if bar_w > 0:
                cv2.rectangle(frame, (10, y), (10 + bar_w, y + 18), (0, 200, 255), -1)
            cv2.rectangle(frame, (10, y), (w - 10, y + 18), (150, 150, 150), 1)
            text = (
                f"V사인 감지 중... "
                f"{sos_result.held_duration:.1f}초/{self._SOS_HOLD_SECONDS:.0f}초"
            )
            cv2.putText(
                frame, text, (12, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame, "SOS 대기 중",
                (12, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA,
            )

    def _draw_disappear_warning(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
    ) -> None:
        """보호 대상 화면 이탈 경고."""
        text = "보호 대상 이탈"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        tx = (w - tw) // 2
        ty = h // 2
        cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # 메인 루프
    # ------------------------------------------------------------------

    def run(self) -> None:
        """파이프라인 메인 루프를 실행합니다.

        q 또는 ESC 키를 누르면 종료됩니다.
        동영상 파일은 마지막 프레임 이후 자동 종료됩니다.

        Raises:
            RuntimeError: 영상 소스를 열 수 없을 때.
        """
        cap_source = self._source if isinstance(self._source, int) else str(self._source)
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            raise RuntimeError(f"영상 소스를 열 수 없습니다: {self._source}")

        logger.info("파이프라인 시작 — 종료: q / ESC")

        with self._hand_tracker:
            while True:
                t0 = time.monotonic()

                ret, frame = cap.read()
                if not ret:
                    logger.info("영상 종료.")
                    break

                if self._is_webcam:
                    frame = cv2.flip(frame, 1)

                _step_times: dict[str, float] = {}

                _t = time.monotonic()
                tracking = self._process_hand_tracking(frame)
                _step_times["손 추적"] = (time.monotonic() - _t) * 1000

                _t = time.monotonic()
                gestures = self._process_gesture(tracking)
                _step_times["제스처 인식"] = (time.monotonic() - _t) * 1000

                _t = time.monotonic()
                sos = self._process_sos(gestures)
                _step_times["SOS 감지"] = (time.monotonic() - _t) * 1000

                _t = time.monotonic()
                persons, threat = self._process_yolo(frame, gestures)
                _step_times["YOLO+위협"] = (time.monotonic() - _t) * 1000

                self._process_registration(sos, gestures, persons)
                protected = self._process_protected_tracking(persons)

                try:
                    self._protector.respond(threat)
                except Exception:
                    logger.exception("Protector 오류")

                # 자동 녹화 갱신
                if self._recorder is not None:
                    try:
                        self._recorder.update(frame, threat.level)
                    except Exception:
                        logger.exception("VideoRecorder 오류")

                # 경찰 신고 시뮬레이션
                if self._alert_manager is not None:
                    try:
                        self._alert_manager.handle_threat(threat)
                    except Exception:
                        logger.exception("AlertManager 오류")

                # API 서버 상태 갱신
                try:
                    broadcaster.update_sync(
                        threat_level=threat.level.name,
                        threat_score=threat.score,
                        is_recording=self._recorder.is_recording if self._recorder else False,
                        protected_person_id=self._protected_tracker.protected_id,
                        is_protected_in_frame=(
                            protected.is_in_frame if protected else False
                        ),
                        sos_detected=sos.is_detected,
                        fps=self._fps,
                        inference_times=_step_times,
                        sos_pending_duration=sos.held_duration if sos.is_pending else 0.0,
                        sos_hold_seconds=self._SOS_HOLD_SECONDS,
                        protected_track_start=self._protection_start_time,
                    )
                except Exception:
                    logger.exception("상태 브로드캐스터 오류")

                try:
                    self.draw_overlay(frame, sos, persons, protected, threat)
                except Exception:
                    logger.exception("오버레이 그리기 오류")

                # REC 표시
                if self._recorder is not None:
                    try:
                        self._recorder.draw_rec_indicator(frame)
                    except Exception:
                        logger.exception("REC 표시 오류")

                # 대시보드용 프레임 JPEG 인코딩
                try:
                    ok, jpg_buf = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                    )
                    if ok:
                        broadcaster.set_frame_jpg(jpg_buf.tobytes())
                except Exception:
                    logger.exception("프레임 인코딩 오류")

                elapsed = time.monotonic() - t0
                instant = 1.0 / elapsed if elapsed > 0 else 30.0
                self._fps = self._fps * 0.9 + instant * 0.1

                cv2.imshow("PSDS AI Pipeline", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

        cap.release()
        cv2.destroyAllWindows()

        if self._recorder is not None:
            self._recorder.stop()


# ------------------------------------------------------------------
# CLI 진입점
# ------------------------------------------------------------------

def main() -> None:
    """``python -m src.pipeline`` 진입점."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="PSDS AI 파이프라인")
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="동영상 파일 경로. 미지정 시 웹캠(0) 사용.",
    )
    parser.add_argument(
        "--hand-model",
        default="models/hand_landmarker.task",
        help="MediaPipe 손 랜드마크 모델 경로.",
    )
    args = parser.parse_args()

    source: int | str = args.input if args.input is not None else 0
    Pipeline(source=source, hand_model_path=args.hand_model).run()


if __name__ == "__main__":
    main()
