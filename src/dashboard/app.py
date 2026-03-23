"""src/dashboard/app.py — PSDS AI 실시간 모니터링 대시보드.

Usage::

    streamlit run src/dashboard/app.py

환경변수 PSDS_API_URL 로 서버 주소를 변경할 수 있습니다 (기본: http://localhost:8000).
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard import api_client

# ------------------------------------------------------------------
# 페이지 설정 (반드시 첫 번째 st 명령)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="PSDS 대시보드",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 세션 상태 초기화
# ------------------------------------------------------------------
_DEFAULTS: dict = {
    "fps_history": [],
    "auto_refresh": True,
    "refresh_count": 0,
    "selected_recording": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ------------------------------------------------------------------
# CSS — 위협 게이지 애니메이션, 반응형 레이아웃
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .threat-gauge-track {
        background: #2e2e2e;
        border-radius: 8px;
        padding: 3px;
        margin-bottom: 6px;
    }
    .threat-gauge-fill {
        height: 22px;
        border-radius: 6px;
        transition: width 0.4s ease, background 0.4s ease;
    }
    .rec-badge {
        display: inline-block;
        background: #c0392b;
        color: white;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: bold;
        animation: blink 1s step-start infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }
    @media (max-width: 768px) {
        .block-container { padding: 1rem 0.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# 서버 상태 조회
# ------------------------------------------------------------------
status = api_client.fetch_status()
connected = status is not None

# FPS 기록 업데이트
if connected and status:
    hist = st.session_state.fps_history
    hist.append(status.get("fps", 0.0))
    if len(hist) > 60:
        hist.pop(0)
    st.session_state.fps_history = hist

# ------------------------------------------------------------------
# 사이드바
# ------------------------------------------------------------------
with st.sidebar:
    st.title("PSDS 대시보드")

    if connected:
        st.success("서버 연결됨")
    else:
        st.error("서버 연결 끊김\n재연결 시도 중...")

    st.divider()
    st.toggle("자동 새로고침 (1초)", key="auto_refresh")

    if st.button("지금 새로고침", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption(f"새로고침: {st.session_state.refresh_count}회")
    st.caption(f"갱신 시각: {datetime.now().strftime('%H:%M:%S')}")
    st.divider()
    st.caption(f"API: `{api_client.API_BASE}`")

# ------------------------------------------------------------------
# 탭
# ------------------------------------------------------------------
tab_monitor, tab_recordings, tab_system, tab_settings = st.tabs(
    ["모니터링", "녹화 관리", "시스템", "설정"]
)


# ==================================================================
# Tab 1: 모니터링
# ==================================================================
with tab_monitor:
    if not connected:
        st.warning(
            "FastAPI 서버에 연결할 수 없습니다. "
            "`uvicorn src.api.server:app --port 8000` 으로 서버를 먼저 시작하세요."
        )

    col_frame, col_status = st.columns([3, 2], gap="medium")

    # ---- 좌측: 라이브 피드 ----
    with col_frame:
        st.subheader("실시간 AI 인식")

        frame_bytes = api_client.fetch_frame() if connected else None
        if frame_bytes:
            st.image(frame_bytes, use_column_width=True, caption="라이브 피드")
        else:
            st.info("파이프라인이 실행 중이 아닙니다.")
            uploaded = st.file_uploader(
                "동영상 파일 업로드 (파이프라인 미실행 시 데모)",
                type=["mp4", "avi", "mov"],
                key="video_upload",
            )
            if uploaded:
                st.video(uploaded)

    # ---- 우측: 상태 패널 ----
    with col_status:
        st.subheader("현재 상태")

        if connected and status:
            level = status.get("threat_level", "NONE")
            score = float(status.get("threat_score", 0.0))

            # 위협 레벨 라벨·색상
            _labels = {
                "NONE": "정상", "LOW": "의심",
                "MEDIUM": "위협", "HIGH": "고위협", "CRITICAL": "긴급",
            }
            _gauge_colors = {
                "NONE": "#27ae60", "LOW": "#f1c40f",
                "MEDIUM": "#e67e22", "HIGH": "#e74c3c", "CRITICAL": "#8e44ad",
            }
            _metric_deltas = {
                "NONE": None, "LOW": "+주의",
                "MEDIUM": "+경고", "HIGH": "+위험!", "CRITICAL": "+긴급!",
            }

            label = _labels.get(level, level)
            color = _gauge_colors.get(level, "#27ae60")

            st.metric(
                "위협 레벨",
                label,
                delta=_metric_deltas.get(level),
                delta_color="inverse" if level not in ("NONE", "LOW") else "off",
            )

            # 커스텀 게이지
            pct = int(score * 100)
            st.markdown(
                f"""
                <div class="threat-gauge-track">
                  <div class="threat-gauge-fill"
                       style="width:{pct}%; background:{color};"></div>
                </div>
                <p style="text-align:center; color:#aaa; font-size:0.82em; margin:0;">
                  위협 점수 {score:.2f}
                </p>
                """,
                unsafe_allow_html=True,
            )

            st.divider()

            # SOS 상태
            sos_detected = status.get("sos_detected", False)
            sos_dur = float(status.get("sos_pending_duration", 0.0))
            hold_secs = float(status.get("sos_hold_seconds", 3.0))

            st.markdown("**SOS 상태**")
            if sos_detected:
                st.success("SOS 확정됨!")
            elif sos_dur > 0:
                ratio = min(sos_dur / hold_secs, 1.0)
                st.info(f"V사인 감지 중 — {sos_dur:.1f}s / {hold_secs:.0f}s")
                st.progress(ratio, text=f"{ratio*100:.0f}%")
            else:
                st.caption("SOS 대기 중")

            st.divider()

            # 보호 대상 정보
            pid = status.get("protected_person_id")
            in_frame = status.get("is_protected_in_frame", False)
            track_start = status.get("protected_track_start")

            st.markdown("**보호 대상**")
            if pid is not None:
                st.metric("ID", f"P{pid}")
                if in_frame:
                    st.success("추적 중 (화면 내)")
                else:
                    st.error("화면 이탈!")

                if track_start:
                    try:
                        dt = datetime.fromisoformat(track_start)
                        elapsed = int((datetime.now() - dt).total_seconds())
                        m, s = divmod(elapsed, 60)
                        st.caption(f"추적 시간: {m:02d}:{s:02d}")
                    except (ValueError, OSError):
                        pass
            else:
                st.caption("미지정")

            st.divider()

            # 녹화 상태 & FPS
            is_recording = status.get("is_recording", False)
            fps_val = float(status.get("fps", 0.0))

            col_rec, col_fps = st.columns(2)
            with col_rec:
                if is_recording:
                    st.markdown(
                        '<span class="rec-badge">● REC</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("녹화 대기")
            with col_fps:
                st.metric("FPS", f"{fps_val:.1f}")
        else:
            st.info("서버 연결 대기 중...")


# ==================================================================
# Tab 2: 녹화 관리
# ==================================================================
with tab_recordings:
    st.subheader("저장된 영상")

    recordings = api_client.fetch_recordings() if connected else []

    if not recordings:
        if not connected:
            st.warning("서버에 연결해야 영상 목록을 볼 수 있습니다.")
        else:
            st.info("저장된 영상이 없습니다.")
    else:
        col_list, col_player = st.columns([2, 3], gap="medium")

        with col_list:
            st.write(f"**총 {len(recordings)}개 영상**")

            for rec in recordings:
                fname = rec["filename"]
                size_mb = rec["size_bytes"] / 1_048_576
                created = rec.get("created_at", "")[:19].replace("T", " ")
                is_sel = st.session_state.selected_recording == fname

                with st.container():
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        btn_label = f"{'▶ ' if is_sel else ''}{fname}"
                        if st.button(btn_label, key=f"sel_{fname}", use_container_width=True):
                            st.session_state.selected_recording = fname
                            st.rerun()
                        st.caption(f"{size_mb:.1f} MB · {created}")
                    with c2:
                        if st.button("삭제", key=f"del_{fname}"):
                            if api_client.delete_recording(fname):
                                if st.session_state.selected_recording == fname:
                                    st.session_state.selected_recording = None
                                st.rerun()
                            else:
                                st.error("삭제 실패")

        with col_player:
            sel = st.session_state.selected_recording
            if sel:
                filepath = Path("recordings") / sel
                if filepath.exists():
                    st.video(str(filepath))
                    st.caption(sel)
                else:
                    # API 다운로드 URL로 폴백
                    st.info(f"로컬 파일 없음 — API에서 스트리밍: {sel}")
                    st.markdown(
                        f"[영상 다운로드]({api_client.API_BASE}/recordings/{sel})"
                    )
            else:
                st.info("목록에서 영상을 선택하면 여기서 재생됩니다.")


# ==================================================================
# Tab 3: 시스템 모니터링
# ==================================================================
with tab_system:
    col_fps_chart, col_mem = st.columns(2, gap="medium")

    with col_fps_chart:
        st.subheader("FPS 실시간 그래프")
        if st.session_state.fps_history:
            df_fps = pd.DataFrame(
                {"FPS": st.session_state.fps_history},
                index=range(len(st.session_state.fps_history)),
            )
            st.line_chart(df_fps, height=200)
            cur_fps = st.session_state.fps_history[-1]
            st.caption(f"현재 {cur_fps:.1f} FPS · 최근 {len(st.session_state.fps_history)}개 샘플")
        else:
            st.info("FPS 데이터 수집 중... (파이프라인 실행 필요)")

    with col_mem:
        st.subheader("메모리 사용량")
        try:
            import psutil

            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / 1_048_576
            total_mb = psutil.virtual_memory().total / 1_048_576
            used_mb = psutil.virtual_memory().used / 1_048_576
            pct = psutil.virtual_memory().percent

            st.metric("프로세스 RSS", f"{mem_mb:.0f} MB")
            st.progress(pct / 100, text=f"시스템 RAM {used_mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)")
        except ImportError:
            st.info("psutil 패키지 필요: `pip install psutil`")

    st.divider()

    # 모듈별 추론 시간
    st.subheader("모듈별 추론 시간")
    if connected and status:
        times: dict = status.get("inference_times", {})
        if times:
            df_times = pd.DataFrame(
                [{"모듈": k, "추론 시간 (ms)": round(v, 1)} for k, v in times.items()]
            )
            st.dataframe(df_times, use_container_width=True, hide_index=True)
            total_ms = sum(times.values())
            st.caption(f"총 파이프라인 처리 시간: {total_ms:.1f} ms / 프레임")
        else:
            st.info("추론 시간 없음 — 파이프라인을 실행하면 표시됩니다.")
    else:
        st.info("서버 연결 필요")

    st.divider()

    # 알림 이력 타임라인
    st.subheader("알림 이력")
    alerts = api_client.fetch_alerts() if connected else []
    if alerts:
        df_alerts = pd.DataFrame(alerts)
        # 표시할 컬럼 정렬
        _want = ("timestamp", "level", "score", "action", "reasons")
        show_cols = [c for c in _want if c in df_alerts.columns]
        st.dataframe(df_alerts[show_cols], use_container_width=True, hide_index=True)
        if st.button("알림 이력 초기화"):
            if connected:
                try:
                    import httpx as _hx
                    _hx.delete(f"{api_client.API_BASE}/alerts", timeout=2.0)
                    st.rerun()
                except Exception:
                    st.error("초기화 실패")
    else:
        st.info("알림 이력이 없습니다.")


# ==================================================================
# Tab 4: 설정
# ==================================================================
with tab_settings:
    st.subheader("파이프라인 설정")

    if not connected:
        st.warning("서버 미연결 — 설정을 변경하려면 서버를 먼저 시작하세요.")

    current = api_client.fetch_settings() if connected else None
    defaults: dict = current or {
        "sos_hold_seconds": 3.0,
        "yolo_confidence": 0.5,
        "threat_medium_threshold": 0.45,
        "threat_high_threshold": 0.75,
        "record_trigger_level": "MEDIUM",
    }

    with st.form("settings_form"):
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            sos_hold = st.slider(
                "SOS 유지 시간 (초)",
                min_value=1.0, max_value=10.0,
                value=float(defaults["sos_hold_seconds"]),
                step=0.5,
                help="V사인을 이 시간만큼 유지해야 SOS 확정",
            )
            yolo_conf = st.slider(
                "YOLO Confidence 임계값",
                min_value=0.1, max_value=0.9,
                value=float(defaults["yolo_confidence"]),
                step=0.05,
                help="낮을수록 더 많은 사람을 감지 (오탐 증가)",
            )

        with col_b:
            threat_medium = st.slider(
                "위협 MEDIUM 임계값",
                min_value=0.1, max_value=0.9,
                value=float(defaults["threat_medium_threshold"]),
                step=0.05,
                help="이 점수 이상이면 위협 레벨 MEDIUM",
            )
            threat_high = st.slider(
                "위협 HIGH 임계값",
                min_value=0.1, max_value=1.0,
                value=float(defaults["threat_high_threshold"]),
                step=0.05,
                help="이 점수 이상이면 위협 레벨 HIGH",
            )

        _levels = ["MEDIUM", "HIGH", "CRITICAL"]
        record_level = st.selectbox(
            "녹화 자동 시작 레벨",
            options=_levels,
            index=_levels.index(str(defaults.get("record_trigger_level", "MEDIUM"))),
            help="이 레벨 이상일 때 자동 녹화를 시작합니다",
        )

        submitted = st.form_submit_button("설정 적용", use_container_width=True)
        if submitted:
            new_cfg = {
                "sos_hold_seconds": sos_hold,
                "yolo_confidence": yolo_conf,
                "threat_medium_threshold": threat_medium,
                "threat_high_threshold": threat_high,
                "record_trigger_level": record_level,
            }
            if connected and api_client.update_settings(new_cfg):
                st.success("설정이 적용됐습니다.")
            elif not connected:
                st.warning("서버에 연결 후 적용 가능합니다.")
            else:
                st.error("설정 적용 실패")

    # 설정 설명
    with st.expander("설정 항목 설명"):
        st.markdown(
            """
            | 항목 | 기본값 | 설명 |
            |------|--------|------|
            | SOS 유지 시간 | 3초 | V사인 제스처를 유지해야 하는 시간 |
            | YOLO Confidence | 0.5 | 인물 감지 확신도 임계값 |
            | 위협 MEDIUM 임계값 | 0.45 | 위협 점수가 이 값 이상이면 경고 + 녹화 시작 |
            | 위협 HIGH 임계값 | 0.75 | 위협 점수가 이 값 이상이면 경찰 신고 |
            | 녹화 자동 시작 레벨 | MEDIUM | 이 레벨부터 자동 녹화를 시작 |
            """
        )


# ------------------------------------------------------------------
# 세션 카운터 증가 + 자동 새로고침
# ------------------------------------------------------------------
st.session_state.refresh_count += 1

if st.session_state.get("auto_refresh", True):
    time.sleep(1)
    st.rerun()
