# PSDS AI Recognition Pipeline

## 프로젝트 개요
PSDS(Personal Safety Drone System)의 AI 인식 파이프라인.
긴급 상황에서 드론이 자동 출동 후 AI로 보호 대상 식별 + 위협 감지.
현재는 드론 없이 웹캠으로 AI 파이프라인만 개발하는 단계.

## 동작 흐름
1. 웹캠 영상 → MediaPipe로 손 21개 관절 추출 + 칼만 필터
2. LSTM으로 손 시퀀스 분석 → SOS 모션 감지 → 보호 대상 식별
3. YOLOv8로 사람/위험물 감지 + 행동 분석 → 위협 레벨(0~3) 판단
4. 위협 시 자동 녹화 + 경고 + 경찰/지인 알림
5. FastAPI 상태 API + Streamlit 대시보드

## SOS 모션 인식
- 입력: 30프레임 x 21관절 x 3좌표(x,y,z)
- 모델: PyTorch LSTM 시퀀스 분류
- 2단계 확인: confidence > 0.85 감지 후 0.5초 뒤 재확인
- 폴백: 실패 시 GPS 기반 가장 가까운 인물을 보호 대상으로 지정

## 위협 레벨
- 0: 정상 | 1: 의심→모니터링 | 2: 위협→경고+녹화 | 3: 긴급→알림+증거업로드

## 기술 규칙
- Python 3.11, 타입힌트 필수, Google style docstring
- pytest 테스트 필수, ruff 린트
- 의존방향: hand_tracking→gesture_recognition→threat_detection→protection→streaming
- 모델 가중치 models/ (git 제외), 시크릿 .env 관리

## 라이브러리
mediapipe, ultralytics(YOLOv8), torch, opencv-python, filterpy, fastapi, streamlit

## 향후 확장
Gazebo+PX4 드론 연동, ROS2 스웜, TensorRT 최적화, Flutter 앱 연동

## 사용자 소통 규칙
- 파일 생성/수정/삭제 승인을 요청할 때 반드시 한글로 설명할 것
- 형식: "📝 [작업 내용] - [이 파일이 하는 역할] - 승인하시겠습니까?"
- 예시: "📝 tracker.py 수정 - 칼만 필터 노이즈 제거 기능 추가 - 승인하시겠습니까?"
- 모든 진행 상황 보고도 한글로 할 것
