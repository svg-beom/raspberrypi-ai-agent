# Raspberry Pi AI Agent — 온디바이스 YOLO 기반 3가지 AI 에이전트

Raspberry Pi와 카메라 모듈, **YOLOv8n 딥러닝 모델**을 이용해 만든 3가지 AI 에이전트입니다.
외부 AI API 없이 Raspberry Pi 안에서 직접 추론하는 **On-device(Edge) AI** 방식으로 동작합니다.

| 작품 | 파일 | 주요 기능 | 상태 |
|---|---|---|---|
| **AI CrowdGuard** | `crowdguard.py` | 공간 인원 수 측정 및 혼잡도 판단, 정원 초과 경고 | ⭐ 최종 작품 |
| **AI FocusGuard** | `focusguard.py` | 공부 중 휴대폰 사용·자리 비움 감지 및 경고 | ✅ 동작 확인 |
| **AI PillGuard** | `pillguard.py` | 약통을 들었다 놓는 동작으로 약 복용 기록 | 🧪 실험 (인식 불안정) |

---

## 공통 구조: 인지 → 기억 → 판단 → 행동

```text
[Perception : 인지]   카메라 + YOLOv8n → 사람 / 물체 위치 인식
        ↓
[Memory : 기억]       최근 인식 결과, 상태 지속 시간, 통계
        ↓
[Decision : 판단]     상태 결정 및 경고 조건 확인
        ↓
[Action : 행동]       화면 표시·경고, CSV 기록
```

YOLOv8n이 영상 속 객체를 인식하고, 그 결과를 바탕으로 직접 설계한 규칙 기반 로직이 상태를 판단합니다.

---

## 1. AI CrowdGuard — 공간 혼잡도 모니터링 (최종 작품)

카메라에 보이는 사람 수를 세서 공간의 혼잡도를 판단합니다.

| 인원 (정원 3명 기준) | 혼잡도 | 화면 색 |
|---|---|---|
| 0명 | `EMPTY` | 회색 |
| 정원의 50% 미만 | `LOW` | 초록 |
| 50% 이상 ~ 100% 미만 | `MEDIUM` | 노랑 |
| 정원 | `FULL` | 주황 |
| 정원 초과 | `OVER` | 빨강 |

- **중앙값 보정**: 최근 1.5초 인원 수의 중앙값을 사용해 순간적인 오인식 제거
- **정원 초과 경고**: `OVER` 상태가 3초 이상 지속되면 `WARNING: OVER CAPACITY!`
- **통계 기록**: 혼잡도 변화, 주기별 평균/최대 인원, 종료 시 세션 요약을 CSV로 저장

## 2. AI FocusGuard — 공부 집중 감시

| 상황 | 상태 |
|---|---|
| 자리에 앉아 공부 중 | `FOCUS` |
| 휴대폰을 5초 이상 봄 | `PHONE` → 경고 |
| 자리를 15초 이상 비움 | `AWAY` → 경고 |

종료 시 집중 / 휴대폰 / 자리 비움 시간을 요약하고 CSV로 저장합니다.

## 3. AI PillGuard — 약 복용 체크 (실험)

약통(`bottle` / `cup`)의 기준 위치를 기억해 두고, 사람이 약통을 들었다가 다시 내려놓으면 복용으로 기록합니다.
복용 시간이 지났는데 기록이 없으면 알림, 이미 복용했는데 다시 들면 중복 복용 경고를 띄웁니다.

> 약통 크기·각도에 따라 YOLO 인식이 불안정해, 사람 인식만으로 안정적으로 동작하는 CrowdGuard를 최종 작품으로 선택했습니다.

---

## 하드웨어 / 소프트웨어

- Raspberry Pi 4 Model B
- Raspberry Pi Camera Module (CSI 연결)
- Raspberry Pi OS, Python 3 가상환경 (`--system-site-packages`)
- Picamera2, OpenCV, PyTorch, Ultralytics (YOLOv8n)

## 설치

```bash
python3 -m venv --system-site-packages ~/myenv
source ~/myenv/bin/activate
pip install ultralytics opencv-python
```

`yolov8n.pt` 모델 파일은 처음 실행할 때 자동으로 다운로드됩니다.

## 실행

```bash
source ~/myenv/bin/activate
cd ~/ai_focusguard

python crowdguard.py     # 혼잡도 모니터링
python focusguard.py     # 공부 집중 감시
python pillguard.py      # 약 복용 체크
```

화면에서 `q` 키를 누르면 종료되고, 세션 요약이 출력되며 `logs/` 폴더에 CSV가 저장됩니다.

## 파일 구성

```text
ai_focusguard/
├─ crowdguard.py
├─ focusguard.py
├─ pillguard.py
├─ README.md
├─ .gitignore
└─ logs/            # 실행 기록 (저장소에는 포함하지 않음)
```

---

## API Key 없이 AI가 동작하는 이유

OpenAI, Claude 같은 외부 AI 서비스를 호출하지 않고, YOLOv8n 가중치 파일을 Raspberry Pi의 CPU에서 직접 실행합니다.

- API Key와 사용 요금이 필요 없음
- 영상이 외부로 전송되지 않아 개인정보 측면에서 유리함

## 한계와 확장 아이디어

| 한계 | 확장 방향 |
|---|---|
| 판단 로직이 고정 규칙 | LLM API로 기록을 분석해 맞춤 피드백 생성 |
| 결과가 CSV에만 저장 | 웹 대시보드로 시간대별 그래프 시각화 |
| 작은 물체 인식 불안정 (PillGuard) | 커스텀 데이터로 YOLO 재학습 |
