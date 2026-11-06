#!/usr/bin/env python3
"""
AI CrowdGuard — 라즈베리파이 기반 AI 공간 혼잡도 모니터링 에이전트

카메라 영상을 YOLOv8n이 Raspberry Pi 안에서 직접 분석하여 사람(person) 수를 세고,
  - 인원 수에 따라 혼잡도(EMPTY / LOW / MEDIUM / FULL / OVER)를 판단
  - 정원을 초과한 상태가 일정 시간 이상 지속되면 경고
  - 일정 간격마다 평균/최대 인원을 CSV로 기록
한다. 종료 시 최대 인원, 평균 인원, 혼잡도별 시간을 요약한다.

실행:  python crowdguard.py      (종료: 화면에서 q 키)
"""

import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ──────────────────────────────────────────────
# 설정값 (시연 상황에 맞게 조절)
# ──────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
FRAME_SIZE = (640, 480)
IMG_SIZE = 320
CONF = 0.40

CAPACITY = 3              # 공간 정원 (시연용: 3명)
SMOOTH_SEC = 1.5          # 이 시간 동안의 인원 수 중앙값을 사용 (깜빡임 방지)
OVER_WARN_SEC = 3.0       # 정원 초과가 이 시간(초) 이상 지속되면 경고
LOG_INTERVAL = 10.0       # 이 간격(초)마다 평균/최대 인원 기록 (실사용 시 60 이상)

BUZZER_PIN = None         # 부저 연결 시 GPIO 번호 (예: 17)
LOG_DIR = "logs"

LEVEL_COLORS = {          # BGR
    "EMPTY": (200, 200, 200),
    "LOW": (0, 200, 0),
    "MEDIUM": (0, 220, 255),
    "FULL": (0, 140, 255),
    "OVER": (0, 0, 255),
}


# ──────────────────────────────────────────────
# 에이전트: 인식 → 기억 → 판단 → 행동
# ──────────────────────────────────────────────
class CrowdAgent:
    def __init__(self, now=None):
        now = now or time.time()
        self.start = now
        self.history = deque()        # 최근 (시각, 인원 수)
        self.count = 0                # 보정된 현재 인원 수
        self.level = "EMPTY"
        self.level_since = now
        self.level_time = {k: 0.0 for k in LEVEL_COLORS}

        self.warned = False
        self.over_warnings = 0
        self.peak = (0, now)          # (최대 인원, 시각)

        self.bucket_start = now       # 구간 기록용
        self.bucket = []
        self.all_counts = []

        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = os.path.join(LOG_DIR, datetime.fromtimestamp(now).strftime("crowd_%Y%m%d_%H%M%S.csv"))
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["time", "event", "detail"])

    def log(self, now, event, detail=""):
        stamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        print(f"[{stamp}] {event} {detail}")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([stamp, event, detail])

    @staticmethod
    def level_of(n):
        if n == 0:
            return "EMPTY"
        if n > CAPACITY:
            return "OVER"
        r = n / CAPACITY
        if r < 0.5:
            return "LOW"
        if r < 1.0:
            return "MEDIUM"
        return "FULL"

    def update(self, raw_count, now):
        """한 프레임의 사람 수를 받아 혼잡도를 판단하고, 경고 여부(True/False)를 반환."""
        # 1) 기억: 최근 인원 수 기록 → 중앙값으로 보정
        self.history.append((now, raw_count))
        while now - self.history[0][0] > SMOOTH_SEC:
            self.history.popleft()
        counts = sorted(c for _, c in self.history)
        self.count = counts[len(counts) // 2]

        if self.count > self.peak[0]:
            self.peak = (self.count, now)
        self.bucket.append(self.count)
        self.all_counts.append(self.count)

        # 2) 판단: 혼잡도
        new_level = self.level_of(self.count)
        if new_level != self.level:
            self.level_time[self.level] += now - self.level_since
            self.log(now, "LEVEL", f"{self.level} -> {new_level} ({self.count}/{CAPACITY})")
            self.level, self.level_since = new_level, now
            self.warned = False

        # 3) 시간 조건: 정원 초과 지속
        alert = self.level == "OVER" and now - self.level_since >= OVER_WARN_SEC
        if alert and not self.warned:
            self.warned = True
            self.over_warnings += 1
            self.log(now, "WARNING_OVER", f"{self.count}/{CAPACITY} #{self.over_warnings}")

        # 4) 주기 기록
        if now - self.bucket_start >= LOG_INTERVAL:
            avg = sum(self.bucket) / len(self.bucket)
            self.log(now, "STATS", f"avg={avg:.1f} max={max(self.bucket)}")
            self.bucket_start, self.bucket = now, []

        return alert

    def finish(self, now):
        self.level_time[self.level] += now - self.level_since
        self.level_since = now
        avg = sum(self.all_counts) / len(self.all_counts) if self.all_counts else 0.0
        summary = {
            "duration_sec": now - self.start,
            "avg_count": avg,
            "peak_count": self.peak[0],
            "peak_time": datetime.fromtimestamp(self.peak[1]).strftime("%H:%M:%S"),
            "over_warnings": self.over_warnings,
            **{f"{k.lower()}_sec": v for k, v in self.level_time.items()},
        }
        for k, v in summary.items():
            self.log(now, "SUMMARY", f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}")
        return summary


# ──────────────────────────────────────────────
# 화면 표시 / 부저
# ──────────────────────────────────────────────
def draw_status(frame, agent, alert, fps):
    w, h = frame.shape[1], frame.shape[0]
    color = LEVEL_COLORS[agent.level]

    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"PEOPLE: {agent.count} / {CAPACITY}   [{agent.level}]",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 혼잡도 막대
    bar_w = w - 20
    fill = int(bar_w * min(agent.count / CAPACITY, 1.0))
    cv2.rectangle(frame, (10, 40), (10 + bar_w, 52), (80, 80, 80), -1)
    cv2.rectangle(frame, (10, 40), (10 + fill, 52), color, -1)
    cv2.putText(frame, f"Peak {agent.peak[0]} | Over warn {agent.over_warnings} | {fps:.1f} FPS",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if alert:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, "WARNING: OVER CAPACITY!", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def make_buzzer():
    if BUZZER_PIN is None:
        return None
    try:
        from gpiozero import Buzzer
        return Buzzer(BUZZER_PIN)
    except Exception as e:
        print(f"[buzzer] 사용 안 함: {e}")
        return None


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────
def main():
    model = YOLO(MODEL_PATH)
    person_id = [i for i, n in model.names.items() if n == "person"]

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE}))
    cam.start()
    time.sleep(1)

    buzzer = make_buzzer()
    agent = CrowdAgent()
    prev, fps = time.time(), 0.0
    print("CrowdGuard 시작 — 화면에서 q 키로 종료")

    try:
        while True:
            frame = cam.capture_array()
            result = model(frame, imgsz=IMG_SIZE, conf=CONF, classes=person_id, verbose=False)[0]
            raw = len(result.boxes)

            now = time.time()
            alert = agent.update(raw, now)

            if buzzer:
                buzzer.on() if alert else buzzer.off()

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            view = result.plot()
            draw_status(view, agent, alert, fps)
            cv2.imshow("AI CrowdGuard", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if buzzer:
            buzzer.off()
        cam.stop()
        cv2.destroyAllWindows()
        s = agent.finish(time.time())

        print("\n===== 세션 요약 =====")
        print(f"측정 시간     : {s['duration_sec'] / 60:.1f}분")
        print(f"평균 인원     : {s['avg_count']:.1f}명")
        print(f"최대 인원     : {s['peak_count']}명 ({s['peak_time']})")
        print(f"정원 초과 경고: {s['over_warnings']}회")
        for lv in LEVEL_COLORS:
            print(f"  {lv:7s}: {s[lv.lower() + '_sec']:.0f}초")
        print(f"로그 저장     : {agent.path}")


if __name__ == "__main__":
    main()
