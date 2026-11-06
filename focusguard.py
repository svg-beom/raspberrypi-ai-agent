#!/usr/bin/env python3
"""
AI FocusGuard — 라즈베리파이 기반 AI 공부 집중 감시 에이전트

카메라 영상을 YOLOv8n이 Raspberry Pi 안에서 직접 분석하여
사람(person)과 휴대폰(cell phone)을 인식하고,
  - 휴대폰을 일정 시간 이상 보고 있으면  → "PHONE" 경고
  - 자리를 일정 시간 이상 비우면          → "AWAY" 경고
를 띄운다. 종료 시 집중/폰/자리비움 시간 통계를 CSV로 저장한다.

실행:  python focusguard.py      (종료: 화면에서 q 키)
"""

import csv
import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ──────────────────────────────────────────────
# 설정값 (시연 상황에 맞게 조절)
# ──────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
FRAME_SIZE = (640, 480)   # 카메라 해상도
IMG_SIZE = 320            # YOLO 입력 크기 (작을수록 빠름)
CONF = 0.40               # 최소 신뢰도

PERSON_GRACE = 3.0        # 사람이 이 시간(초) 이상 안 보여야 '자리 비움'으로 봄
PHONE_GRACE = 1.5         # 폰이 이 시간(초) 이상 안 보여야 '폰 내려놓음'으로 봄
PHONE_WARN_SEC = 5.0      # 폰을 이 시간(초) 이상 보면 경고
AWAY_WARN_SEC = 15.0      # 자리를 이 시간(초) 이상 비우면 경고 (시연용으로 짧게)

BUZZER_PIN = None         # 부저 연결 시 GPIO 번호 입력 (예: 17)
LOG_DIR = "logs"

TARGET_CLASSES = ["person", "cell phone", "book", "laptop"]

STATE_COLORS = {          # OpenCV는 BGR 순서
    "WAITING": (200, 200, 200),
    "FOCUS": (0, 200, 0),
    "PHONE": (0, 140, 255),
    "AWAY": (0, 0, 255),
}


# ──────────────────────────────────────────────
# 에이전트: 인식 결과 → 상태 기억 → 판단 → 행동
# ──────────────────────────────────────────────
class FocusAgent:
    def __init__(self):
        now = time.time()
        self.session_start = now
        self.state = "WAITING"
        self.state_since = now

        self.last_person = None   # 사람을 마지막으로 본 시각
        self.last_phone = None    # 폰을 마지막으로 본 시각
        self.phone_start = None   # 이번 폰 사용이 시작된 시각

        self.totals = {"FOCUS": 0.0, "PHONE": 0.0, "AWAY": 0.0}
        self.warn_count = {"PHONE": 0, "AWAY": 0}
        self.warned = False       # 현재 상태에서 이미 경고를 기록했는지
        self.events = []          # (시각, 이벤트, 상세)

    def log(self, now, event, detail=""):
        stamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        self.events.append((stamp, event, detail))
        print(f"[{stamp}] {event} {detail}")

    def update(self, labels, now):
        """한 프레임의 인식 결과(labels)를 받아 상태를 갱신하고, 경고 종류를 반환."""
        # 1) 기억 갱신
        if "person" in labels:
            self.last_person = now
        if "cell phone" in labels:
            if self.last_phone is None or now - self.last_phone > PHONE_GRACE:
                self.phone_start = now          # 새 폰 사용 시작
            self.last_phone = now

        person_here = self.last_person is not None and now - self.last_person < PERSON_GRACE
        phone_here = self.last_phone is not None and now - self.last_phone < PHONE_GRACE

        # 2) 상태 판단
        if self.last_person is None:
            new_state = "WAITING"
        elif not person_here:
            new_state = "AWAY"
        elif phone_here:
            new_state = "PHONE"
        else:
            new_state = "FOCUS"

        self._transition(new_state, now)

        # 3) 시간 조건에 따른 경고 판단
        alert = None
        if self.state == "PHONE" and now - self.phone_start >= PHONE_WARN_SEC:
            alert = "PHONE"
        elif self.state == "AWAY" and now - self.state_since >= AWAY_WARN_SEC:
            alert = "AWAY"

        if alert and not self.warned:
            self.warned = True
            self.warn_count[alert] += 1
            self.log(now, f"WARNING_{alert}", f"#{self.warn_count[alert]}")
        return alert

    def _transition(self, new_state, now):
        if new_state == self.state:
            return
        if self.state in self.totals:
            self.totals[self.state] += now - self.state_since
        self.log(now, "STATE", f"{self.state} -> {new_state}")
        self.state = new_state
        self.state_since = now
        self.warned = False

    def elapsed_in_state(self, now):
        if self.state == "PHONE" and self.phone_start:
            return now - self.phone_start
        return now - self.state_since

    def finish(self, now):
        if self.state in self.totals:
            self.totals[self.state] += now - self.state_since
        self.state_since = now
        tracked = sum(self.totals.values())
        focus_ratio = self.totals["FOCUS"] / tracked * 100 if tracked else 0.0
        return {
            "session_sec": now - self.session_start,
            "focus_sec": self.totals["FOCUS"],
            "phone_sec": self.totals["PHONE"],
            "away_sec": self.totals["AWAY"],
            "focus_ratio": focus_ratio,
            "phone_warnings": self.warn_count["PHONE"],
            "away_warnings": self.warn_count["AWAY"],
        }


# ──────────────────────────────────────────────
# 화면 표시 / 부저 / 저장
# ──────────────────────────────────────────────
def draw_status(frame, agent, alert, now, fps):
    color = STATE_COLORS[agent.state]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, f"STATE: {agent.state}  ({agent.elapsed_in_state(now):.0f}s)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    focus_now = agent.totals["FOCUS"] + (now - agent.state_since if agent.state == "FOCUS" else 0)
    cv2.putText(frame, f"Focus {focus_now / 60:.1f}min | Phone warn {agent.warn_count['PHONE']}"
                       f" | Away warn {agent.warn_count['AWAY']} | {fps:.1f} FPS",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if alert == "PHONE":
        msg = "WARNING: PUT DOWN YOUR PHONE!"
    elif alert == "AWAY":
        msg = "WARNING: YOU LEFT YOUR SEAT!"
    else:
        return
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 50), (frame.shape[1], h), (0, 0, 255), -1)
    cv2.putText(frame, msg, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def make_buzzer():
    if BUZZER_PIN is None:
        return None
    try:
        from gpiozero import Buzzer
        return Buzzer(BUZZER_PIN)
    except Exception as e:  # 부저가 없거나 gpiozero 미설치여도 본 기능은 동작
        print(f"[buzzer] 사용 안 함: {e}")
        return None


def save_session(agent, summary):
    os.makedirs(LOG_DIR, exist_ok=True)
    name = datetime.fromtimestamp(agent.session_start).strftime("session_%Y%m%d_%H%M%S.csv")
    path = os.path.join(LOG_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "event", "detail"])
        w.writerows(agent.events)
        w.writerow([])
        w.writerow(["summary", "value"])
        for k, v in summary.items():
            w.writerow([k, f"{v:.1f}" if isinstance(v, float) else v])
    return path


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────
def main():
    model = YOLO(MODEL_PATH)
    class_ids = [i for i, n in model.names.items() if n in TARGET_CLASSES]

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE}))  # 메모리상 BGR → OpenCV/YOLO에 바로 사용
    cam.start()
    time.sleep(1)

    buzzer = make_buzzer()
    agent = FocusAgent()
    prev = time.time()
    fps = 0.0
    print("FocusGuard 시작 — 화면에서 q 키로 종료")

    try:
        while True:
            frame = cam.capture_array()
            result = model(frame, imgsz=IMG_SIZE, conf=CONF, classes=class_ids, verbose=False)[0]
            labels = {model.names[int(c)] for c in result.boxes.cls}

            now = time.time()
            alert = agent.update(labels, now)

            if buzzer:
                buzzer.on() if alert else buzzer.off()

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            view = result.plot()          # 바운딩 박스 + 라벨 + 신뢰도
            draw_status(view, agent, alert, now, fps)
            cv2.imshow("AI FocusGuard", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        summary = agent.finish(time.time())
        path = save_session(agent, summary)
        if buzzer:
            buzzer.off()
        cam.stop()
        cv2.destroyAllWindows()

        print("\n===== 세션 요약 =====")
        print(f"전체 시간     : {summary['session_sec'] / 60:.1f}분")
        print(f"집중 시간     : {summary['focus_sec'] / 60:.1f}분 ({summary['focus_ratio']:.0f}%)")
        print(f"휴대폰 사용   : {summary['phone_sec'] / 60:.1f}분 (경고 {summary['phone_warnings']}회)")
        print(f"자리 비움     : {summary['away_sec'] / 60:.1f}분 (경고 {summary['away_warnings']}회)")
        print(f"로그 저장     : {path}")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
AI FocusGuard — 라즈베리파이 기반 AI 공부 집중 감시 에이전트

카메라 영상을 YOLOv8n이 Raspberry Pi 안에서 직접 분석하여
사람(person)과 휴대폰(cell phone)을 인식하고,
  - 휴대폰을 일정 시간 이상 보고 있으면  → "PHONE" 경고
  - 자리를 일정 시간 이상 비우면          → "AWAY" 경고
를 띄운다. 종료 시 집중/폰/자리비움 시간 통계를 CSV로 저장한다.

실행:  python focusguard.py      (종료: 화면에서 q 키)
"""

import csv
import os
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ──────────────────────────────────────────────
# 설정값 (시연 상황에 맞게 조절)
# ──────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
FRAME_SIZE = (640, 480)   # 카메라 해상도
IMG_SIZE = 320            # YOLO 입력 크기 (작을수록 빠름)
CONF = 0.40               # 최소 신뢰도

PERSON_GRACE = 3.0        # 사람이 이 시간(초) 이상 안 보여야 '자리 비움'으로 봄
PHONE_GRACE = 1.5         # 폰이 이 시간(초) 이상 안 보여야 '폰 내려놓음'으로 봄
PHONE_WARN_SEC = 5.0      # 폰을 이 시간(초) 이상 보면 경고
AWAY_WARN_SEC = 15.0      # 자리를 이 시간(초) 이상 비우면 경고 (시연용으로 짧게)

BUZZER_PIN = None         # 부저 연결 시 GPIO 번호 입력 (예: 17)
LOG_DIR = "logs"

TARGET_CLASSES = ["person", "cell phone", "book", "laptop"]

STATE_COLORS = {          # OpenCV는 BGR 순서
    "WAITING": (200, 200, 200),
    "FOCUS": (0, 200, 0),
    "PHONE": (0, 140, 255),
    "AWAY": (0, 0, 255),
}


# ──────────────────────────────────────────────
# 에이전트: 인식 결과 → 상태 기억 → 판단 → 행동
# ──────────────────────────────────────────────
class FocusAgent:
    def __init__(self):
        now = time.time()
        self.session_start = now
        self.state = "WAITING"
        self.state_since = now

        self.last_person = None   # 사람을 마지막으로 본 시각
        self.last_phone = None    # 폰을 마지막으로 본 시각
        self.phone_start = None   # 이번 폰 사용이 시작된 시각

        self.totals = {"FOCUS": 0.0, "PHONE": 0.0, "AWAY": 0.0}
        self.warn_count = {"PHONE": 0, "AWAY": 0}
        self.warned = False       # 현재 상태에서 이미 경고를 기록했는지
        self.events = []          # (시각, 이벤트, 상세)

    def log(self, now, event, detail=""):
        stamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        self.events.append((stamp, event, detail))
        print(f"[{stamp}] {event} {detail}")

    def update(self, labels, now):
        """한 프레임의 인식 결과(labels)를 받아 상태를 갱신하고, 경고 종류를 반환."""
        # 1) 기억 갱신
        if "person" in labels:
            self.last_person = now
        if "cell phone" in labels:
            if self.last_phone is None or now - self.last_phone > PHONE_GRACE:
                self.phone_start = now          # 새 폰 사용 시작
            self.last_phone = now

        person_here = self.last_person is not None and now - self.last_person < PERSON_GRACE
        phone_here = self.last_phone is not None and now - self.last_phone < PHONE_GRACE

        # 2) 상태 판단
        if self.last_person is None:
            new_state = "WAITING"
        elif not person_here:
            new_state = "AWAY"
        elif phone_here:
            new_state = "PHONE"
        else:
            new_state = "FOCUS"

        self._transition(new_state, now)

        # 3) 시간 조건에 따른 경고 판단
        alert = None
        if self.state == "PHONE" and now - self.phone_start >= PHONE_WARN_SEC:
            alert = "PHONE"
        elif self.state == "AWAY" and now - self.state_since >= AWAY_WARN_SEC:
            alert = "AWAY"

        if alert and not self.warned:
            self.warned = True
            self.warn_count[alert] += 1
            self.log(now, f"WARNING_{alert}", f"#{self.warn_count[alert]}")
        return alert

    def _transition(self, new_state, now):
        if new_state == self.state:
            return
        if self.state in self.totals:
            self.totals[self.state] += now - self.state_since
        self.log(now, "STATE", f"{self.state} -> {new_state}")
        self.state = new_state
        self.state_since = now
        self.warned = False

    def elapsed_in_state(self, now):
        if self.state == "PHONE" and self.phone_start:
            return now - self.phone_start
        return now - self.state_since

    def finish(self, now):
        if self.state in self.totals:
            self.totals[self.state] += now - self.state_since
        self.state_since = now
        tracked = sum(self.totals.values())
        focus_ratio = self.totals["FOCUS"] / tracked * 100 if tracked else 0.0
        return {
            "session_sec": now - self.session_start,
            "focus_sec": self.totals["FOCUS"],
            "phone_sec": self.totals["PHONE"],
            "away_sec": self.totals["AWAY"],
            "focus_ratio": focus_ratio,
            "phone_warnings": self.warn_count["PHONE"],
            "away_warnings": self.warn_count["AWAY"],
        }


# ──────────────────────────────────────────────
# 화면 표시 / 부저 / 저장
# ──────────────────────────────────────────────
def draw_status(frame, agent, alert, now, fps):
    color = STATE_COLORS[agent.state]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(frame, f"STATE: {agent.state}  ({agent.elapsed_in_state(now):.0f}s)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    focus_now = agent.totals["FOCUS"] + (now - agent.state_since if agent.state == "FOCUS" else 0)
    cv2.putText(frame, f"Focus {focus_now / 60:.1f}min | Phone warn {agent.warn_count['PHONE']}"
                       f" | Away warn {agent.warn_count['AWAY']} | {fps:.1f} FPS",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if alert == "PHONE":
        msg = "WARNING: PUT DOWN YOUR PHONE!"
    elif alert == "AWAY":
        msg = "WARNING: YOU LEFT YOUR SEAT!"
    else:
        return
    h = frame.shape[0]
    cv2.rectangle(frame, (0, h - 50), (frame.shape[1], h), (0, 0, 255), -1)
    cv2.putText(frame, msg, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def make_buzzer():
    if BUZZER_PIN is None:
        return None
    try:
        from gpiozero import Buzzer
        return Buzzer(BUZZER_PIN)
    except Exception as e:  # 부저가 없거나 gpiozero 미설치여도 본 기능은 동작
        print(f"[buzzer] 사용 안 함: {e}")
        return None


def save_session(agent, summary):
    os.makedirs(LOG_DIR, exist_ok=True)
    name = datetime.fromtimestamp(agent.session_start).strftime("session_%Y%m%d_%H%M%S.csv")
    path = os.path.join(LOG_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time", "event", "detail"])
        w.writerows(agent.events)
        w.writerow([])
        w.writerow(["summary", "value"])
        for k, v in summary.items():
            w.writerow([k, f"{v:.1f}" if isinstance(v, float) else v])
    return path


# ──────────────────────────────────────────────
# 메인 루프
# ──────────────────────────────────────────────
def main():
    model = YOLO(MODEL_PATH)
    class_ids = [i for i, n in model.names.items() if n in TARGET_CLASSES]

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE}))  # 메모리상 BGR → OpenCV/YOLO에 바로 사용
    cam.start()
    time.sleep(1)

    buzzer = make_buzzer()
    agent = FocusAgent()
    prev = time.time()
    fps = 0.0
    print("FocusGuard 시작 — 화면에서 q 키로 종료")

    try:
        while True:
            frame = cam.capture_array()
            result = model(frame, imgsz=IMG_SIZE, conf=CONF, classes=class_ids, verbose=False)[0]
            labels = {model.names[int(c)] for c in result.boxes.cls}

            now = time.time()
            alert = agent.update(labels, now)

            if buzzer:
                buzzer.on() if alert else buzzer.off()

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            view = result.plot()          # 바운딩 박스 + 라벨 + 신뢰도
            draw_status(view, agent, alert, now, fps)
            cv2.imshow("AI FocusGuard", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        summary = agent.finish(time.time())
        path = save_session(agent, summary)
        if buzzer:
            buzzer.off()
        cam.stop()
        cv2.destroyAllWindows()

        print("\n===== 세션 요약 =====")
        print(f"전체 시간     : {summary['session_sec'] / 60:.1f}분")
        print(f"집중 시간     : {summary['focus_sec'] / 60:.1f}분 ({summary['focus_ratio']:.0f}%)")
        print(f"휴대폰 사용   : {summary['phone_sec'] / 60:.1f}분 (경고 {summary['phone_warnings']}회)")
        print(f"자리 비움     : {summary['away_sec'] / 60:.1f}분 (경고 {summary['away_warnings']}회)")
        print(f"로그 저장     : {path}")


if __name__ == "__main__":
    main()
