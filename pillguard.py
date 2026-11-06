#!/usr/bin/env python3
"""
AI PillGuard — 라즈베리파이 기반 AI 약 복용 체크 에이전트

카메라 영상을 YOLOv8n이 Raspberry Pi 안에서 직접 분석하여
사람(person)과 약통(bottle)을 인식하고,
  - 약통이 들렸다가(2초 이상) 제자리로 돌아오면   → 복용 기록
  - 복용 시간이 지났는데 기록이 없으면            → 복용 알림 (REMINDER)
  - 이미 복용했는데 또 약통을 들면                → 중복 복용 경고 (DOUBLE)
복용 기록은 logs/pill_날짜.csv 에 바로바로 저장된다.

실행:  python pillguard.py
키:    q = 종료,  r = 약통 위치 다시 등록
"""

import csv
import os
import time
from datetime import datetime, timedelta

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ──────────────────────────────────────────────
# 설정값 (시연 상황에 맞게 조절)
# ──────────────────────────────────────────────
MODEL_PATH = "yolov8n.pt"
FRAME_SIZE = (640, 480)
IMG_SIZE = 320
CONF = 0.30               # 약통(bottle)은 작아서 신뢰도를 조금 낮게

# 복용 시간표 (실제 사용)
SCHEDULE = [("morning", "09:00"), ("evening", "21:00")]
# 시연 모드: 숫자를 넣으면 시간표 대신 "실행 후 N초"를 복용 시간으로 사용
DEMO_DOSE_AFTER_SEC = 60  # 실제 시간표를 쓰려면 None

HOME_REGISTER_SEC = 3.0   # 약통이 이 시간 동안 같은 자리에 있으면 '보관 자리'로 등록
HOME_MOVE_PX = 40         # 등록 중 이 픽셀 이상 움직이면 다시 측정
HOME_IOU = 0.30           # 보관 자리 박스와 이 이상 겹치면 '제자리'
MISS_GRACE = 1.0          # 약통이 이 시간(초) 이상 제자리에 없으면 '들림'
RETURN_CONFIRM = 1.0      # 약통이 이 시간(초) 이상 제자리에 있어야 '돌아옴'
MIN_PICK_SEC = 2.0        # 이 시간 이상 들고 있어야 복용으로 인정
PERSON_GRACE = 3.0        # 사람을 이 시간(초) 안에 봤으면 '사람 있음'
TAKEN_SHOW_SEC = 3.0      # TAKEN 표시 유지 시간

BUZZER_PIN = None         # 부저 연결 시 GPIO 번호 (예: 17)
LOG_DIR = "logs"

CONTAINER_CLASSES = ["bottle", "cup"]   # 약통으로 인정할 물체 (컵/병 모두 가능)
TARGET_CLASSES = ["person"] + CONTAINER_CLASSES

STATE_COLORS = {          # BGR
    "SETUP": (200, 200, 200),
    "IDLE": (0, 200, 0),
    "PICKED": (0, 140, 255),
    "TAKEN": (255, 120, 0),
    "MISSING": (0, 200, 255),
}


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


# ──────────────────────────────────────────────
# 에이전트: 인식 → 기억 → 판단 → 행동
# ──────────────────────────────────────────────
class PillAgent:
    def __init__(self, now=None):
        now = now or time.time()
        self.start = now
        self.state = "SETUP"

        # 약통 보관 자리 등록용
        self.home = None
        self.cand = None
        self.cand_since = None
        self.cand_last = None

        # 기억
        self.last_person = None
        self.last_at_home = None
        self.home_streak_start = None
        self.pick_start = None
        self.taken_until = 0.0
        self.double_alert = False

        # 오늘 복용 기록
        self.today = None
        self.taken = {}           # dose -> "HH:MM:SS"
        self.reminded = set()
        self._new_day(now)

    # ---------- 기록 ----------
    def log_path(self, now):
        return os.path.join(LOG_DIR, datetime.fromtimestamp(now).strftime("pill_%Y%m%d.csv"))

    def log(self, now, event, detail=""):
        stamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        print(f"[{stamp}] {event} {detail}")
        os.makedirs(LOG_DIR, exist_ok=True)
        path = self.log_path(now)
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["time", "event", "detail"])
            w.writerow([stamp, event, detail])

    def _new_day(self, now):
        """날짜가 바뀌면 기록 초기화. 프로그램을 재시작해도 오늘 복용 기록은 CSV에서 불러온다."""
        self.today = datetime.fromtimestamp(now).date()
        self.taken, self.reminded = {}, set()
        path = self.log_path(now)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("event") == "TAKEN":
                        self.taken[row["detail"]] = row["time"]

    # ---------- 시간표 ----------
    def doses(self, now):
        if DEMO_DOSE_AFTER_SEC is not None:
            return [("demo", datetime.fromtimestamp(self.start) + timedelta(seconds=DEMO_DOSE_AFTER_SEC))]
        base = datetime.fromtimestamp(now)
        out = []
        for name, hm in SCHEDULE:
            h, m = map(int, hm.split(":"))
            out.append((name, base.replace(hour=h, minute=m, second=0, microsecond=0)))
        return sorted(out, key=lambda d: d[1])

    def current_dose(self, now):
        """지금 해당되는 회차: 이미 시간이 지난 마지막 회차, 없으면 첫 회차(미리 복용 허용)."""
        doses = self.doses(now)
        nowdt = datetime.fromtimestamp(now)
        passed = [d for d in doses if d[1] <= nowdt]
        return (passed[-1] if passed else doses[0]), bool(passed)

    # ---------- 메인 판단 ----------
    def update(self, dets, now):
        """dets: [(label, conf, (x1,y1,x2,y2)), ...]  →  경고 종류(None/'REMINDER'/'DOUBLE') 반환"""
        if datetime.fromtimestamp(now).date() != self.today:
            self._new_day(now)

        if any(d[0] == "person" for d in dets):
            self.last_person = now
        bottles = [d for d in dets if d[0] in CONTAINER_CLASSES]

        # 1) 약통 보관 자리 등록
        if self.home is None:
            self._register(bottles, now)
            return None

        # 2) 약통이 제자리에 있는지 기억 갱신
        at_home = any(iou(b[2], self.home) > HOME_IOU for b in bottles)
        if at_home:
            if self.last_at_home is None or now - self.last_at_home > 0.5:
                self.home_streak_start = now
            self.last_at_home = now

        missing = now - self.last_at_home >= MISS_GRACE
        returned = at_home and now - self.home_streak_start >= RETURN_CONFIRM
        person_here = self.last_person is not None and now - self.last_person < PERSON_GRACE
        (dose, dose_time), dose_passed = self.current_dose(now)

        # 3) 상태 판단
        if self.state == "TAKEN" and now > self.taken_until:
            self.state = "IDLE"

        if self.state in ("IDLE", "TAKEN"):
            if missing:
                if person_here:
                    self._set(now, "PICKED")
                    self.pick_start = self.last_at_home
                    if dose in self.taken:
                        self.double_alert = True
                        self.log(now, "WARNING_DOUBLE", f"{dose} already taken at {self.taken[dose]}")
                else:
                    self._set(now, "MISSING")   # 가려졌거나 오인식 → 복용으로 보지 않음

        elif self.state == "PICKED":
            if returned:
                away = self.home_streak_start - self.pick_start
                if away >= MIN_PICK_SEC and not self.double_alert:
                    self.taken[dose] = datetime.fromtimestamp(now).strftime("%H:%M:%S")
                    self.log(now, "TAKEN", dose)
                    self._set(now, "TAKEN")
                    self.taken_until = now + TAKEN_SHOW_SEC
                else:
                    self._set(now, "IDLE")
                self.double_alert = False

        elif self.state == "MISSING":
            if returned:
                self._set(now, "IDLE")
            elif person_here:
                self._set(now, "PICKED")
                self.pick_start = now

        # 4) 경고 판단
        if self.state == "PICKED" and self.double_alert:
            return "DOUBLE"
        if dose_passed and dose not in self.taken:
            if dose not in self.reminded:
                self.reminded.add(dose)
                self.log(now, "REMINDER", f"{dose} not taken")
            return "REMINDER"
        return None

    def _register(self, bottles, now):
        if not bottles:
            if self.cand_last and now - self.cand_last > 1.0:
                self.cand = None
            return
        box = max(bottles, key=lambda b: b[1])[2]
        if self.cand is None or dist(center(box), center(self.cand)) > HOME_MOVE_PX:
            self.cand, self.cand_since = box, now
        self.cand_last = now
        if now - self.cand_since >= HOME_REGISTER_SEC:
            self.home = box
            self.last_at_home = self.home_streak_start = now
            self.log(now, "HOME_REGISTERED", str(tuple(int(v) for v in box)))
            self._set(now, "IDLE")

    def reset_home(self, now):
        self.home = self.cand = None
        self._set(now, "SETUP")

    def _set(self, now, new_state):
        if new_state != self.state:
            print(f"[{datetime.fromtimestamp(now):%H:%M:%S}] STATE {self.state} -> {new_state}")
            self.state = new_state


# ──────────────────────────────────────────────
# 화면 표시 / 부저
# ──────────────────────────────────────────────
def draw_status(frame, agent, alert, now, fps):
    w, h = frame.shape[1], frame.shape[0]
    if agent.home is not None:
        x1, y1, x2, y2 = map(int, agent.home)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "HOME", (x1, max(y1 - 6, 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    label = "SETUP: place pill bottle (3s)" if agent.state == "SETUP" else agent.state
    cv2.putText(frame, f"STATE: {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                STATE_COLORS[agent.state], 2)

    (dose, dose_time), _ = agent.current_dose(now)
    status = f"TAKEN {agent.taken[dose]}" if dose in agent.taken else f"due {dose_time:%H:%M:%S}"
    cv2.putText(frame, f"Dose [{dose}] {status} | {fps:.1f} FPS",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if alert == "DOUBLE":
        msg, color = "WARNING: ALREADY TAKEN TODAY!", (0, 0, 255)
    elif alert == "REMINDER":
        msg, color = "REMINDER: TIME TO TAKE YOUR MEDICINE!", (0, 140, 255)
    elif agent.state == "TAKEN":
        msg, color = f"MEDICINE TAKEN ({dose})", (255, 120, 0)
    else:
        return
    cv2.rectangle(frame, (0, h - 50), (w, h), color, -1)
    cv2.putText(frame, msg, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


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
    class_ids = [i for i, n in model.names.items() if n in TARGET_CLASSES]

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE}))
    cam.start()
    time.sleep(1)

    buzzer = make_buzzer()
    agent = PillAgent()
    prev, fps = time.time(), 0.0
    print("PillGuard 시작 — 약통을 3초간 제자리에 두면 등록됩니다. (q 종료, r 재등록)")

    try:
        while True:
            frame = cam.capture_array()
            result = model(frame, imgsz=IMG_SIZE, conf=CONF, classes=class_ids, verbose=False)[0]
            dets = [(model.names[int(c)], float(p), tuple(map(float, b)))
                    for c, p, b in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)]

            now = time.time()
            alert = agent.update(dets, now)

            if buzzer:
                buzzer.on() if alert == "DOUBLE" else buzzer.off()

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            view = result.plot()
            draw_status(view, agent, alert, now, fps)
            cv2.imshow("AI PillGuard", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                agent.reset_home(now)
    finally:
        if buzzer:
            buzzer.off()
        cam.stop()
        cv2.destroyAllWindows()
        print("\n===== 오늘 복용 기록 =====")
        for dose, t in agent.taken.items():
            print(f"{dose:10s}: {t}")
        if not agent.taken:
            print("기록 없음")
        print(f"로그 파일   : {agent.log_path(time.time())}")


if __name__ == "__main__":
    main()
