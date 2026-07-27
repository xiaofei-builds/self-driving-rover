#!/usr/bin/env python3
"""collect_data.py - Phase 3 data collection (behavioral cloning)."""
import os, sys, csv, time, threading, termios, tty
from datetime import datetime
import cv2
from picarx import Picarx
from vilib import Vilib

DATASET_DIR   = os.path.expanduser("~/dataset")
IMG_DIR       = os.path.join(DATASET_DIR, "data")
LABELS_CSV    = os.path.join(DATASET_DIR, "labels.csv")
IMG_W, IMG_H  = 160, 120
FORWARD_SPEED = 10
STEER_STEP    = 5
STEER_MAX     = 30
RECORD_HZ     = 15
CAM_PAN       = 0
CAM_TILT      = -10

state = {"angle": 0, "throttle": 0, "recording": False, "quit": False}
lock  = threading.Lock()

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

def keyboard_loop():
    while not state["quit"]:
        ch = getch().lower()
        with lock:
            if   ch == 'w': state["throttle"] = FORWARD_SPEED
            elif ch == 's': state["throttle"] = 0
            elif ch == 'a': state["angle"] = max(-STEER_MAX, state["angle"] - STEER_STEP)
            elif ch == 'd': state["angle"] = min( STEER_MAX, state["angle"] + STEER_STEP)
            elif ch == 'c': state["angle"] = 0
            elif ch == 'r':
                state["recording"] = not state["recording"]
                print(f"\r[REC {'ON ' if state['recording'] else 'OFF'}]                 ")
            elif ch == 'q': state["quit"] = True

def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    new_file = not os.path.exists(LABELS_CSV)
    csv_f = open(LABELS_CSV, "a", newline="")
    writer = csv.writer(csv_f)
    if new_file:
        writer.writerow(["index", "timestamp", "image", "steering", "throttle"])
    existing = [f for f in os.listdir(IMG_DIR) if f.startswith("img_") and f.endswith(".jpg")]
    idx = (max(int(f[4:10]) for f in existing) + 1) if existing else 0

    px = Picarx()
    px.set_dir_servo_angle(0)
    px.set_cam_pan_angle(CAM_PAN)
    px.set_cam_tilt_angle(CAM_TILT)
    Vilib.camera_start(vflip=False, hflip=False)
    Vilib.display(local=False, web=True)
    time.sleep(2)

    threading.Thread(target=keyboard_loop, daemon=True).start()
    print("Ready. Open http://<pi-ip>:9000/mjpg to see the camera.")
    print("Controls: w/s go/stop | a/d/c steer | r record | q quit")

    period = 1.0 / RECORD_HZ
    saved = 0
    try:
        while not state["quit"]:
            t0 = time.time()
            with lock:
                angle, throttle, rec = state["angle"], state["throttle"], state["recording"]
            px.set_dir_servo_angle(angle)
            px.forward(throttle) if throttle > 0 else px.stop()
            if rec and throttle > 0:
                frame = Vilib.img
                if frame is not None:
                    small = cv2.resize(frame, (IMG_W, IMG_H))
                    name = f"img_{idx:06d}.jpg"
                    cv2.imwrite(os.path.join(IMG_DIR, name), small)
                    writer.writerow([idx, datetime.now().isoformat(), name, angle, throttle])
                    idx += 1; saved += 1
                    if saved % 15 == 0:
                        csv_f.flush()
                        print(f"\rsaved {saved} frames | steering {angle:+d}   ", end="")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        state["quit"] = True
        px.forward(0); px.stop(); px.set_dir_servo_angle(0)
        csv_f.flush(); csv_f.close()
        try: Vilib.camera_close()
        except Exception: pass
        print(f"\nStopped. Saved this run: {saved}. Dataset: {DATASET_DIR}")

if __name__ == "__main__":
    main()
