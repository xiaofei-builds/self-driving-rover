#!/usr/bin/env python3
"""autopilot.py -- autonomous driving with the trained PilotNet model (Session 7, DEPLOY)."""
import argparse
import time
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from picarx import Picarx
from vilib import Vilib

MODEL_PATH  = "/home/pi/pilot.tflite"
IMG_W, IMG_H = 160, 120     # cv2.resize takes (width, height)
THROTTLE     = 10           # matches data collection speed
CAM_TILT     = -10          # matches data collection camera angle
STEER_LIMIT  = 30           # model output -1..1 -> +/-30 degrees


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true",
                    help="actually spin the drive motors (default: steering only)")
    ap.add_argument("--seconds", type=float, default=0,
                    help="auto-stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--gain", type=float, default=1.0, help="steering gain")
    args = ap.parse_args()

    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print("model input :", in_det["shape"], in_det["dtype"])
    print("model output:", out_det["shape"], out_det["dtype"])

    px = Picarx()
    px.set_cam_pan_angle(0)
    px.set_cam_tilt_angle(CAM_TILT)
    px.set_dir_servo_angle(0)

    Vilib.camera_start()
    Vilib.display(local=False, web=True)   # watch at http://192.168.86.35:9000/mjpg
    time.sleep(2)                          # camera warm-up

    mode = "LIVE DRIVE" if args.drive else "BENCH TEST (motors OFF)"
    print(f"\n=== {mode} ===  Ctrl-C to stop\n")
    time.sleep(3)

    frames = 0
    t0 = time.time()
    try:
        if args.drive:
            px.forward(THROTTLE)
        while True:
            img = Vilib.img
            if img is None:
                continue
            small = cv2.resize(img, (IMG_W, IMG_H))    # (120,160,3), BGR
            rgb   = small[:, :, ::-1]                   # BGR -> RGB
            x     = (rgb.astype(np.float32) / 255.0)    # 0..1
            x     = np.expand_dims(x, 0)                # [1,120,160,3]
            interp.set_tensor(in_det["index"], x)
            interp.invoke()
            y = float(interp.get_tensor(out_det["index"])[0][0])  # -1..1
            angle = max(-STEER_LIMIT, min(STEER_LIMIT, y * STEER_LIMIT * args.gain))
            px.set_dir_servo_angle(angle)

            frames += 1
            if frames % 15 == 0:
                fps = frames / (time.time() - t0)
                bar = "L" * int(max(0, -angle) / 3) + "|" + "R" * int(max(0, angle) / 3)
                print(f"steer={angle:+6.1f}  {fps:4.1f} fps   {bar}")

            if args.seconds and (time.time() - t0) >= args.seconds:
                print("\nreached time limit")
                break
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        px.forward(0)
        px.stop()
        px.set_dir_servo_angle(0)
        try:
            Vilib.camera_close()
        except Exception:
            pass
        print("motors off, steering centered, camera closed")


if __name__ == "__main__":
    main()
