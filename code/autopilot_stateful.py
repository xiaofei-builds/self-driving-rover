#!/usr/bin/env python3
"""Frame-stacked (memory) autopilot. Feeds the CNN a stack of 4 frames spaced
~0.267s apart in REAL TIME (matches training STRIDE=4 @ 15Hz), selected from a
timestamped ring buffer. Safe by default; --drive spins motors; --seconds N; --gain X."""
import time, argparse, collections, numpy as np, cv2
from ai_edge_litert.interpreter import Interpreter
from picarx import Picarx
from vilib import Vilib

N, DT_STRIDE = 4, 4/15.0            # 0.267s spacing -> 0.80s span
SPAN = (N-1)*DT_STRIDE
H, W = 120, 160

ap = argparse.ArgumentParser()
ap.add_argument("--drive", action="store_true")
ap.add_argument("--seconds", type=float, default=0)
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--model", default="/home/pi/pilot_stateful.tflite")
a = ap.parse_args()

itp = Interpreter(model_path=a.model); itp.allocate_tensors()
inp = itp.get_input_details()[0]; out = itp.get_output_details()[0]
print("model input shape:", tuple(inp["shape"]))
assert tuple(inp["shape"]) == (1,H,W,N*3), "not a 12-channel stacked model!"

px = Picarx(); px.set_cam_tilt_angle(-10)          # extrinsic must match collection
Vilib.camera_start(vflip=False, hflip=False); Vilib.display(local=False, web=True)
time.sleep(2)

buf = collections.deque()
def preprocess(bgr):
    r = cv2.resize(bgr,(W,H)); return (r[:, :, ::-1].astype(np.float32))/255.0
def pick_stack(now):
    picks=[]
    for k in range(N-1,-1,-1):                       # old -> new
        target = now - k*DT_STRIDE
        _,fr = min(buf, key=lambda tf_: abs(tf_[0]-target))
        picks.append(fr)
    return np.concatenate(picks, axis=2)[None].astype(np.float32)

t0=time.time(); n=0; last=t0
try:
    while True:
        now=time.time(); frame=Vilib.img
        if frame is None: continue
        buf.append((now, preprocess(frame)))
        while buf and now-buf[0][0] > SPAN+0.3: buf.popleft()
        if buf[-1][0]-buf[0][0] < SPAN*0.9:          # wait for enough history
            continue
        x=pick_stack(now)
        itp.set_tensor(inp["index"], x); itp.invoke()
        steer=float(itp.get_tensor(out["index"])[0][0])*30.0*a.gain
        steer=max(-30,min(30,steer))
        px.set_dir_servo_angle(steer)
        if a.drive: px.forward(10)
        n+=1
        if now-last>=1.0:
            print(f"fps~{n/(now-last):.0f}  steer={steer:+.1f}  buf={len(buf)}"); n=0; last=now
        if a.seconds and now-t0>a.seconds: break
finally:
    px.forward(0); px.set_dir_servo_angle(0); px.stop(); Vilib.camera_close()
