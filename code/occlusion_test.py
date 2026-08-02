#!/usr/bin/env python3
"""occlusion_test.py - does the model steer on the TAPE or the BACKGROUND?"""
import os, csv, random
import numpy as np, cv2
from ai_edge_litert.interpreter import Interpreter

DATA  = os.path.expanduser("~/dataset")
MODEL = os.path.expanduser("~/pilot.tflite")

it = Interpreter(model_path=MODEL); it.allocate_tensors()
inp = it.get_input_details()[0]; out = it.get_output_details()[0]
def predict(rgb01):
    x = rgb01.astype(np.float32)[None, ...]
    it.set_tensor(inp['index'], x); it.invoke()
    return float(it.get_tensor(out['index'])[0][0]) * 30.0

def tape_mask(bgr):                     # night-tuned blue HSV gate
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (90, 90, 80), (130, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

rows = list(csv.DictReader(open(os.path.join(DATA, "labels.csv"))))
buckets = {"L": [], "C": [], "R": []}
for r in rows:
    s = int(r["steering"]); buckets["L" if s<=-10 else "R" if s>=10 else "C"].append(r)
random.seed(0)
picks = sum((random.sample(buckets[k], min(3, len(buckets[k]))) for k in "LCR"), [])

print(f"{'file':16}{'true':>6}{'orig':>8}{'tapeOnly':>10}{'bgOnly':>8}{'tape%':>7}")
for r in picks:
    bgr = cv2.imread(os.path.join(DATA, "data", r["image"]))
    if bgr is None: continue
    if bgr.shape[:2] != (120, 160): bgr = cv2.resize(bgr, (160, 120))
    m = tape_mask(bgr); mean = bgr.reshape(-1, 3).mean(0)
    orig = bgr[:, :, ::-1] / 255.0
    to = bgr.astype(np.float32).copy(); to[m == 0] = mean; to = to[:, :, ::-1] / 255.0
    bo = bgr.astype(np.float32).copy(); bo[m > 0] = mean; bo = bo[:, :, ::-1] / 255.0
    print(f"{r['image']:16}{int(r['steering']):6d}{predict(orig):8.1f}"
          f"{predict(to):10.1f}{predict(bo):8.1f}{100*m.mean()/255:6.1f}%")
print("\ntapeOnly~orig & bgOnly flat  -> model uses the TAPE (appearance shift)")
print("bgOnly~orig & tapeOnly flat  -> model uses the BACKGROUND (your shortcut hypothesis)")
