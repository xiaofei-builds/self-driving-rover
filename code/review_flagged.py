#!/usr/bin/env python3
"""review_flagged.py - paginated contact sheets of ALL flagged frames, by reason."""
import os, csv, math
import numpy as np, cv2

DATASET_DIR = os.path.expanduser("~/dataset")
IMG_DIR     = os.path.join(DATASET_DIR, "data")
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")
REVIEW_DIR  = os.path.join(DATASET_DIR, "review")
os.makedirs(REVIEW_DIR, exist_ok=True)

DARK_MIN, BLUR_PCTL, DUP_MIN = 40, 5, 2.0
COLS, ROWS = 6, 8
PER = COLS * ROWS

rows = []
with open(LABELS_CSV) as f:
    for r in csv.DictReader(f):
        rows.append(r)

bright, blur, diff, prev = [], [], [], None
for row in rows:
    im = cv2.imread(os.path.join(IMG_DIR, row["image"]))
    if im is None:
        bright.append(None); blur.append(None); diff.append(None); prev = None; continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bright.append(float(g.mean())); blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    diff.append(float(np.abs(g.astype(int) - prev.astype(int)).mean()) if prev is not None else 255.0)
    prev = g
BLUR_MIN = float(np.percentile([b for b in blur if b is not None], BLUR_PCTL))

buckets = {"corrupt": [], "dark": [], "blurry": [], "dup": []}
for i in range(len(rows)):
    if bright[i] is None:      buckets["corrupt"].append(i)
    elif bright[i] < DARK_MIN: buckets["dark"].append(i)
    elif blur[i]   < BLUR_MIN: buckets["blurry"].append(i)
    elif diff[i]   < DUP_MIN:  buckets["dup"].append(i)

def sheet(indices, reason):
    pages = math.ceil(len(indices) / PER) if indices else 0
    for p in range(pages):
        tiles = []
        for i in indices[p*PER:(p+1)*PER]:
            im = cv2.imread(os.path.join(IMG_DIR, rows[i]["image"]))
            if im is None: im = np.zeros((120, 160, 3), np.uint8)
            im = cv2.resize(im, (160, 120))
            cv2.putText(im, f"{i} {rows[i]['steering']}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(im)
        while len(tiles) < PER: tiles.append(np.zeros((120, 160, 3), np.uint8))
        grid = cv2.vconcat([cv2.hconcat(tiles[r*COLS:(r+1)*COLS]) for r in range(ROWS)])
        cv2.imwrite(os.path.join(REVIEW_DIR, f"{reason}_p{p:02d}.jpg"), grid)
    return pages

print(f"blur threshold = {BLUR_MIN:.1f} (bottom {BLUR_PCTL}%)")
for reason, idxs in buckets.items():
    if idxs:
        n = sheet(idxs, reason)
        print(f"{reason:8s}: {len(idxs):4d} frames -> {n} page(s)  review/{reason}_p00.jpg ...")
print(f"\nBrowse them at:  http://<pi-ip>:8000/review/")
