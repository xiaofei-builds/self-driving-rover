#!/usr/bin/env python3
"""inspect_dataset.py - sanity-check the behavioral-cloning dataset."""
import os, csv, math, collections
import cv2

DATASET_DIR = os.path.expanduser("~/dataset")
IMG_DIR     = os.path.join(DATASET_DIR, "data")
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")

rows = []
with open(LABELS_CSV) as f:
    for row in csv.DictReader(f):
        rows.append(row)

imgs    = {f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}
labeled = {row["image"] for row in rows}
missing = labeled - imgs
orphan  = imgs - labeled

steer  = collections.Counter(int(row["steering"]) for row in rows)
left   = sum(v for k, v in steer.items() if k < 0)
center = steer.get(0, 0)
right  = sum(v for k, v in steer.items() if k > 0)

print(f"CSV rows:        {len(rows)}")
print(f"Image files:     {len(imgs)}")
print(f"Labels w/o image:{len(missing)}")
print(f"Images w/o label:{len(orphan)}")
print()
print(f"Steering balance -> LEFT {left} | CENTER {center} | RIGHT {right}")
print("Histogram (angle: count):")
peak = max(steer.values()) if steer else 1
for k in sorted(steer):
    print(f"  {k:+3d}: {steer[k]:5d} {'#' * (steer[k] * 40 // peak)}")

step   = max(1, len(rows) // 25)
sample = rows[::step][:25]
tiles  = []
for row in sample:
    im = cv2.imread(os.path.join(IMG_DIR, row["image"]))
    if im is None:
        continue
    im = cv2.resize(im, (160, 120))
    cv2.putText(im, row["steering"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    tiles.append(im)
if tiles:
    cols  = 5
    nrows = math.ceil(len(tiles) / cols)
    while len(tiles) < cols * nrows:
        tiles.append(tiles[0] * 0)
    grid = cv2.vconcat([cv2.hconcat(tiles[i*cols:(i+1)*cols]) for i in range(nrows)])
    out  = os.path.join(DATASET_DIR, "sample_grid.jpg")
    cv2.imwrite(out, grid)
    print(f"\nContact sheet saved: {out}")
