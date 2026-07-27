#!/usr/bin/env python3
"""clean_dataset.py - flag (dry-run) or remove (--apply) low-quality frames."""
import os, sys, csv, shutil, math
import numpy as np, cv2

DATASET_DIR = os.path.expanduser("~/dataset")
IMG_DIR     = os.path.join(DATASET_DIR, "data")
LABELS_CSV  = os.path.join(DATASET_DIR, "labels.csv")
REJECT_DIR  = os.path.join(DATASET_DIR, "rejected")
APPLY       = "--apply" in sys.argv

DARK_MIN, BRIGHT_MAX, DUP_MIN, BLUR_PCTL, JERK_MAX = 40, 220, 2.0, 5, 20

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
    bright.append(float(g.mean()))
    blur.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    diff.append(float(np.abs(g.astype(int) - prev.astype(int)).mean()) if prev is not None else 255.0)
    prev = g

valid_blur = [b for b in blur if b is not None]
BLUR_MIN = float(np.percentile(valid_blur, BLUR_PCTL)) if valid_blur else 0
steers = [int(r["steering"]) for r in rows]

flagged, reasons_count, jerks = [], {}, []
for i, row in enumerate(rows):
    reasons = []
    if bright[i] is None:
        reasons.append("corrupt")
    else:
        if bright[i] < DARK_MIN:   reasons.append("dark")
        if bright[i] > BRIGHT_MAX: reasons.append("bright")
        if blur[i]  < BLUR_MIN:    reasons.append("blurry")
        if diff[i]  < DUP_MIN:     reasons.append("dup")
    nb = [steers[j] for j in (i-1, i+1) if 0 <= j < len(steers)]
    if nb and abs(steers[i] - sum(nb)/len(nb)) >= JERK_MAX:
        jerks.append(i)
    if reasons:
        flagged.append((i, reasons))
        for r in reasons:
            reasons_count[r] = reasons_count.get(r, 0) + 1

print(f"Total frames:     {len(rows)}")
print(f"Auto-flagged:     {len(flagged)}  ({100*len(flagged)//max(1,len(rows))}%)")
for k, v in sorted(reasons_count.items()):
    print(f"   {k:8s}: {v}")
print(f"Steering jerks (review, NOT auto-removed): {len(jerks)}")
print(f"(blur threshold = {BLUR_MIN:.1f}, the {BLUR_PCTL}th pct of this set)")

sample = flagged[:25]
tiles = []
for i, reasons in sample:
    im = cv2.imread(os.path.join(IMG_DIR, rows[i]["image"]))
    if im is None: im = np.zeros((120, 160, 3), np.uint8)
    im = cv2.resize(im, (160, 120))
    cv2.putText(im, ",".join(reasons), (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    tiles.append(im)
if tiles:
    cols = 5; nrows = math.ceil(len(tiles) / cols)
    while len(tiles) < cols * nrows: tiles.append(np.zeros((120, 160, 3), np.uint8))
    grid = cv2.vconcat([cv2.hconcat(tiles[i*cols:(i+1)*cols]) for i in range(nrows)])
    cv2.imwrite(os.path.join(DATASET_DIR, "flagged_grid.jpg"), grid)
    print(f"Review montage saved: {os.path.join(DATASET_DIR, 'flagged_grid.jpg')}")

if not APPLY:
    print("\nDRY RUN. Nothing removed. Review flagged_grid.jpg, then rerun with --apply to move them out.")
else:
    os.makedirs(REJECT_DIR, exist_ok=True)
    shutil.copy(LABELS_CSV, LABELS_CSV + ".bak")
    flag_idx = {i for i, _ in flagged}
    kept = []
    for i, row in enumerate(rows):
        if i in flag_idx:
            src = os.path.join(IMG_DIR, row["image"])
            if os.path.exists(src): shutil.move(src, os.path.join(REJECT_DIR, row["image"]))
        else:
            kept.append(row)
    with open(LABELS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(kept)
    print(f"\nAPPLIED. Moved {len(flag_idx)} frames to {REJECT_DIR}. Kept {len(kept)}. Backup: labels.csv.bak")
