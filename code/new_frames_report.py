#!/usr/bin/env python3
"""Report on frames collected TODAY: count, steering balance, contact sheet."""
import os, csv, datetime
import numpy as np, cv2

DATA = os.path.expanduser("~/dataset")
today = datetime.date.today().isoformat()
rows = [r for r in csv.DictReader(open(os.path.join(DATA, "labels.csv")))
        if r["timestamp"].startswith(today)]
print(f"New frames today ({today}): {len(rows)}")
if not rows:
    raise SystemExit("No frames stamped today — check the date / that recording was ON.")

st = [int(r["steering"]) for r in rows]
L = sum(s <= -10 for s in st); C = sum(-10 < s < 10 for s in st); R = sum(s >= 10 for s in st)
print(f"steering  L(<=-10): {L}   C: {C}   R(>=10): {R}   range [{min(st)}, {max(st)}]")
import collections
h = collections.Counter((s // 5) * 5 for s in st)
for b in sorted(h): print(f"  {b:+3d}..{b+4:+3d}: {'#'*h[b]} ({h[b]})")

# contact sheet of up to 24 evenly-sampled new frames
os.makedirs(os.path.join(DATA, "review"), exist_ok=True)
pick = rows[:: max(1, len(rows)//24)][:24]
tiles = []
for r in pick:
    img = cv2.imread(os.path.join(DATA, "data", r["image"]))
    if img is None: continue
    img = cv2.resize(img, (160, 120))
    cv2.putText(img, f"{int(r['steering']):+d}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    tiles.append(img)
cols = 6
rows_n = (len(tiles) + cols - 1) // cols
while len(tiles) < rows_n * cols: tiles.append(np.zeros((120, 160, 3), np.uint8))
grid = np.vstack([np.hstack(tiles[i*cols:(i+1)*cols]) for i in range(rows_n)])
out = os.path.join(DATA, "review", "new_daylight.jpg")
cv2.imwrite(out, grid)
print(f"\nContact sheet: {out}")
