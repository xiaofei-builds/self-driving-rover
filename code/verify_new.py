import os, csv, collections
from datetime import datetime
import numpy as np, cv2

DIR = os.path.expanduser("~/dataset")
IMG_DIR = os.path.join(DIR, "data")
rows = list(csv.DictReader(open(os.path.join(DIR, "labels.csv"))))
times = [datetime.fromisoformat(r['timestamp']) for r in rows]
gaps = [(times[i+1]-times[i]).total_seconds() for i in range(len(times)-1)]
k = int(np.argmax(gaps))                       # boundary: today's block starts at k+1
new = rows[k+1:]
print(f"largest gap = {gaps[k]/3600:.1f} h  ->  NEW block = {len(new)} frames of {len(rows)} total")

st = [float(r['steering']) for r in new]
bins = collections.Counter(int(round(s/5.0))*5 for s in st)
print("\nsteering histogram (new frames):")
for b in range(-30, 31, 5):
    print(f"{b:+4d} | {'#'*bins.get(b,0)} ({bins.get(b,0)})")
L = sum(s < -2 for s in st); C = sum(-2 <= s <= 2 for s in st); R = sum(s > 2 for s in st)
print(f"\nL={L}  C={C}  R={R}")

def imgpath(v):
    for p in (os.path.join(DIR, v), os.path.join(IMG_DIR, os.path.basename(v))):
        if os.path.exists(p): return p
    return None

n = min(48, len(new)); tiles = []
for i in np.linspace(0, len(new)-1, n).astype(int):
    p = imgpath(new[i]['image'])
    if not p: continue
    im = cv2.resize(cv2.imread(p), (160, 120))
    cv2.putText(im, f"{float(new[i]['steering']):+.0f}", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    tiles.append(im)
cols = 6
while len(tiles) % cols: tiles.append(np.zeros((120,160,3), np.uint8))
grid = np.vstack([np.hstack(tiles[r:r+cols]) for r in range(0, len(tiles), cols)])
out = os.path.join(DIR, "new_review.jpg"); cv2.imwrite(out, grid)
print(f"\ncontact sheet -> {out}")
