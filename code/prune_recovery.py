#!/usr/bin/env python3
"""Prune recovery frames with no visible blue tape, via largest connected blue blob.
Dry-run by default. --thresh N sets min blob area (px). --apply moves flagged to rejected/."""
import os, sys, glob, csv, shutil
import cv2, numpy as np

DS=os.path.expanduser("~/dataset"); IMG=os.path.join(DS,"data")
CSV=os.path.join(DS,"labels.csv"); REJ=os.path.join(DS,"rejected")
LAST_N=351
LO=np.array([95,90,80]); HI=np.array([130,255,255])
K=np.ones((3,3),np.uint8)

def blob(p):
    im=cv2.imread(p)
    if im is None: return 0
    m=cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV),LO,HI)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,K)
    n,_,st,_=cv2.connectedComponentsWithStats(m,8)
    return int(st[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0

def sheet(files,name):
    s=files[::max(1,len(files)//30)][:30] if files else []
    im=[cv2.resize(cv2.imread(f),(160,120)) for f in s]
    while im and len(im)%6: im.append(np.zeros((120,160,3),np.uint8))
    if im:
        g=np.vstack([np.hstack(im[i:i+6]) for i in range(0,len(im),6)])
        cv2.imwrite(os.path.join(DS,name),g); print("wrote",name)

def main():
    a=sys.argv
    thr=int(a[a.index("--thresh")+1]) if "--thresh" in a else 30
    apply="--apply" in a
    fs=sorted(glob.glob(os.path.join(IMG,"img_*.jpg")))[-LAST_N:]
    fr=[(f,blob(f)) for f in fs]; vals=sorted(v for _,v in fr)
    print(f"examined {len(fs)} | largest-blue-blob percentiles (px):")
    for p in [0,10,25,50,75,90,100]:
        print(f"  {p:3d}%: {int(np.percentile(vals,p))}")
    flagged=[f for f,v in fr if v<thr]; kept=[f for f,v in fr if v>=thr]
    print(f"\nthresh={thr}px  ->  FLAGGED (no tape): {len(flagged)}   KEEP: {len(kept)}")
    sheet(flagged,"prune_flagged.jpg"); sheet(kept,"prune_kept.jpg")
    if not apply:
        print("DRY RUN. Check the two sheets. Tune with --thresh N, then add --apply."); return
    os.makedirs(REJ,exist_ok=True); shutil.copy(CSV,CSV+".bak")
    fset={os.path.basename(f) for f in flagged}
    rows=list(csv.reader(open(CSV))); head,body=rows[0],rows[1:]
    keep=[r for r in body if r[2] not in fset]
    for f in flagged: shutil.move(f,os.path.join(REJ,os.path.basename(f)))
    with open(CSV,"w",newline="") as o:
        w=csv.writer(o); w.writerow(head); w.writerows(keep)
    print(f"MOVED {len(flagged)} to {REJ}. labels.csv now {len(keep)} rows (+.bak).")

if __name__=="__main__": main()
