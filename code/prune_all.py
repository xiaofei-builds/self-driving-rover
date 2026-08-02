#!/usr/bin/env python3
"""Prune tape-less frames across the WHOLE dataset. Tunable HSV gate + dense review sheets."""
import os, sys, glob, csv, shutil, datetime, collections
import cv2, numpy as np
DS=os.path.expanduser("~/dataset"); IMG=os.path.join(DS,"data")
CSV=os.path.join(DS,"labels.csv"); REJ=os.path.join(DS,"rejected")
K=np.ones((3,3),np.uint8); TODAY=datetime.date.today().isoformat()
def arg(name,dflt):
    a=sys.argv; return int(a[a.index(name)+1]) if name in a else dflt
HMIN=arg("--hmin",95); HMAX=arg("--hmax",130); SMIN=arg("--smin",90); VMIN=arg("--vmin",80)
LO=np.array([HMIN,SMIN,VMIN]); HI=np.array([HMAX,255,255])
def blob(p):
    im=cv2.imread(p)
    if im is None: return 0
    m=cv2.morphologyEx(cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV),LO,HI),cv2.MORPH_OPEN,K)
    n,_,st,_=cv2.connectedComponentsWithStats(m,8)
    return int(st[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0
def pct(vals,tag):
    if not vals: print(f"  [{tag}] none"); return
    vs=sorted(vals); print(f"  [{tag}] n={len(vs)}  "+" ".join(f"{p}%={int(np.percentile(vs,p))}" for p in (0,10,25,50,75,90,100)))
def sheet(files,name,maxn=96):
    s=files[::max(1,len(files)//maxn)][:maxn] if files else []
    im=[cv2.resize(cv2.imread(f),(160,120)) for f in s if cv2.imread(f) is not None]
    while im and len(im)%6: im.append(np.zeros((120,160,3),np.uint8))
    if im:
        g=np.vstack([np.hstack(im[i:i+6]) for i in range(0,len(im),6)]); cv2.imwrite(os.path.join(DS,name),g); print("wrote",name,f"({len(s)} frames)")
def main():
    thr=arg("--thresh",30); apply="--apply" in sys.argv; include_day="--include-day" in sys.argv
    print(f"mask H[{HMIN}-{HMAX}] S>={SMIN} V>={VMIN} | thresh={thr} include_day={include_day}")
    meta={r["image"]:(r["timestamp"][:10],int(r["steering"])) for r in csv.DictReader(open(CSV))}
    fs=sorted(glob.glob(os.path.join(IMG,"img_*.jpg"))); print(f"scanning {len(fs)} frames ...")
    night_v,day_v,rec=[],[],[]
    for i,f in enumerate(fs):
        b=os.path.basename(f); date,steer=meta.get(b,("?",0)); v=blob(f); isday=(date==TODAY)
        (day_v if isday else night_v).append(v); rec.append((f,b,v,isday,steer))
        if i and i%2000==0: print(f"  ...{i}")
    print("largest-blue-blob percentiles (px):"); pct(night_v,"NIGHT"); pct(day_v,"DAY")
    fa=lambda isday: include_day or (not isday)
    flagged=[x for x in rec if x[2]<thr and fa(x[3])]
    fn=sum(1 for x in flagged if not x[3]); fd=sum(1 for x in flagged if x[3])
    print(f"\n-> FLAGGED {len(flagged)} (night {fn}, day {fd})")
    print("  flagged |steering| buckets:",dict(sorted(collections.Counter(abs(x[4])//10*10 for x in flagged).items())))
    sheet([x[0] for x in flagged],"prune_all_flagged.jpg",96)
    sheet([x[0] for x in flagged if abs(x[4])>=20],"prune_all_flagged_hard.jpg",96)
    if not apply:
        print("DRY RUN. Review both sheets (esp _hard). Then re-run with --apply."); return
    os.makedirs(REJ,exist_ok=True); shutil.copy(CSV,CSV+".bak")
    fset={x[1] for x in flagged}; rows=list(csv.reader(open(CSV))); head,body=rows[0],rows[1:]
    keep=[r for r in body if r[2] not in fset]
    for f,b,_,_,_ in flagged:
        if os.path.exists(f): shutil.move(f,os.path.join(REJ,b))
    with open(CSV,"w",newline="") as o:
        w=csv.writer(o); w.writerow(head); w.writerows(keep)
    print(f"MOVED {len(flagged)} to {REJ}. labels.csv now {len(keep)} rows (+.bak).")
if __name__=="__main__": main()
