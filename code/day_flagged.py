#!/usr/bin/env python3
import os, glob, csv, datetime
import cv2, numpy as np
DS=os.path.expanduser("~/dataset"); IMG=os.path.join(DS,"data"); CSV=os.path.join(DS,"labels.csv")
LO=np.array([95,90,60]); HI=np.array([130,255,255]); K=np.ones((3,3),np.uint8)   # matches --vmin 60 --smin 90
TODAY=datetime.date.today().isoformat()
day={r["image"]:int(r["steering"]) for r in csv.DictReader(open(CSV)) if r["timestamp"][:10]==TODAY}
def blob(p):
    im=cv2.imread(p)
    if im is None: return 0
    m=cv2.morphologyEx(cv2.inRange(cv2.cvtColor(im,cv2.COLOR_BGR2HSV),LO,HI),cv2.MORPH_OPEN,K)
    n,_,st,_=cv2.connectedComponentsWithStats(m,8)
    return int(st[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0
flag=[]
for f in sorted(glob.glob(os.path.join(IMG,"img_*.jpg"))):
    b=os.path.basename(f)
    if b in day and blob(f)<30: flag.append((f,day[b]))
print("day flagged:",len(flag))
tiles=[]
for f,st in flag:
    im=cv2.resize(cv2.imread(f),(320,240))
    cv2.putText(im,f"{st:+d}",(6,26),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
    tiles.append(im)
while tiles and len(tiles)%3: tiles.append(np.zeros((240,320,3),np.uint8))
if tiles:
    g=np.vstack([np.hstack(tiles[i:i+3]) for i in range(0,len(tiles),3)])
    cv2.imwrite(os.path.join(DS,"day_flagged.jpg"),g); print("wrote day_flagged.jpg")
