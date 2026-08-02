#!/usr/bin/env python3
"""Analyze temporal structure of labels.csv to pick the frame-stack stride and
count buildable sequences. A 'run' = frames with no timestamp gap > GAP_S."""
import csv, os, statistics as st
from datetime import datetime
HOME=os.path.expanduser("~"); LABELS=os.path.join(HOME,"dataset","labels.csv")
GAP_S=0.5; N=4; STRIDES=[1,2,3,4]

def parse(t):
    try: return datetime.fromisoformat(t).timestamp()
    except ValueError: return float(t)   # fallback if some rows are epoch floats

rows=[]
with open(LABELS,newline="") as f:
    for r in csv.DictReader(f):
        rows.append((int(r["index"]), parse(r["timestamp"]), r["image"]))
rows.sort(key=lambda x:x[1])
ts=[r[1] for r in rows]
dts=[ts[i]-ts[i-1] for i in range(1,len(ts))]
intra=[d for d in dts if 0<d<=GAP_S]
med=st.median(intra) if intra else 0
print(f"frames={len(rows)}")
print(f"median intra-frame dt={med*1000:.1f} ms  (~{1/med:.1f} Hz)" if med else "no dt")

runs=[]; start=0
for i in range(1,len(rows)):
    if ts[i]-ts[i-1] > GAP_S:
        runs.append((start,i)); start=i
runs.append((start,len(rows)))
lens=[b-a for a,b in runs]
print(f"runs={len(runs)}  run-length min/median/max = {min(lens)}/{int(st.median(lens))}/{max(lens)}")
for s in STRIDES:
    span=(N-1)*s*med; need=(N-1)*s
    buildable=sum(max(0,(b-a)-need) for a,b in runs)
    print(f"  N={N} stride={s}: span ~{span:.2f}s | buildable sequences={buildable}")
