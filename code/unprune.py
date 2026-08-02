#!/usr/bin/env python3
"""unprune.py — Session 10: restore the 800 tape-less frames pruned in S9 so runs
are contiguous and apex frames return as targets for the stateful (stacked) model.
Restores only frames whose labels survive in labels.csv.bak. Dry-run by default."""
import csv, os, sys, shutil
HOME=os.path.expanduser("~"); DS=os.path.join(HOME,"dataset")
DATA=os.path.join(DS,"data"); REJECTED=os.path.join(DS,"rejected")
LABELS=os.path.join(DS,"labels.csv"); BAK=os.path.join(DS,"labels.csv.bak")
PRE=os.path.join(DS,"labels.csv.pre_unprune"); APPLY="--apply" in sys.argv

def read_images(path):
    imgs=[]
    with open(path,newline="") as f:
        for row in csv.DictReader(f): imgs.append(row["image"])
    return imgs

def main():
    for p in (LABELS,BAK,DATA,REJECTED):
        if not os.path.exists(p): print("ABORT: missing",p); return
    bak=read_images(BAK); act=read_images(LABELS)
    bset,aset=set(bak),set(act)
    print(f"backup labels : {len(bak)} frames"); print(f"active labels : {len(act)} frames")
    if aset-bset:
        print("WARNING: active has frames not in backup — stopping."); return
    to_restore=sorted(bset-aset); print(f"to restore    : {len(to_restore)} frames")
    in_data=set(os.listdir(DATA)); missing=[]; already=[]
    for img in to_restore:
        if img in in_data: already.append(img)
        elif not os.path.exists(os.path.join(REJECTED,img)): missing.append(img)
    if already: print(f"NOTE: {len(already)} already in data/: {already[:3]}")
    if missing: print(f"ABORT: {len(missing)} not in rejected/: {missing[:5]}"); return
    print(f"\nAll {len(to_restore)} located in rejected/. After restore data/ holds {len(bak)} frames.")
    if not APPLY: print("\nDRY RUN — nothing changed. Re-run with --apply."); return
    print("\nAPPLYING..."); shutil.copy2(LABELS,PRE); print("  backed up ->",os.path.basename(PRE))
    moved=0
    for img in to_restore:
        s=os.path.join(REJECTED,img); d=os.path.join(DATA,img)
        if os.path.exists(s): shutil.move(s,d); moved+=1
    print(f"  moved {moved} images rejected/ -> data/")
    shutil.copy2(BAK,LABELS); print(f"  swapped in fuller labels ({len(bak)} frames)")
    new=read_images(LABELS); dnow=set(os.listdir(DATA))
    miss=[i for i in new if i not in dnow]
    print(f"\nINTEGRITY: labels={len(new)}  images_in_data={len(dnow)}")
    print("  WARNING missing:",miss[:5]) if miss else print("  OK — all labelled frames present.")
    print("Done. (Reverse via labels.csv.pre_unprune if needed.)")

main()
