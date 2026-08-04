#!/usr/bin/env python3
"""Off-Pi unit tests for the PURE decision logic in autopilot_recovery.py.
No hardware imports are touched (main() holds those). numpy only."""
import numpy as np
from collections import deque
import autopilot_recovery as R

fails = 0
def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails += 1

# ---- thrash_stats ----------------------------------------------------------------
std, flips = R.thrash_stats([20, -18, 19, -20, 18, -19, 20, -17])   # hard thrash
check("thrash: high std on alternating +/-20", std > 15)
check("thrash: counts ~7 sign flips on 8-alternating", flips >= 6)

std, flips = R.thrash_stats([20, 21, 19, 20, 22, 20, 19, 21])       # steady hard-right turn
check("steady turn: low std", std < 3)
check("steady turn: zero sign flips", flips == 0)

std, flips = R.thrash_stats([0.5, -0.4, 0.3, -0.6, 0.2, -0.5, 0.4, -0.3])  # near-zero jitter
check("deadband: tiny wobble around 0 -> zero flips (ignored)", flips == 0)

check("thrash: empty/short window is safe", R.thrash_stats([]) == (0.0, 0))

# ---- is_confused / is_calm + hysteresis -----------------------------------------
W = 8
thrash = deque([20, -18, 19, -20, 18, -19, 20, -17], maxlen=W)
turn   = deque([20, 21, 19, 20, 22, 20, 19, 21], maxlen=W)
short  = deque([20, -20, 19], maxlen=W)

check("confused: thrash window is confused", R.is_confused(thrash, W, 12.0, 3))
check("confused: steady turn is NOT confused", not R.is_confused(turn, W, 12.0, 3))
check("confused: under-full window never trips", not R.is_confused(short, W, 12.0, 3))
check("calm: steady turn is calm", R.is_calm(turn, W, 5.0, 1))
check("calm: thrash is NOT calm", not R.is_calm(thrash, W, 5.0, 1))
mid = deque([-10, -5, 0, 5, 10, 5, 0, -5], maxlen=W)   # std~6.1 (5<..<12), 2 flips (1<..<3)
check("hysteresis: mid window is neither confused nor calm",
      (not R.is_confused(mid, W, 12.0, 3)) and (not R.is_calm(mid, W, 5.0, 1)))

# ---- flip_check / confident_flip: fast collapse of a committed turn --------------
# (defaults mag=8, swing=12). Detects both full reversals AND mid-corner give-ups.
check("flip: full reversal -20->+20 detected", R.confident_flip([-20,-20,-19,-21,20,21,20,20]))
# mid-corner GIVE-UP: committed turn collapses toward the other side but never commits it
check("flip: give-up -14->+5 detected (never commits opposite)", R.confident_flip([-14,-15,-14,-15,5,5,6,5]))
# the exact run1.csv event: committed left mean ~-12.4 collapses to ~+2.8 (drop 15.2)
check("flip: the run1.csv collapse (-12 -> +3) detected", R.confident_flip([-12,-13,-12,-13,3,3,2,3]))
# the user's +15 -> -1 example
check("flip: +15 -> -1 give-up detected", R.confident_flip([15,15,16,15,-1,-1,0,-1]))
# turning HARDER is NOT a failure (drop is negative)
check("flip: turning harder (+15 -> +30) is NOT a flip", not R.confident_flip([15,15,16,15,30,30,29,30]))
# steady committed turn: no collapse
check("flip: steady committed turn is not a flip", not R.confident_flip([20]*8))
# thrash: the committed (old) half averages ~0 -> |old|<mag -> not a flip (is_confused handles it)
check("flip: oscillation is not a flip", not R.confident_flip([20,-20,19,-21,20,-19,21,-20]))
# legit SLOW S-curve: small drop within the short window -> never a flip
scurve = list(np.linspace(20, -20, 16))
check("flip: slow S-curve ramp is NOT a flip (any window)",
      not any(R.confident_flip(scurve[i:i+8]) for i in range(len(scurve) - 7)))
# a collapse from a turn that was never committed (|old|<mag) is ignored
check("flip: collapse from a small (non-committed) turn ignored", not R.confident_flip([5,4,5,4,-6,-6,-5,-6]))
# flip_check returns the committed direction (sign of the abandoned turn) for the retrace gate
fired, d = R.flip_check([-14,-15,-14,-15,5,5,6,5])
check("flip_check: committed dir LEFT (neg) for a left-turn give-up", fired and d < 0)
fired2, d2 = R.flip_check([15,15,16,15,-1,-1,0,-1])
check("flip_check: committed dir RIGHT (pos) for a right-turn give-up", fired2 and d2 > 0)

# ---- grayscale change-detection: update_floor + tape_jump ------------------------
floor = [981.6, 1109.4, 1222.7]     # S4 floor baseline
tape  = [1196.6, 1349.9, 1410.4]    # S4 tape reading
check("tape_jump: floor->tape crossing detected (jump 150)", R.tape_jump(tape, floor, 150))
check("tape_jump: staying on floor -> no jump", not R.tape_jump(floor, floor, 150))
# overlap trap: floor sensor-3 (1222) > tape sensor-1 (1196); per-sensor deltas still work
check("tape_jump: overlap-safe (only sensor-3 lit, still detected)",
      R.tape_jump([1000, 1120, 1410], floor, 150))

# running-min floor estimate self-corrects the DRIFT + ON-TAPE-AT-SNAPSHOT cases:
# snapshot taken while ON tape (high). As the car reverses onto floor, the min drops to
# the true floor; a later tape crossing then shows as a jump above that min.
gs_floor = list(tape)               # bad snapshot: confusion happened while still on tape
for r in ([1180,1330,1395], [1000,1120,1235], [985,1110,1225]):   # backing onto floor
    gs_floor = R.update_floor(gs_floor, r)
check("update_floor: running-min converges to true floor after a bad on-tape snapshot",
      gs_floor[0] <= 990 and gs_floor[2] <= 1230)
check("tape_jump: re-crossing tape AFTER the min corrected is detected",
      R.tape_jump(tape, gs_floor, 150))
# spatial floor drift: floor got brighter further along the track; min tracks it, tape still pops
drifted_floor = R.update_floor([1100,1200,1300],[1100,1200,1300])
check("tape_jump: on a brighter floor patch, floor alone is not a false jump",
      not R.tape_jump([1120,1215,1320], drifted_floor, 150))

# ---- handback rule: should_reacquire (CNN necessary; grayscale accelerates) ------
check("reacquire: sustained calm alone recovers (no gs)",
      R.should_reacquire(cnn_ready=True, calm_ct=5, reacquire_ticks=5, gs_hint=False))
check("reacquire: calm + gs jump recovers EARLY (calm_ct below threshold)",
      R.should_reacquire(cnn_ready=True, calm_ct=1, reacquire_ticks=5, gs_hint=True))
check("reacquire: gs jump WITHOUT cnn_ready does NOT recover (point 2: CNN necessary)",
      not R.should_reacquire(cnn_ready=False, calm_ct=0, reacquire_ticks=5, gs_hint=True))
check("reacquire: not calm, no gs, below threshold -> hold in RECOVER",
      not R.should_reacquire(cnn_ready=False, calm_ct=2, reacquire_ticks=5, gs_hint=False))

# ---- STATE MACHINE simulation (mirror of main()'s transition logic) -------------
def simulate(stream, gs_hint_stream, *, win=8, std_hi=12, flip_hi=3, std_lo=5, flip_lo=1,
             confuse_ticks=8, flip_mag=8, flip_swing=12, flip_ticks=2, reacquire_ticks=5,
             rev_max_ticks=100, no_gs=False):
    hist = deque(maxlen=win)
    state = "DRIVE"; confuse_ct = flip_ct = calm_ct = 0; rev_ticks = 0; recover_dir = 0.0
    visited = []
    for steer, gs_hint in zip(stream, gs_hint_stream):
        hist.append(steer)
        if state == "DRIVE":
            confuse_ct = confuse_ct + 1 if R.is_confused(hist, win, std_hi, flip_hi) else 0
            flipped, flip_dir = R.flip_check(list(hist), flip_mag, flip_swing)
            flip_ct = flip_ct + 1 if flipped else 0
            if confuse_ct >= confuse_ticks:
                recover_dir = 0.0                            # thrash: no single correct side
                state = "RECOVER"; calm_ct = 0; rev_ticks = 0; confuse_ct = flip_ct = 0; hist.clear()
            elif flip_ct >= flip_ticks:
                recover_dir = flip_dir                        # correct dir = abandoned turn's sign
                state = "RECOVER"; calm_ct = 0; rev_ticks = 0; confuse_ct = flip_ct = 0; hist.clear()
        elif state == "RECOVER":
            rev_ticks += 1
            cnn_ready = R.is_calm(hist, win, std_lo, flip_lo)
            dir_ok = (recover_dir == 0.0) or (steer * recover_dir > 0.0)
            cnn_ok = cnn_ready and dir_ok
            calm_ct = calm_ct + 1 if cnn_ok else 0
            hint = (not no_gs) and gs_hint
            if R.should_reacquire(cnn_ok, calm_ct, reacquire_ticks, hint):
                state = "DRIVE"; confuse_ct = 0; hist.clear()
            elif rev_ticks > rev_max_ticks:
                state = "STOPPED"
        visited.append(state)
    return visited

# near-straight before/after so ONLY the thrash path fires in these scenarios
# (a big committed value here would also trip the flip/collapse detector)
calm_run   = [2]*10
thrash_run = [20, -19, 18, -20, 19, -18, 20, -19]*3     # 24 ticks OOD
settle_run = [2]*20

# Scenario 1: OOD -> recover via CNN re-confidence alone (grayscale never hints)
v = simulate(calm_run + thrash_run + settle_run,
             [False]*(len(calm_run)+len(thrash_run)+len(settle_run)))
check("SM: starts in DRIVE", v[0] == "DRIVE")
check("SM: enters RECOVER during sustained thrash", "RECOVER" in v)
check("SM: recovers to DRIVE on CNN re-confidence alone", v[-1] == "DRIVE")

# Scenario 2: same, but a grayscale jump arrives -> should hand back EARLIER
gs_hint_stream = ([False]*(len(calm_run)+len(thrash_run))) + ([True]*len(settle_run))
v2 = simulate(calm_run + thrash_run + settle_run, gs_hint_stream)
def first_drive_after_recover(v):
    i = v.index("RECOVER")
    while i < len(v) and v[i] == "RECOVER": i += 1
    return i
check("SM: grayscale jump makes handback earlier than CNN-only",
      first_drive_after_recover(v2) < first_drive_after_recover(v))

# Scenario 3: thrash never settles AND no grayscale -> fail-safe STOP under rev cap
v3 = simulate(calm_run + thrash_run + [0, -1, 0, 1]*20,
              [False]*(len(calm_run)+len(thrash_run)+80), rev_max_ticks=20)
check("SM: fail-safe STOP when never recovers within rev cap", v3[-1] == "STOPPED")

# Scenario 4: grayscale jumps but CNN keeps thrashing -> must NOT hand back (point 2)
v4 = simulate(calm_run + thrash_run + [20,-20,19,-18]*20,
              [False]*(len(calm_run)+len(thrash_run)) + [True]*80, rev_max_ticks=200)
check("SM: gs jump with a still-thrashing CNN does NOT hand back", "DRIVE" not in v4[len(calm_run)+len(thrash_run):])

# Scenario 5: brief thrash blip shorter than confuse_ticks must NOT trip recovery
v5 = simulate([15]*10 + [20, -20, 19] + [15]*10, [False]*23, confuse_ticks=8)
check("SM: brief blip below confuse_ticks does NOT enter RECOVER", "RECOVER" not in v5)

# Scenario 6: the LIVE-LOG failure — committed left, confident flip to right, stays wrong.
# The thrash detector would MISS this (std collapses after the flip); the flip trigger must catch it.
flip_stream = [-18, -20, -21, -20, -19, -20] + [20, 21, 20, 22, 20, 21, 20, 20, 21, 20]  # flip then STEADY wrong
v6 = simulate(flip_stream, [False]*len(flip_stream), confuse_ticks=8, flip_ticks=3)
check("SM: confident flip (steady-wrong) IS caught via FLIP trigger", "RECOVER" in v6)
# ...and confirm the pure thrash path alone would NOT have caught it (std goes calm after flip)
def thrash_only(stream, win=8, std_hi=12, flip_hi=3, confuse_ticks=8):
    h = deque(maxlen=win); ct = 0
    for s in stream:
        h.append(s)
        ct = ct + 1 if R.is_confused(h, win, std_hi, flip_hi) else 0
        if ct >= confuse_ticks: return True
    return False
check("SM: thrash-only detector would MISS the confident flip (why we added FLIP)",
      not thrash_only(flip_stream))

# Scenario 7: the LIVE bug — after a flip, the model stays confidently WRONG (steady +14).
# Must NOT hand back while wrong; must resume only once it returns to the correct sign.
pre   = [-13, -14, -13, -12, -13, -14]        # committed LEFT (correct)
flip  = [13, 13, 14, 13]                       # confident flip to RIGHT -> trips FLIP
wrong = [14, 14, 13, 14, 13, 14, 13, 14, 13, 14, 13, 14]   # reverses, but model STILL says right
good  = [-13, -14, -13, -13, -14, -13, -13, -14] * 2        # model finally steers left again (flush + sustain)
v7 = simulate(pre + flip + wrong + good, [False]*(len(pre)+len(flip)+len(wrong)+len(good)),
              flip_mag=8, flip_ticks=2, reacquire_ticks=5, rev_max_ticks=999)
r0 = v7.index("RECOVER")
# during the 'wrong' phase it must remain in RECOVER (steady but wrong-direction != recovered)
wrong_phase = v7[r0: r0 + len(wrong)]
check("SM: does NOT hand back while model is steady-but-WRONG after a flip",
      all(s == "RECOVER" for s in wrong_phase))
check("SM: resumes DRIVE once the model steers the correct direction again", v7[-1] == "DRIVE")

print("\n%s  (%d failures)" % ("ALL PASS" if fails == 0 else "SOME FAILED", fails))
