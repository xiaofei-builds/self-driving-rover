#!/usr/bin/env python3
"""autopilot_recovery.py  —  stateful autopilot + a supervisory RECOVERY fallback.

Wraps the frame-stacked (memory) CNN policy from autopilot_stateful.py with a small
state machine that watches the policy for signs of confusion and takes over when the
car goes out-of-distribution (OOD).

WHY (the idea, banked in Session 11/12):
  When the tape is not meaningfully present in any of the 4 stacked frames, the CNN has
  no real cue and its steering THRASHES — rapid +/- sign-flips frame to frame. That
  thrash IS the uncertainty signal (epistemic uncertainty: OOD / thin data — reducible).
  Rather than let a confused net drive, we:
      DRIVE   -> run the CNN, but measure thrash on the LIVE output stream
      RECOVER -> stop, reverse SLOWLY retracing the last committed arc, and watch the
                 CNN output RE-STABILIZE. Hand control back the moment the CNN is
                 confident again (its steering settles); a grayscale "tape jump" can
                 let us hand back SOONER but never blocks.
      STOPPED -> fail-safe: if we reverse past REV_MAX seconds without recovering, stop
                 (we have NO rear sensor — never reverse blindly for long).

DESIGN NOTES (from Xiaofei's critique, S12):
  * The handback criterion is CNN RE-CONFIDENCE, not "nose is on tape". The policy
    drives off the CAMERA (forward view); whether the down-facing grayscale sensors sit
    over tape is not the success condition. So grayscale is a HINT, never a gate.
  * A fixed/startup grayscale baseline is useless: bare-floor reflectance and lighting
    vary ALONG the track, so an absolute reference goes stale. Instead we ZERO the
    grayscale LOCALLY at the instant of confusion and detect a *change* (a jump) during
    reverse — spatial floor drift no longer matters.
  * Thrash does not guarantee the car is fully off tape (a memory failure can thrash
    with the nose still on tape). So the floor estimate is a RUNNING MINIMUM during
    reverse (tape reads higher than floor -> the lowest reading = bare floor), which
    self-corrects if the snapshot happened to land on tape.
  * Why keep grayscale at all: output-stability can't distinguish "confidently sees
    tape" from "confidently stuck on a wrong value". A grayscale jump is INDEPENDENT
    physical evidence that tape returned — it guards the blind spot of the thrash proxy.
    (True MC-Dropout uncertainty is the planned follow-up; TF-Lite bakes dropout as a
    no-op at inference, so MC-Dropout needs a dropout-always-on re-export, not a retrain.)

Safe by default: NO motors unless --drive is passed (steering servo still moves so you
can watch decisions). --seconds N auto-stops. --gain X scales CNN steering.

Deploy contract inherited from autopilot_stateful.py: cam tilt -10, resize 160x120,
BGR->RGB, /255 float32, stack of N=4 frames spaced DT_STRIDE=4/15s in REAL TIME (picked
from a timestamped ring buffer, NOT by loop iteration), model output x30 = degrees.
"""

import numpy as np
from collections import deque

# ----------------------------------------------------------------------------------
# PURE DECISION LOGIC (numpy only — no hardware; unit-tested off-Pi)
# ----------------------------------------------------------------------------------

def thrash_stats(window, deadband=2.0):
    """Given a window of recent commanded steering angles (degrees), return
    (std, sign_flips). sign_flips = number of +/- reversals, ignoring near-zero
    values (|angle| < deadband) so straight-ahead jitter doesn't count as thrash."""
    a = np.asarray(window, dtype=np.float32)
    if a.size < 2:
        return 0.0, 0
    std = float(np.std(a))
    signs = np.sign(np.where(np.abs(a) < deadband, 0.0, a))
    nz = signs[signs != 0.0]
    flips = int(np.sum(nz[1:] != nz[:-1])) if nz.size >= 2 else 0
    return std, flips


def is_confused(hist, W, std_hi, flip_hi, deadband=2.0):
    """True if the last W outputs look like OOD thrash: high spread OR many sign-flips."""
    if len(hist) < W:
        return False
    std, flips = thrash_stats(list(hist)[-W:], deadband)
    return (std > std_hi) or (flips >= flip_hi)


def is_calm(hist, W, std_lo, flip_lo, deadband=2.0):
    """True if the last W outputs look settled: low spread AND few sign-flips.
    std_lo < std_hi and flip_lo < flip_hi give hysteresis (a dead-band between the
    'confused' and 'calm' verdicts so we don't chatter on the boundary)."""
    if len(hist) < W:
        return False
    std, flips = thrash_stats(list(hist)[-W:], deadband)
    return (std < std_lo) and (flips <= flip_lo)


def flip_check(window, mag=8.0, swing=12.0, min_side=3):
    """Detect a FAST, LARGE collapse of a committed turn — the failure family that covers
    both a confident sign-REVERSAL (-20 -> +20) and a mid-corner GIVE-UP (-15 -> +2, where
    the turn is abandoned even though the new value never commits the other way; the live
    run1.csv showed exactly this: mean -12.4 -> +2.8, a 15.2 drop).

    Returns (fired, committed_dir): committed_dir is the sign of the turn that was
    abandoned = the direction the CNN must return to before we hand control back.

    Scans every split into an earlier part and a later part (each >= min_side samples).
    Fires if for some split the earlier part is a committed turn (|mean| > mag) AND the
    turn WEAKENED/REVERSED by more than `swing` degrees along that committed direction:
        drop = (old_mean - new_mean) * sign(old_mean) > swing
    Measuring the drop along the committed direction means:
      * turning HARDER (old +15 -> new +30) gives a negative drop -> never fires.
      * a SLOW corner exit ramps gently -> small drop inside the short window -> no fire.
      * thrash: the committed (old) half averages toward ~0 -> |old|>mag fails -> no fire
        (caught by is_confused instead).
      * only a fast, large abandonment of a real turn trips it (the car cannot physically
        swing that far that fast, so it is implausible = a model failure).
    Scanning + min_side keep it True across a few ticks (so --flip-ticks can confirm) and
    stop a single glitch frame from fabricating a committed side."""
    a = np.asarray(window, dtype=np.float32)
    n = a.size
    if n < 2 * min_side:
        return False, 0.0
    for k in range(min_side, n - min_side + 1):
        old = float(a[:k].mean()); new = float(a[k:].mean())
        if abs(old) > mag:
            s = 1.0 if old > 0 else -1.0
            if (old - new) * s > swing:
                return True, s
    return False, 0.0


def confident_flip(window, mag=8.0, swing=12.0, min_side=3):
    """Boolean wrapper over flip_check (see it for the full rationale)."""
    return flip_check(window, mag, swing, min_side)[0]


def update_floor(gs_floor, vals):
    """Running per-sensor MINIMUM = adaptive bare-floor estimate during recovery.
    Tape reads HIGHER than floor, so the lowest reading seen so far tracks the floor
    even as lighting/surface drift, and self-corrects if the recovery snapshot happened
    to land on tape."""
    return [min(gs_floor[i], v) for i, v in enumerate(vals)]


def tape_jump(vals, gs_floor, jump_margin):
    """Change-detection: True if ANY sensor now reads jump_margin ABOVE the running
    floor estimate = a floor->tape crossing. Relative, so it doesn't depend on absolute
    brightness (which drifts with lighting) — only on the local contrast."""
    return any(v > gs_floor[i] + jump_margin for i, v in enumerate(vals))


def should_reacquire(cnn_ready, calm_ct, reacquire_ticks, gs_hint):
    """Handback rule. CNN re-confidence is NECESSARY (the policy must be able to steer
    reliably again). Sustained calm alone recovers; a grayscale tape-jump lets us hand
    back the instant the CNN is calm, without waiting the full sustained count.
    Grayscale accelerates; it never gates."""
    return (calm_ct >= reacquire_ticks) or (cnn_ready and gs_hint)


# ----------------------------------------------------------------------------------
# LIVE DRIVER (hardware imports happen inside main so this file imports clean off-Pi)
# ----------------------------------------------------------------------------------

def main():
    import time, argparse, cv2
    from ai_edge_litert.interpreter import Interpreter
    from picarx import Picarx
    from vilib import Vilib

    N, DT_STRIDE = 4, 4 / 15.0           # 4 frames, 0.267s apart -> 0.80s span (train match)
    SPAN = (N - 1) * DT_STRIDE
    H, W = 120, 160

    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", action="store_true", help="actually spin the motors")
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop after N s")
    ap.add_argument("--gain", type=float, default=1.2, help="scale CNN steering (S11 used 1.2)")
    ap.add_argument("--model", default="/home/pi/pilot_stateful.tflite")
    ap.add_argument("--throttle", type=int, default=10, help="forward speed (match collection=10)")
    # --- recovery tuning ---
    ap.add_argument("--win", type=int, default=8, help="thrash window (# recent outputs)")
    ap.add_argument("--std-hi", type=float, default=12.0, help="std(deg) above = confused")
    ap.add_argument("--flip-hi", type=int, default=3, help="sign-flips in window above = confused")
    ap.add_argument("--std-lo", type=float, default=5.0, help="std(deg) below = calm (hysteresis)")
    ap.add_argument("--flip-lo", type=int, default=1, help="sign-flips at/below = calm")
    ap.add_argument("--confuse-ticks", type=int, default=8, help="consecutive THRASH ticks to trip")
    ap.add_argument("--flip-mag", type=float, default=8.0, help="|steer| the abandoned turn must have reached")
    ap.add_argument("--flip-swing", type=float, default=12.0, help="deg the committed turn must collapse (fast) to trip")
    ap.add_argument("--flip-ticks", type=int, default=2, help="consecutive flip ticks to trip (fires fast)")
    ap.add_argument("--reacquire-ticks", type=int, default=5, help="consecutive calm ticks to resume (CNN-only path)")
    ap.add_argument("--rev-throttle", type=int, default=8, help="reverse speed during recovery")
    ap.add_argument("--rev-max", type=float, default=2.5, help="MAX reverse seconds before fail-safe STOP")
    # --- grayscale HINT (change-detection; accelerates handback, never gates) ---
    ap.add_argument("--gs-jump", type=float, default=150.0, help="rise over running floor = tape crossing")
    ap.add_argument("--no-gs", action="store_true", help="ignore grayscale; recover on CNN re-confidence only")
    ap.add_argument("--log", default="", help="write a per-tick CSV (t,state,steer,std,flips,flip) for tuning")
    a = ap.parse_args()

    logf = open(a.log, "w") if a.log else None
    if logf:
        logf.write("t,state,steer,std,flips,flip\n")

    itp = Interpreter(model_path=a.model); itp.allocate_tensors()
    inp = itp.get_input_details()[0]; out = itp.get_output_details()[0]
    assert tuple(inp["shape"]) == (1, H, W, N * 3), inp["shape"]   # 12-channel stateful model

    px = Picarx(); px.set_cam_tilt_angle(-10)                      # extrinsic must match collection
    Vilib.camera_start(vflip=False, hflip=False); Vilib.display(local=False, web=True)
    time.sleep(2)

    # ---- ring buffer + preprocessing (same as autopilot_stateful.py) ----
    buf = deque()   # (t, frame_rgb_float) oldest..newest
    def preprocess(bgr):
        r = cv2.resize(bgr, (W, H)); return (r[:, :, ::-1].astype(np.float32)) / 255.0

    def pick_stack(now):
        picks = []
        for k in range(N - 1, -1, -1):                 # old -> new
            target = now - k * DT_STRIDE
            _, fr = min(buf, key=lambda tfr: abs(tfr[0] - target))
            picks.append(fr)
        return np.concatenate(picks, axis=2)[None].astype(np.float32)  # (1,H,W,12)

    def cnn_steer(now):
        x = pick_stack(now)
        itp.set_tensor(inp["index"], x); itp.invoke()
        raw = float(itp.get_tensor(out["index"])[0][0]) * 30.0
        return max(-30.0, min(30.0, raw * a.gain))

    def read_gs():
        try:
            return list(px.get_grayscale_data())
        except Exception:
            return None

    hist = deque(maxlen=a.win)          # recent commanded steering (deg)
    state = "DRIVE"
    confuse_ct = 0
    flip_ct = 0
    calm_ct = 0
    recover_dir = 0.0                    # after a FLIP: sign the CNN must return to before handback
    rev_start = None
    gs_floor = None                     # per-sensor running-min floor estimate (set at RECOVER entry)
    gs_hint = False
    last_print = 0.0                    # throttle the per-tick heartbeat
    t0 = time.time()
    print("STATE: DRIVE  (safe mode)" if not a.drive else "STATE: DRIVE  (--drive: motors live)")

    try:
        while True:
            now = time.time()
            frame = Vilib.img
            if frame is None:
                continue
            buf.append((now, preprocess(frame)))
            while buf and now - buf[0][0] > SPAN + 0.3:
                buf.popleft()
            if buf[-1][0] - buf[0][0] < SPAN * 0.9:     # not enough history yet
                continue

            steer = cnn_steer(now)
            hist.append(steer)

            if state == "DRIVE":
                px.set_dir_servo_angle(steer)
                if a.drive:
                    px.forward(a.throttle)
                # Two independent confusion triggers, each debounced on its own timescale:
                #  - THRASH: sustained high-variance oscillation (needs --confuse-ticks)
                #  - FLIP: a fast collapse/reversal of a committed turn; goes steady almost
                #          immediately, so it must fire fast (--flip-ticks)
                confuse_ct = confuse_ct + 1 if is_confused(hist, a.win, a.std_hi, a.flip_hi) else 0
                flipped, flip_dir = flip_check(list(hist), a.flip_mag, a.flip_swing)
                flip_ct = flip_ct + 1 if flipped else 0
                trip = None
                if confuse_ct >= a.confuse_ticks:
                    trip = "THRASH"
                elif flip_ct >= a.flip_ticks:
                    trip = "FLIP"
                if trip:
                    std, flips = thrash_stats(list(hist), 2.0)
                    if trip == "FLIP":
                        recover_dir = flip_dir          # correct direction = the abandoned turn's sign
                        print("CONFUSED[FLIP] collapse -> RECOVER (reverse straight); "
                              "correct dir = %s" % ("LEFT" if recover_dir < 0 else "RIGHT"))
                    else:
                        recover_dir = 0.0               # thrash has no single 'correct' side
                        print("CONFUSED[THRASH] (std=%.1f flips=%d) -> RECOVER (reverse straight)"
                              % (std, flips))
                    state = "RECOVER"
                    rev_start = now; calm_ct = 0; confuse_ct = 0; flip_ct = 0
                    hist.clear()                        # measure re-stabilization on FRESH reverse-phase outputs
                    gs_floor = read_gs()                # ZERO grayscale locally, right now (adaptive baseline)
                    if a.drive: px.forward(0)

            elif state == "RECOVER":
                px.set_dir_servo_angle(0)               # STRAIGHT reverse (simpler, safer)
                if a.drive:
                    px.backward(a.rev_throttle)

                cnn_ready = is_calm(hist, a.win, a.std_lo, a.flip_lo)
                # after a FLIP: don't hand back to a still-confidently-WRONG model. Require
                # its steering to return to the correct (pre-flip) direction. THRASH has no
                # directional requirement (recover_dir == 0 -> always dir_ok).
                dir_ok = (recover_dir == 0.0) or (steer * recover_dir > 0.0)
                cnn_ok = cnn_ready and dir_ok
                calm_ct = calm_ct + 1 if cnn_ok else 0

                gs_hint = False
                if not a.no_gs:
                    gs = read_gs()
                    if gs is not None:
                        if gs_floor is None:
                            gs_floor = gs
                        gs_floor = update_floor(gs_floor, gs)   # running-min = adaptive floor
                        gs_hint = tape_jump(gs, gs_floor, a.gs_jump)

                if should_reacquire(cnn_ok, calm_ct, a.reacquire_ticks, gs_hint):
                    print("REACQUIRED (cnn_ready=%s dir_ok=%s gs_jump=%s) -> DRIVE" % (cnn_ready, dir_ok, gs_hint))
                    if a.drive: px.backward(0)
                    state = "DRIVE"; confuse_ct = 0; flip_ct = 0; hist.clear()   # fresh window before re-arming
                elif now - rev_start > a.rev_max:
                    print("FAIL-SAFE: reversed %.1fs without recovering -> STOP" % a.rev_max)
                    state = "STOPPED"

            elif state == "STOPPED":
                px.forward(0); px.set_dir_servo_angle(0); px.stop()
                break

            # ---- per-tick CSV log (unthrottled) for offline threshold tuning ----
            if logf:
                lstd, lflips = thrash_stats(list(hist), 2.0)
                logf.write("%.3f,%s,%.1f,%.1f,%d,%d\n" % (
                    now - t0, state, steer, lstd, lflips,
                    int(confident_flip(list(hist), a.flip_mag, a.flip_swing))))

            # ---- live heartbeat (throttled) so you can watch the steering / thrash ----
            if now - last_print >= 0.3:
                std, flips = thrash_stats(list(hist), 2.0)
                if state == "RECOVER":
                    print("[RECOVER] reverse straight  cnn=%+6.1f  need_dir=%s  std=%4.1f  gs_jump=%s"
                          % (steer, "LEFT" if recover_dir < 0 else ("RIGHT" if recover_dir > 0 else "any"),
                             std, gs_hint))
                else:
                    flip_flag = " FLIP!" if confident_flip(list(hist), a.flip_mag, a.flip_swing) else ""
                    print("[%s] steer=%+6.1f  std=%4.1f flips=%d%s" % (state, steer, std, flips, flip_flag))
                last_print = now

            if a.seconds and now - t0 > a.seconds:
                break
    finally:
        px.forward(0); px.set_dir_servo_angle(0); px.stop()
        Vilib.camera_close()
        if logf:
            logf.close()


if __name__ == "__main__":
    main()
