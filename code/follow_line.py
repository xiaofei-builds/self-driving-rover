from picarx import Picarx
import time

px = Picarx()

FLOOR_CAL = [981.6, 1109.4, 1222.7]
TAPE_CAL  = [1196.6, 1349.9, 1410.4]
MID = [(FLOOR_CAL[i] + TAPE_CAL[i]) / 2 for i in range(3)]

SANE_MAX = 2000
HYS   = 25
ADAPT = 0.05
SPEED = 10
STEER = 30
GAIN  = 30
DRIVE = True

def warm_up():
    stable, last = 0, None
    while stable < 10:
        g = px.get_grayscale_data()
        ok = all(v < SANE_MAX for v in g)
        if ok and last and all(abs(g[i]-last[i]) < 60 for i in range(3)):
            stable += 1
        else:
            stable = 0
        last = g
        print(f"warmup g={g} stable={stable}")
        time.sleep(0.05)
    print("--- settled, starting ---")

warm_up()

delta   = [0.0, 0.0, 0.0]
on_tape = [False, False, False]
last_side  = 1
lost_count = 0
crossed    = False
prev_all_on = False

try:
    while True:
        g = px.get_grayscale_data()
        if any(v >= SANE_MAX for v in g):
            print(f"g={g} BAD READ -> skip"); time.sleep(0.05); continue

        thr = [MID[i] + delta[i] for i in range(3)]
        for i in range(3):
            if on_tape[i]:
                if g[i] < thr[i] - HYS: on_tape[i] = False
            else:
                if g[i] > thr[i] + HYS: on_tape[i] = True
        for i in range(3):
            if   g[i] > thr[i] + HYS: delta[i] = (1-ADAPT)*delta[i] + ADAPT*(g[i]-TAPE_CAL[i])
            elif g[i] < thr[i] - HYS: delta[i] = (1-ADAPT)*delta[i] + ADAPT*(g[i]-FLOOR_CAL[i])

        if on_tape[0] and not on_tape[2]:   last_side = -1
        elif on_tape[2] and not on_tape[0]: last_side = 1

        any_on = any(on_tape)
        all_on = all(on_tape)

        if any_on:                                         # FOLLOW
            lost_count = 0; crossed = False
            leftmost = min(i for i in range(3) if on_tape[i])
            error = (leftmost - 0.5) - 0.5
            angle = max(-STEER, min(STEER, error*GAIN))
            px.set_dir_servo_angle(angle); px.forward(SPEED)
            state = f"FOLLOW tape={[int(t) for t in on_tape]} ang={angle:+.1f} side={last_side:+d}"
        else:
            lost_count += 1
            if lost_count == 1: crossed = prev_all_on      # just drove across the whole line?
            if crossed and lost_count <= 10:               # CROSSED -> back straight up
                px.set_dir_servo_angle(0); px.backward(SPEED-2)
                state = "CROSSED -> backup"
            elif lost_count <= 20:                         # RECOVER -> toward last side
                px.set_dir_servo_angle(last_side*STEER); px.forward(SPEED-3)
                state = f"RECOVER side={last_side:+d}"
            elif lost_count <= 70:                         # SEARCH -> sweep both ways
                sdir = last_side if (lost_count//12) % 2 == 0 else -last_side
                px.set_dir_servo_angle(sdir*STEER); px.forward(SPEED-4)
                state = "SEARCH sweep"
            else:                                          # STOP
                px.stop(); state = "STOP (give up)"

        prev_all_on = all_on
        print(f"g={g} {state}")
        time.sleep(0.05)

except KeyboardInterrupt:
    pass
finally:
    px.set_dir_servo_angle(0); px.stop()
