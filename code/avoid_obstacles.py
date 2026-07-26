from picarx import Picarx
import time

px = Picarx()

SAFE_CM   = 25    # closer than this = blocked
LOST_NEAR = 40    # if echo lost right after seeing something this close, assume it's still there
SPEED     = 30

last_valid = 300  # what we saw most recently; start optimistic (open)

def back_and_turn():
    px.set_dir_servo_angle(30)
    px.backward(SPEED)
    time.sleep(0.8)
    px.set_dir_servo_angle(0)

try:
    while True:
        dist = px.get_distance()                 # SENSE

        if 0 <= dist <= 300:                     # valid reading
            last_valid = dist
            if dist < SAFE_CM:
                state = "BLOCKED -> back+turn"
                back_and_turn()
            else:
                state = "clear -> forward"
                px.set_dir_servo_angle(0); px.forward(SPEED)

        else:                                    # -2 / lost echo: AMBIGUOUS
            if last_valid < LOST_NEAR:
                state = f"lost echo, but saw {last_valid}cm -> assume wall, back+turn"
                back_and_turn()
                last_valid = 300                 # reset after escaping
            else:
                state = "lost echo, was open -> forward"
                px.set_dir_servo_angle(0); px.forward(SPEED)

        print(f"raw={dist:6.1f}  last_valid={last_valid:5.1f}  {state}")
        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    px.set_dir_servo_angle(0); px.stop()
