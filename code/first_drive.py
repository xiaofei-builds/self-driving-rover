from picarx import Picarx
from time import sleep

px = Picarx()
try:
    px.set_dir_servo_angle(0)   # straight — applies your saved -8.4 offset
    sleep(0.5)
    px.forward(30)              # 30% throttle
    sleep(1.5)                  # drive for 1.5 seconds
    px.stop()
finally:
    px.stop()                   # guarantees motors cut even on Ctrl+C
    px.set_dir_servo_angle(0)
