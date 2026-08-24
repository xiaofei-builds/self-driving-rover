#!/usr/bin/env python3
"""
override_teleop.py — the safety-driver console for the M6 stack (Session 17b).

Replaces the "infer who's driving from the Twist values" mistake in the first mux.
This node publishes the operator's INTENT explicitly — which mode, the actual steering
angle, whether the operator is currently steering, and a manual speed — on a single
topic. The mux reads that intent and never has to guess. That one change fixes all four
problems we hit on the car:
  * a/d works instantly (steering angle is sent directly, not as a yaw that vanishes at
    zero speed),
  * taking over doesn't jump the speed (full-manual seeds from the car's current speed),
  * handing back is instant (a mode key, no forced stop),
  * STOP is a real latched e-stop.

THREE MODES (you switch between them; the mux obeys):
  GUARDED (default) — autopilot drives; a/d NUDGE the steering at the autopilot's speed,
                      and release returns steering to the autopilot after a short hold.
  MANUAL            — you drive everything (throttle + steer). Seeded from the car's
                      current speed on entry so there's no jump.
  STOP              — latched full stop until you re-engage.

KEYS
  a / d    steer left / right   (GUARDED = a nudge; MANUAL = your steering)
  c        centre steering
  w / s    faster / slower      (also grabs FULL MANUAL — touching throttle = you drive)
  m        take FULL MANUAL     (seeded from current speed)
  p        hand back to AUTOPILOT (guarded)
  space    STOP (latched)
  q        quit

Publishes
---------
  /teleop/override   std_msgs/Float32MultiArray  [mode, steer_deg, steer_active, speed_mps]
                       mode: 0=GUARDED 1=MANUAL 2=STOP
                       steer_deg: operator steering ANGLE, +ve = LEFT (operator model)
                       steer_active: 1.0 if steering right now (recent a/d), else 0.0
                       speed_mps: operator's commanded speed (used in MANUAL)
Subscribes
----------
  /cmd_vel   geometry_msgs/Twist   — only to read the car's CURRENT speed, so entering
                                      MANUAL can seed from it (no speed jump).

The bicycle model + steer_sign live in the MUX now, not here: this node emits a raw
steering ANGLE and lets the mux convert it to a yaw at whatever speed applies. That is
why GUARDED can re-use the autopilot's speed for your nudge.

Run:
    python3 override_teleop.py
"""

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

GUARDED, MANUAL, STOP = 0.0, 1.0, 2.0

HELP = """
  a / d   steer left / right   (GUARDED = nudge; MANUAL = drive)
  c       centre steering
  w / s   faster / slower      (grabs FULL MANUAL)
  m       take FULL MANUAL
  p       hand back to AUTOPILOT
  space   STOP (latched)
  q       quit
"""


class OverrideTeleop(Node):

    def __init__(self):
        super().__init__('override_teleop')

        self.declare_parameter('speed_step', 0.02)      # m/s per keypress
        self.declare_parameter('steer_step_deg', 5.0)   # degrees per keypress
        self.declare_parameter('max_speed', 0.35)
        self.declare_parameter('max_steer_deg', 30.0)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('nudge_hold_s', 0.5)     # steering stays "active" this long after last a/d

        g = self.get_parameter
        self.speed_step = float(g('speed_step').value)
        self.steer_step = float(g('steer_step_deg').value)
        self.max_speed = float(g('max_speed').value)
        self.max_steer = float(g('max_steer_deg').value)
        rate = float(g('rate_hz').value)
        self.nudge_hold = float(g('nudge_hold_s').value)

        self.mode = GUARDED
        self.steer_deg = 0.0            # +ve = LEFT (operator model)
        self.speed = 0.0               # manual speed
        self.steer_active_until = None # rclpy Time; steering counts as "active" until here
        self.last_cmd_v = 0.0          # current car speed, from /cmd_vel (for seeding MANUAL)

        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.pub = self.create_publisher(Float32MultiArray, 'teleop/override', 10)
        self.create_timer(1.0 / rate, self.tick)

        print(HELP)
        print('mode=GUARDED (autopilot driving; a/d to nudge)')

    # ------------------------------------------------------------------
    def on_cmd_vel(self, msg: Twist):
        self.last_cmd_v = msg.linear.x

    # ------------------------------------------------------------------
    def _clamp_steer(self, x):
        return max(-self.max_steer, min(self.max_steer, x))

    def _arm_steer(self):
        self.steer_active_until = self.get_clock().now() + \
            rclpy.duration.Duration(seconds=self.nudge_hold)

    def _enter_manual(self):
        """Switch to full manual, seeding speed from the car's CURRENT speed so the
        takeover is smooth instead of snapping to whatever teleop last held."""
        if self.mode != MANUAL:
            self.speed = max(-self.max_speed, min(self.max_speed, self.last_cmd_v))
        self.mode = MANUAL

    # ------------------------------------------------------------------
    def _step_speed(self, delta):
        new = self.speed + delta
        if self.speed > 0 and new < 0:
            new = 0.0
        elif self.speed < 0 and new > 0:
            new = 0.0
        self.speed = max(-self.max_speed, min(self.max_speed, new))

    def apply_key(self, k):
        if k == 'a':
            self.steer_deg = self._clamp_steer(self.steer_deg + self.steer_step)
            self._arm_steer()
        elif k == 'd':
            self.steer_deg = self._clamp_steer(self.steer_deg - self.steer_step)
            self._arm_steer()
        elif k == 'c':
            self.steer_deg = 0.0
            self._arm_steer()
        elif k == 'w':
            self._enter_manual()
            self._step_speed(self.speed_step)
        elif k == 's':
            self._enter_manual()
            self._step_speed(-self.speed_step)
        elif k == 'm':
            self._enter_manual()
        elif k == 'p':
            self.mode = GUARDED           # hand back to autopilot, no stop
        elif k == ' ':
            self.mode = STOP              # latched stop
            self.speed = 0.0
        elif k == 'q':
            return False
        return True

    # ------------------------------------------------------------------
    def _steer_active(self):
        if self.steer_active_until is None:
            return False
        return self.get_clock().now() < self.steer_active_until

    def tick(self):
        active = self._steer_active()
        # In GUARDED, once a nudge lapses, recentre so the next tap starts from straight
        # (in MANUAL the operator holds a real steering angle, so don't touch it).
        if self.mode == GUARDED and not active:
            self.steer_deg = 0.0

        msg = Float32MultiArray()
        msg.data = [float(self.mode), float(self.steer_deg),
                    1.0 if active else 0.0, float(self.speed)]
        self.pub.publish(msg)

        name = {GUARDED: 'GUARDED', MANUAL: 'MANUAL', STOP: 'STOP '}[self.mode]
        print(f'\r [{name}] steer={self.steer_deg:+5.1f} {"NUDGE" if active else "     "} '
              f'speed={self.speed:+.2f} m/s   ', end='', flush=True)


def read_key(timeout=0.05):
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return ''


def main():
    rclpy.init()
    node = OverrideTeleop()
    settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            k = read_key()
            if k and not node.apply_key(k.lower()):
                break
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        # Leave a STOP intent on the way out so the car doesn't coast under autopilot
        # the instant the console dies.
        stop = Float32MultiArray()
        stop.data = [STOP, 0.0, 0.0, 0.0]
        node.pub.publish(stop)
        print('\nstopped')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
