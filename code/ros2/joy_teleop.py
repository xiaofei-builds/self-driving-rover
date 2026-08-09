#!/usr/bin/env python3
"""
joy_teleop.py — drive the rover with a Joy-Con (or any gamepad).

Subscribes : /joy       sensor_msgs/Joy      (published by `ros2 run joy joy_node`)
Publishes  : /cmd_vel   geometry_msgs/Twist  (consumed by drive_node)

Same command model as key_teleop.py: the operator commands a STEERING ANGLE,
which is converted to a yaw rate at the current speed before publishing, so the
feel does not change with speed.

DEADMAN: nothing moves unless a deadman button is held. Releasing it publishes
zeros immediately rather than going silent -- an explicit "stop" beats waiting
for the watchdog to time out, and it keeps the topic alive so downstream nodes
can tell the difference between "commanded to stop" and "link died".

Run:
    ros2 run joy joy_node          # terminal A
    python3 joy_teleop.py          # terminal B

If it steers the wrong way:
    python3 joy_teleop.py --ros-args -p steer_axis_sign:=-1.0
If forward/back is inverted:
    python3 joy_teleop.py --ros-args -p speed_axis_sign:=-1.0
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class JoyTeleop(Node):

    def __init__(self):
        super().__init__('joy_teleop')

        # --- which stick / button does what (measured, not assumed) ---------
        # Measured on a right Joy-Con held sideways, Session 14 (2026-08-08):
        #   axis 0 = stick up/down, "away from you" reads POSITIVE  -> sign +1
        #   axis 1 = stick left/right, "right" reads POSITIVE      -> sign +1
        # NOTE: +1 is correct END-TO-END, verified on the car. Reasoning about
        # this node alone says -1 (REP-103 wants +ve = left), but drive_node
        # already applies steer_sign=-1 for the SunFounder servo, and the two
        # flips cancel. Sign conventions are only verifiable end-to-end.
        self.declare_parameter('speed_axis', 0)
        self.declare_parameter('steer_axis', 1)
        self.declare_parameter('speed_axis_sign', 1.0)
        self.declare_parameter('steer_axis_sign', 1.0)
        self.declare_parameter('deadman_buttons', [4, 5])  # SL / SR on the rail

        # --- how much command a full stick deflection means -----------------
        self.declare_parameter('max_speed', 0.30)        # m/s
        self.declare_parameter('max_steer_deg', 30.0)    # matches the car
        self.declare_parameter('wheelbase_m', 0.094)     # must match drive_node
        self.declare_parameter('deadzone', 0.15)         # stick drift guard
        self.declare_parameter('rate_hz', 20.0)

        g = self.get_parameter
        self.speed_axis = g('speed_axis').value
        self.steer_axis = g('steer_axis').value
        self.speed_sign = g('speed_axis_sign').value
        self.steer_sign = g('steer_axis_sign').value
        self.deadman = list(g('deadman_buttons').value)
        self.max_speed = g('max_speed').value
        self.max_steer = g('max_steer_deg').value
        self.L = g('wheelbase_m').value
        self.deadzone = g('deadzone').value
        rate = g('rate_hz').value

        self.v = 0.0
        self.steer_deg = 0.0
        self.live = False          # deadman held?
        self.seen_joy = False

        self.create_subscription(Joy, 'joy', self.on_joy, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f'joy_teleop up | speed=axis{self.speed_axis}x{self.speed_sign:+.0f} '
            f'steer=axis{self.steer_axis}x{self.steer_sign:+.0f} '
            f'deadman=buttons{self.deadman} | HOLD a rail button to drive')

    # ------------------------------------------------------------------
    @staticmethod
    def apply_deadzone(x: float, dz: float) -> float:
        """Zero out small values, then rescale so the usable range still
        reaches full deflection. Without the rescale you would lose `dz` of
        travel and never reach 1.0."""
        if abs(x) < dz:
            return 0.0
        return (x - math.copysign(dz, x)) / (1.0 - dz)

    # ------------------------------------------------------------------
    def on_joy(self, msg: Joy):
        self.seen_joy = True

        self.live = any(
            b < len(msg.buttons) and msg.buttons[b] == 1 for b in self.deadman)

        if not self.live:
            self.v = 0.0
            self.steer_deg = 0.0
            return

        raw_speed = msg.axes[self.speed_axis] if self.speed_axis < len(msg.axes) else 0.0
        raw_steer = msg.axes[self.steer_axis] if self.steer_axis < len(msg.axes) else 0.0

        s = self.apply_deadzone(raw_speed * self.speed_sign, self.deadzone)
        t = self.apply_deadzone(raw_steer * self.steer_sign, self.deadzone)

        self.v = s * self.max_speed
        self.steer_deg = t * self.max_steer

    # ------------------------------------------------------------------
    def tick(self):
        # Inverse bicycle model, identical to key_teleop.py.
        w = self.v * math.tan(math.radians(self.steer_deg)) / self.L

        msg = Twist()
        msg.linear.x = float(self.v)
        msg.angular.z = float(w)
        self.pub.publish(msg)

        state = 'DRIVE' if self.live else ('idle ' if self.seen_joy else 'NO /joy')
        print(f'\r [{state}] v={self.v:+.2f} m/s  steer={self.steer_deg:+5.1f} deg   ',
              end='', flush=True)


def main():
    rclpy.init()
    node = JoyTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())      # explicit stop on the way out
        print('\nstopped')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
