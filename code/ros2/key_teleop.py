#!/usr/bin/env python3
"""
key_teleop.py — keyboard teleop publishing geometry_msgs/Twist on /cmd_vel.

Replaces the broken ros-jazzy-teleop-twist-keyboard deb (which shipped the
ROS 1 source). Key mapping deliberately matches collect_data.py from Session 5.

    w / s   speed up / slow down   (s below zero = reverse)
    a / d   steer left / right
    c       centre the steering
    space   STOP (zero speed and steering)
    q       quit

Why it publishes continuously at a fixed rate rather than only on keypress:
drive_node has a 0.5 s watchdog. A teleop that only sent on keypress would
trip it constantly. Real teleop links behave the same way — a continuous
heartbeat of intent, so silence unambiguously means "something is wrong".

Run:
    python3 key_teleop.py
    python3 key_teleop.py --ros-args -p speed_step:=0.05 -p omega_step:=0.3
"""

import math
import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

HELP = """
    w / s   faster / slower (negative = reverse)
    a / d   steer left / right
    c       centre steering
    space   STOP
    q       quit
"""


class KeyTeleop(Node):

    def __init__(self):
        super().__init__('key_teleop')

        self.declare_parameter('speed_step', 0.05)      # m/s per keypress
        self.declare_parameter('steer_step_deg', 5.0)   # degrees per keypress
        self.declare_parameter('max_speed', 0.35)       # m/s
        self.declare_parameter('max_steer_deg', 30.0)   # matches the car's limit
        self.declare_parameter('wheelbase_m', 0.094)    # must match drive_node
        self.declare_parameter('rate_hz', 20.0)

        self.speed_step = self.get_parameter('speed_step').value
        self.steer_step = self.get_parameter('steer_step_deg').value
        self.max_speed = self.get_parameter('max_speed').value
        self.max_steer = self.get_parameter('max_steer_deg').value
        self.L = self.get_parameter('wheelbase_m').value
        rate = self.get_parameter('rate_hz').value

        # We track a STEERING ANGLE, not a yaw rate. A steering wheel means the
        # same thing at every speed; a yaw rate does not. Converted to
        # angular.z on publish so the wire format stays a standard Twist.
        self.v = 0.0
        self.steer_deg = 0.0    # +ve = LEFT, per REP-103

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(1.0 / rate, self.tick)

        print(HELP)

    # ------------------------------------------------------------------
    def apply_key(self, k: str) -> bool:
        """Returns False if the user asked to quit."""
        if k == 'w':
            self.v = min(self.max_speed, self.v + self.speed_step)
        elif k == 's':
            self.v = max(-self.max_speed, self.v - self.speed_step)
        elif k == 'a':
            self.steer_deg = min(self.max_steer, self.steer_deg + self.steer_step)
        elif k == 'd':
            self.steer_deg = max(-self.max_steer, self.steer_deg - self.steer_step)
        elif k == 'c':
            self.steer_deg = 0.0
        elif k == ' ':
            self.v = 0.0
            self.steer_deg = 0.0
        elif k == 'q':
            return False
        return True

    # ------------------------------------------------------------------
    def tick(self):
        # Inverse bicycle model: steering angle -> yaw rate at the current speed.
        # At v = 0 the yaw rate is 0 no matter how the wheels are turned, which
        # is physically correct: a stationary car has no yaw rate.
        w = self.v * math.tan(math.radians(self.steer_deg)) / self.L

        msg = Twist()
        msg.linear.x = float(self.v)
        msg.angular.z = float(w)
        self.pub.publish(msg)
        # \r keeps it on one line without scrolling the terminal
        print(f'\r v={self.v:+.2f} m/s   steer={self.steer_deg:+5.1f} deg   '
              f'(w={w:+.2f} rad/s)      ', end='', flush=True)


def read_key(timeout=0.05):
    """Non-blocking single-character read. Returns '' if nothing was typed."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return ''


def main():
    rclpy.init()
    node = KeyTeleop()

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
        # Send a zero command on the way out so the car does not coast away.
        stop = Twist()
        node.pub.publish(stop)
        print('\nstopped')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
