#!/usr/bin/env python3
"""
drive_node.py — ROS 2 actuation gateway for the PiCar-X.

The ONLY process allowed to touch the hardware. Subscribes to /cmd_vel
(geometry_msgs/Twist), converts it to a front-wheel steering angle with the
kinematic bicycle model, and drives the motors.

Session 14 (Phase 4A) of the self-driving-rover project.

Run (safe, steering servo only, motors disabled):
    python3 drive_node.py

Run for real:
    python3 drive_node.py --ros-args -p enable_motors:=true

Override vehicle parameters:
    python3 drive_node.py --ros-args -p wheelbase_m:=0.095 -p steer_sign:=-1

Topics
------
subscribes : /cmd_vel            geometry_msgs/Twist
publishes  : /picarx/status      std_msgs/Float32MultiArray
                                 [steer_deg, throttle, cmd_age_s, motors_on]
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

from picarx import Picarx


class DriveNode(Node):

    def __init__(self):
        super().__init__('drive_node')

        # ---- vehicle parameters -------------------------------------------
        # These are CONFIG, not code. Change them at launch, not by editing.
        # Measured on the car, Session 14 (2026-08-07):
        #   wheelbase 94 mm  ->  min turning radius = L/tan(30deg) = 0.16 m
        #   steer_sign -1 because SunFounder uses +ve = RIGHT, while REP-103
        #   says +ve angular.z is counterclockwise = LEFT.
        self.declare_parameter('wheelbase_m', 0.094)
        self.declare_parameter('max_steer_deg', 30.0)     # matches training data
        self.declare_parameter('max_speed_mps', 0.35)     # speed that = throttle 100
        self.declare_parameter('steer_sign', -1.0)
        self.declare_parameter('cmd_timeout_s', 0.5)      # watchdog
        self.declare_parameter('min_speed_mps', 0.02)     # below this, hold steering
        self.declare_parameter('enable_motors', False)    # safe by default
        self.declare_parameter('rate_hz', 20.0)

        p = self.get_parameter
        self.L = p('wheelbase_m').value
        self.max_steer = p('max_steer_deg').value
        self.max_speed = p('max_speed_mps').value
        self.steer_sign = p('steer_sign').value
        self.timeout = p('cmd_timeout_s').value
        self.min_speed = p('min_speed_mps').value
        self.motors_on = p('enable_motors').value
        rate = p('rate_hz').value

        # ---- hardware ------------------------------------------------------
        # Instantiated ONCE. This node is the sole owner of the I2C bus.
        self.px = Picarx()
        self.px.set_dir_servo_angle(0.0)
        self.px.stop()

        # ---- state ---------------------------------------------------------
        self.v = 0.0              # last commanded linear.x  (m/s)
        self.w = 0.0              # last commanded angular.z (rad/s)
        self.last_cmd_t = None    # time of last /cmd_vel message
        self.steer_deg = 0.0      # held across zero-speed commands
        self.throttle = 0.0

        # ---- ROS interface ---------------------------------------------------
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.status_pub = self.create_publisher(
            Float32MultiArray, 'picarx/status', 10)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f'drive_node up | L={self.L:.3f}m  max_steer={self.max_steer:.0f}deg  '
            f'max_speed={self.max_speed:.2f}m/s  steer_sign={self.steer_sign:+.0f}  '
            f'motors={"ON" if self.motors_on else "OFF (safe mode)"}')
        if not self.motors_on:
            self.get_logger().warn(
                'Motors DISABLED. Steering servo will move so you can watch '
                'decisions. Re-run with -p enable_motors:=true to drive.')

    # -------------------------------------------------------------------------
    def on_cmd_vel(self, msg: Twist):
        """Store the command. Do NOT actuate here — actuation happens on the
        timer, so the watchdog runs at a fixed rate regardless of input rate."""
        self.v = msg.linear.x
        self.w = msg.angular.z
        self.last_cmd_t = self.get_clock().now()

    # -------------------------------------------------------------------------
    def bicycle_steer_deg(self, v: float, w: float) -> float:
        """Kinematic bicycle model: collapse 4 wheels to 2, assume no tyre slip.

            steer = atan( L * yaw_rate / speed )

        Undefined at v=0 (you cannot turn a parked car by asking for yaw rate),
        so below min_speed we HOLD the previous angle rather than snap to zero.
        """
        if abs(v) < self.min_speed:
            return self.steer_deg
        delta_rad = math.atan(self.L * w / v)
        delta_deg = math.degrees(delta_rad) * self.steer_sign
        return max(-self.max_steer, min(self.max_steer, delta_deg))

    # -------------------------------------------------------------------------
    def tick(self):
        """Fixed-rate control + watchdog. Runs whether or not commands arrive."""
        now = self.get_clock().now()

        if self.last_cmd_t is None:
            age = float('inf')
        else:
            age = (now - self.last_cmd_t).nanoseconds / 1e9

        if age > self.timeout:
            # DEAD-MAN'S SWITCH: no fresh command -> stop. A teleop robot that
            # keeps driving when the network drops is how you break a wall.
            if self.throttle != 0.0:
                self.get_logger().warn(f'cmd_vel stale ({age:.2f}s) — stopping')
            self.throttle = 0.0
            self.px.stop()
        else:
            self.steer_deg = self.bicycle_steer_deg(self.v, self.w)
            self.px.set_dir_servo_angle(float(self.steer_deg))

            thr = max(-100.0, min(100.0, (self.v / self.max_speed) * 100.0))
            self.throttle = thr

            if self.motors_on:
                if thr > 1.0:
                    self.px.forward(int(thr))
                elif thr < -1.0:
                    self.px.backward(int(-thr))
                else:
                    self.px.stop()

        msg = Float32MultiArray()
        msg.data = [float(self.steer_deg), float(self.throttle),
                    float(min(age, 99.0)), 1.0 if self.motors_on else 0.0]
        self.status_pub.publish(msg)

    # -------------------------------------------------------------------------
    def shutdown(self):
        try:
            self.px.stop()
            self.px.set_dir_servo_angle(0.0)
        except Exception:
            pass


def main():
    rclpy.init()
    node = DriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Motors ALWAYS cut, whatever happened. Same discipline as first_drive.py.
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
