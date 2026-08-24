#!/usr/bin/env python3
"""
twist_mux.py — the command ARBITER for the M6 stack (Session 17b rewrite).

First version inferred "is the human driving?" from the Twist values on /cmd_vel_manual.
On the car that proved wrong: at zero speed a steering key produces (0,0), indistinguish-
able from idle, so a/d didn't override, a cold brake didn't stop, and handback forced a
2 s halt. The fix is to stop guessing: override_teleop now sends the operator's INTENT
explicitly on /teleop/override, and this node obeys it.

    override_teleop ─/teleop/override─┐
                                      ├─►[ twist_mux ]─/cmd_vel─►[ drive_node ]─► wheels
    autopilot_node  ─/cmd_vel_auto───┘

THREE MODES (the operator picks; the mux executes):
  GUARDED  autopilot drives. If the operator is actively steering (a/d), the mux keeps
           the AUTOPILOT'S SPEED but replaces the yaw with the operator's steering angle
           (a "nudge"); otherwise it passes the autopilot's command through untouched.
           This is shared / guarded control — steering and throttle arbitrated on
           separate axes.
  MANUAL   the operator drives everything; the autopilot is ignored.
  STOP     latched full stop, regardless of the autopilot, until the operator re-engages.

WHY the bicycle model lives here now: the operator sends a steering ANGLE, not a yaw.
The mux turns angle -> yaw (w = human_steer_sign * v * tan(angle) / L) at whichever speed
applies — the autopilot's in GUARDED, the operator's in MANUAL. That is what lets a nudge
inherit the autopilot's speed. human_steer_sign = -1.0 is the OPERATOR convention (the
S16 teleop sign); the autopilot's own command already carries its +1.0 convention and is
passed through unchanged, so both sources reach drive_node correct end-to-end.

WATCHDOG: publishes at a fixed rate. In GUARDED with the autopilot stale, or with nothing
fresh, it publishes zeros so drive_node stops. Layered atop drive_node's own watchdog.

THE SOCKET FOR LATER: the LiDAR safety monitor becomes a top-priority clamp here that can
force zeros in any mode — "camera proposes, LiDAR vetoes."

Topics
------
subscribes : /teleop/override   std_msgs/Float32MultiArray  [mode, steer_deg, steer_active, speed_mps]
             /cmd_vel_auto       geometry_msgs/Twist
publishes  : /cmd_vel                   geometry_msgs/Twist  (to drive_node)
             /twist_mux/active_source   std_msgs/String       (AUTO|GUARD|MANUAL|STOP|NONE)

Run:
    python3 twist_mux.py
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, String

GUARDED, MANUAL, STOP = 0, 1, 2


class TwistMux(Node):

    def __init__(self):
        super().__init__('twist_mux')

        self.declare_parameter('override_timeout_s', 0.5)  # /teleop/override "fresh" within this
        self.declare_parameter('auto_timeout_s', 0.5)      # /cmd_vel_auto "fresh" within this
        self.declare_parameter('wheelbase_m', 0.094)       # must match drive_node
        self.declare_parameter('human_steer_sign', -1.0)   # OPERATOR convention (S16 teleop sign)
        self.declare_parameter('max_steer_deg', 30.0)
        self.declare_parameter('rate_hz', 30.0)

        g = self.get_parameter
        self.override_timeout = float(g('override_timeout_s').value)
        self.auto_timeout = float(g('auto_timeout_s').value)
        self.L = float(g('wheelbase_m').value)
        self.hsign = float(g('human_steer_sign').value)
        self.max_steer = float(g('max_steer_deg').value)
        rate = float(g('rate_hz').value)

        self.ovr = None            # last [mode, steer_deg, active, speed]
        self.auto = None           # last /cmd_vel_auto Twist
        self.t_ovr = None
        self.t_auto = None
        self.last_source = None

        self.create_subscription(
            Float32MultiArray, 'teleop/override', self.on_override, 10)
        self.create_subscription(Twist, 'cmd_vel_auto', self.on_auto, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.src_pub = self.create_publisher(String, 'twist_mux/active_source', 10)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            'twist_mux up | GUARDED/MANUAL/STOP from /teleop/override | '
            'human_steer_sign=%+.0f | publishing /cmd_vel' % self.hsign)

    # ------------------------------------------------------------------
    def on_override(self, msg: Float32MultiArray):
        if len(msg.data) >= 4:
            self.ovr = list(msg.data[:4])
            self.t_ovr = self.get_clock().now()

    def on_auto(self, msg: Twist):
        self.auto = msg
        self.t_auto = self.get_clock().now()

    def _fresh(self, t_last, timeout):
        if t_last is None:
            return False
        return (self.get_clock().now() - t_last).nanoseconds / 1e9 <= timeout

    # ------------------------------------------------------------------
    def _yaw_from_angle(self, v, steer_deg):
        """Operator steering ANGLE -> yaw rate at speed v (bicycle model + operator sign)."""
        s = max(-self.max_steer, min(self.max_steer, steer_deg))
        return self.hsign * v * math.tan(math.radians(s)) / self.L

    def _twist(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        return m

    def tick(self):
        ovr_fresh = self._fresh(self.t_ovr, self.override_timeout)
        auto_fresh = self._fresh(self.t_auto, self.auto_timeout)

        # No operator console -> behave as GUARDED with no nudge (autopilot is the baseline).
        if ovr_fresh and self.ovr is not None:
            mode = int(round(self.ovr[0]))
            steer_deg = self.ovr[1]
            steer_active = self.ovr[2] >= 0.5
            speed = self.ovr[3]
        else:
            mode, steer_deg, steer_active, speed = GUARDED, 0.0, False, 0.0

        if mode == STOP:
            out, source = self._twist(0.0, 0.0), 'STOP'

        elif mode == MANUAL:
            out = self._twist(speed, self._yaw_from_angle(speed, steer_deg))
            source = 'MANUAL'

        else:  # GUARDED
            if not auto_fresh:
                # autopilot gone -> we have no speed to guard; stop.
                out, source = self._twist(0.0, 0.0), 'NONE'
            else:
                v = self.auto.linear.x
                if steer_active:
                    # nudge: keep the autopilot's speed, use the operator's steering.
                    out = self._twist(v, self._yaw_from_angle(v, steer_deg))
                    source = 'GUARD'
                else:
                    # pass the autopilot through untouched (its sign is already correct).
                    out = self._twist(v, self.auto.angular.z)
                    source = 'AUTO'

        self.pub.publish(out)

        s = String()
        s.data = source
        self.src_pub.publish(s)
        if source != self.last_source:
            self.get_logger().info('source -> %s' % source)
            self.last_source = source


def main():
    rclpy.init()
    node = TwistMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
