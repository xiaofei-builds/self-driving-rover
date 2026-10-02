#!/usr/bin/env python3
r"""
odom_node.py -- dead-reckoning command odometry for the PiCar-X (Phase 5, N1).

WHAT THIS NODE DOES
-------------------
Publishes the robot's running pose estimate -- the odom -> base_link transform --
by INTEGRATING the commanded velocity on /cmd_vel over time. This is the
"smooth but drifting" half of the localization stack (REP-105):

    map -> odom -> base_link -> base_laser
     \____SLAM___/\__THIS NODE_/\__N0 URDF__/

- map -> odom       : published later by SLAM/AMCL; accurate, but JUMPS on correction.
- odom -> base_link : THIS node; continuous and smooth, but DRIFTS without bound.
- base_link->base_laser : the fixed LiDAR mount (N0, robot_state_publisher).

WHY DEAD-RECKONING (and why it will drift)
------------------------------------------
This rover has NO wheel encoders, so we cannot measure how far the wheels
actually turned. The best signal we have is the command we SENT the wheels:
twist_mux already publishes a fully-formed (v, w) on /cmd_vel --
    linear.x  = commanded forward speed (m/s)
    angular.z = commanded yaw rate     (rad/s)  [bicycle model done upstream]
so we integrate that. It drifts because commanded != actual: the throttle is
open-loop (no speed feedback), the servo lags, tyres slip, and the min-speed
floor and 30-deg steering clamp mean the car often does less than we asked.
Seeing that drift is the whole point of N1 -- it motivates the LiDAR scan-
matching and IMU we fuse in later sessions (robot_localization EKF).

THE SIGN COIN-FLIP (read before trusting the map -- the Session 18 lesson)
-------------------------------------------------------------------------
angular.z on /cmd_vel carries whatever sign makes the CAR steer correctly end-
to-end (twist_mux human_steer_sign * drive_node steer_sign, both physically
calibrated). That is NOT guaranteed to match the ROS convention (REP-103:
+z = CCW = LEFT). So 'yaw_sign' below is an UNVERIFIED assumption until we watch
it: command a LEFT turn and confirm odom yaw INCREASES. If it decreases,
relaunch with the opposite yaw_sign, re-check, and bake it -- same discipline as
the steering-sign calibration.

INTEGRATION
-----------
A fixed-rate timer holds the last (v, w) between messages (mirrors drive_node)
and uses the EXACT ARC (constant-curvature) update, not a straight-line step, so
a curve does not accumulate a systematic inward/outward bias:
    theta' = theta + w*dt
    if |w| ~ 0:  x += v*cos(theta)*dt ; y += v*sin(theta)*dt
    else:        x += (v/w)*(sin(theta') - sin(theta))
                 y += (v/w)*(cos(theta)  - cos(theta'))
When /cmd_vel goes stale (drive_node's dead-man has already stopped the car), we
integrate zeros -- no phantom motion.

Topics
------
subscribes : /cmd_vel   geometry_msgs/Twist
publishes  : /odom       nav_msgs/Odometry               (pose+twist, odom->base_link)
             /tf          geometry_msgs/TransformStamped  (odom -> base_link)

Run:
    ros2 run picarx_ros odom_node
    # watch the heading while you turn, to verify yaw_sign:
    ros2 topic echo /odom --field pose.pose.orientation
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


def yaw_to_quat(theta: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(theta / 2.0)
    q.w = math.cos(theta / 2.0)
    return q


class OdomNode(Node):

    def __init__(self):
        super().__init__('odom_node')

        # ---- parameters (CONFIG, not code) --------------------------------
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('cmd_timeout_s', 0.5)   # match drive_node dead-man
        # /cmd_vel angular.z is SIGN-INVERTED vs ROS (+z=CCW=LEFT): a left-turn
        # command carries negative angular.z on this car (twist_mux hsign=-1 x
        # drive steer_sign=+1). CONFIRMED S26 by watching odom yaw rise on a left
        # turn only with -1.0. Same class of fact as the S18 steering-sign calib.
        self.declare_parameter('yaw_sign', -1.0)
        self.declare_parameter('publish_tf', True)
        # speed calibration (measured S27, 2.007 m run): actual ground speed
        # is AFFINE in the commanded speed, not equal to it:
        #   actual = speed_offset + speed_gain*|cmd|   (offset 0.148 m/s, gain 0.73)
        # A single gain can't fix the offset. We rescale BOTH v and w by
        # k = actual/cmd so the turning radius (v/w, fixed by steering angle) is
        # preserved. Set speed_offset=0, speed_gain=1 for RAW (uncalibrated) odom.
        self.declare_parameter('speed_offset', 0.148)
        self.declare_parameter('speed_gain', 0.734)
        self.declare_parameter('min_cmd_speed', 0.01)   # below this cmd -> car stopped

        g = self.get_parameter
        self.odom_frame = g('odom_frame').value
        self.base_frame = g('base_frame').value
        rate = float(g('rate_hz').value)
        self.timeout = float(g('cmd_timeout_s').value)
        self.yaw_sign = float(g('yaw_sign').value)
        self.publish_tf = bool(g('publish_tf').value)
        self.speed_offset = float(g('speed_offset').value)
        self.speed_gain = float(g('speed_gain').value)
        self.min_cmd_speed = float(g('min_cmd_speed').value)

        # ---- pose state ----------------------------------------------------
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v_cmd = 0.0      # last commanded linear.x (m/s)
        self.w_cmd = 0.0      # last commanded yaw rate, yaw_sign applied (rad/s)
        self.last_cmd_t = None
        self.last_tick = self.get_clock().now()

        # ---- ROS interface -------------------------------------------------
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_bc = TransformBroadcaster(self)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            f'odom_node up | {self.odom_frame} -> {self.base_frame} | '
            f'yaw_sign={self.yaw_sign:+.0f} (VERIFY on a left turn) | '
            f'tf={"on" if self.publish_tf else "off"}')

    # -------------------------------------------------------------------------
    def on_cmd_vel(self, msg: Twist):
        self.v_cmd = msg.linear.x
        self.w_cmd = self.yaw_sign * msg.angular.z
        self.last_cmd_t = self.get_clock().now()

    # -------------------------------------------------------------------------
    def tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_tick).nanoseconds / 1e9
        self.last_tick = now
        if dt <= 0.0:
            return

        # Stale command -> drive_node's dead-man has already stopped the car.
        # Integrate zeros rather than coasting the estimate.
        if self.last_cmd_t is None or \
           (now - self.last_cmd_t).nanoseconds / 1e9 > self.timeout:
            v, w = 0.0, 0.0
        else:
            vc = self.v_cmd
            if abs(vc) < self.min_cmd_speed:
                v, w = 0.0, 0.0
            else:
                # map commanded -> actual ground speed (affine), keep sign
                v = math.copysign(self.speed_offset + self.speed_gain * abs(vc), vc)
                k = v / vc                 # actual/commanded ratio
                w = self.w_cmd * k         # scale yaw the same -> radius v/w unchanged

        # exact constant-curvature (arc) integration
        if abs(w) < 1e-6:
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        else:
            th2 = self.theta + w * dt
            r = v / w
            self.x += r * (math.sin(th2) - math.sin(self.theta))
            self.y += r * (math.cos(self.theta) - math.cos(th2))
            self.theta = th2
        # normalise theta to (-pi, pi]
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        stamp = now.to_msg()

        # ---- /odom ---------------------------------------------------------
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = yaw_to_quat(self.theta)
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        # Covariance: pose grows without bound (dead-reckoning), so mark it
        # loosely; the velocities are what a later EKF should actually trust.
        # 6x6 row-major, diagonal = [x, y, z, roll, pitch, yaw].
        odom.pose.covariance[0] = 0.05     # x
        odom.pose.covariance[7] = 0.05     # y
        odom.pose.covariance[35] = 0.10    # yaw
        odom.twist.covariance[0] = 0.02    # vx
        odom.twist.covariance[35] = 0.05   # vyaw
        self.odom_pub.publish(odom)

        # ---- TF odom -> base_link -----------------------------------------
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation = yaw_to_quat(self.theta)
            self.tf_bc.sendTransform(t)


def main():
    rclpy.init()
    node = OdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
