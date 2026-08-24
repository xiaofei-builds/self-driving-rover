#!/usr/bin/env python3
"""
camera_node.py — ROS 2 camera driver for the PiCar-X.

Milestone M4 of the Phase 4A ROS 2 migration. This node is the SENSOR half of
what the old autopilot script did in one process: it grabs frames from the
camera and publishes them on the graph. Nothing else. It does not resize for a
model, does not detect tape, does not steer.

Publishes
---------
  /camera/image_raw              sensor_msgs/Image            (bgr8)
  /camera/image_raw/compressed   sensor_msgs/CompressedImage  (jpeg)

Both are published with SENSOR_DATA QoS (BEST_EFFORT). See the long note in
the QoS section below -- this is the #1 source of "why does rviz show nothing".

Run (on the Pi):
    python3 camera_node.py

Publish smaller / slower (weak WiFi):
    python3 camera_node.py --ros-args -p width:=240 -p height:=180 -p rate_hz:=10

Raw off, compressed only (saves CPU + bandwidth when only viewing remotely):
    python3 camera_node.py --ros-args -p publish_raw:=false

Check it from another terminal:
    ros2 topic list
    ros2 topic hz /camera/image_raw
    ros2 topic bw /camera/image_raw
    ros2 topic bw /camera/image_raw/compressed
    ros2 topic echo /camera/image_raw --field header      # don't echo the pixels!

Session 15 of the self-driving-rover project.
"""

import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CompressedImage


class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        # ---- parameters -----------------------------------------------------
        # Publish size is CONFIG. The camera captures at whatever Vilib sets up
        # (640x480); we downscale before publishing because every byte here is
        # paid for at 15-30 times a second, forever.
        self.declare_parameter('width', 320)
        self.declare_parameter('height', 240)
        self.declare_parameter('rate_hz', 15.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('publish_raw', True)
        self.declare_parameter('publish_compressed', True)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('vflip', False)
        self.declare_parameter('hflip', False)

        g = self.get_parameter
        self.W = int(g('width').value)
        self.H = int(g('height').value)
        self.rate = float(g('rate_hz').value)
        self.jpeg_q = int(g('jpeg_quality').value)
        self.do_raw = bool(g('publish_raw').value)
        self.do_jpeg = bool(g('publish_compressed').value)
        self.frame_id = str(g('frame_id').value)

        # ---- hardware -------------------------------------------------------
        # Imported here, not at module top, so this file can be opened and
        # syntax-checked on a laptop that has no camera. Same pattern as
        # autopilot_recovery.py.
        #
        # NOTE ON OWNERSHIP: this node does NOT create Picarx(). The camera
        # TILT servo lives on the same I2C bus that drive_node owns, and two
        # processes talking to one bus is how you get random servo jitter.
        # Tilt is drive_node's job now (see the cam_tilt_deg patch in the
        # Session 15 brief). This node owns the image sensor, nothing else.
        import cv2
        from vilib import Vilib
        self.cv2 = cv2
        self.Vilib = Vilib

        Vilib.camera_start(vflip=bool(g('vflip').value),
                           hflip=bool(g('hflip').value))
        # display(web=False): the old script streamed a web preview, which cost
        # CPU and bandwidth we now want for the ROS topics. rviz/Foxglove is
        # the viewer from here on.
        Vilib.display(local=False, web=False)
        time.sleep(2.0)          # let the capture thread produce a first frame

        # ---- QoS ------------------------------------------------------------
        # qos_profile_sensor_data = BEST_EFFORT + KEEP_LAST(5).
        #
        # Why best-effort for images: a retransmitted frame is a picture of the
        # past. If a frame is lost, the right answer is "wait 66 ms for the next
        # one", not "make the network resend the stale one and stall the queue
        # behind it". /cmd_vel is the opposite -- a dropped stop command is
        # unacceptable, so that one stays RELIABLE.
        #
        # THE GOTCHA: a BEST_EFFORT publisher will NOT connect to a RELIABLE
        # subscriber. No error, no warning -- the topic just sits at 0 Hz.
        # If a tool shows nothing, check QoS before you check your code:
        #     ros2 topic info /camera/image_raw --verbose
        qos = qos_profile_sensor_data

        self.pub_raw = self.create_publisher(Image, 'camera/image_raw', qos) \
            if self.do_raw else None
        self.pub_jpeg = self.create_publisher(
            CompressedImage, 'camera/image_raw/compressed', qos) \
            if self.do_jpeg else None

        # ---- diagnostics ----------------------------------------------------
        self.n_pub = 0
        self.n_stale = 0
        self.bytes_raw = 0
        self.bytes_jpeg = 0
        self.last_report = time.time()
        self.last_frame_sig = None

        self.create_timer(1.0 / self.rate, self.tick)
        self.create_timer(5.0, self.report)

        self.get_logger().info(
            f'camera_node up | {self.W}x{self.H} @ {self.rate:.0f}Hz  '
            f'raw={self.do_raw} jpeg={self.do_jpeg}(q{self.jpeg_q})  '
            f'frame_id={self.frame_id}')

    # ------------------------------------------------------------------------
    def tick(self):
        frame = self.Vilib.img
        if frame is None or getattr(frame, 'size', 0) == 0:
            return

        # Vilib.img is a shared buffer refreshed by the capture thread. If we
        # publish faster than the camera captures, we would republish the same
        # picture with a NEW timestamp -- which is a lie, and it poisons any
        # downstream rate or latency measurement. Cheap duplicate check: a few
        # sampled pixels + the mean. Not cryptographic, just enough.
        sig = (float(frame[::40, ::40].mean()), int(frame[0, 0, 0]),
               int(frame[-1, -1, -1]))
        if sig == self.last_frame_sig:
            self.n_stale += 1
            return
        self.last_frame_sig = sig

        # Capture time. Strictly this is "time we noticed the frame", which is
        # a few ms after the shutter. Real drivers stamp with the sensor's own
        # capture time, because in sensor fusion a 20 ms stamping error at
        # 0.3 m/s is 6 mm of position error -- and on a real car at 30 m/s it
        # is 60 cm. Timestamp accuracy IS a perception spec.
        stamp = self.get_clock().now().to_msg()

        small = self.cv2.resize(frame, (self.W, self.H))

        if self.pub_raw is not None:
            msg = Image()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.height = self.H
            msg.width = self.W
            msg.encoding = 'bgr8'          # OpenCV's native order. NOT rgb8.
            msg.is_bigendian = 0
            msg.step = self.W * 3          # bytes per row
            msg.data = np.ascontiguousarray(small).tobytes()
            self.pub_raw.publish(msg)
            self.bytes_raw += len(msg.data)

        if self.pub_jpeg is not None:
            ok, enc = self.cv2.imencode(
                '.jpg', small,
                [int(self.cv2.IMWRITE_JPEG_QUALITY), self.jpeg_q])
            if ok:
                cmsg = CompressedImage()
                cmsg.header.stamp = stamp
                cmsg.header.frame_id = self.frame_id
                cmsg.format = 'jpeg'
                cmsg.data = enc.tobytes()
                self.pub_jpeg.publish(cmsg)
                self.bytes_jpeg += len(cmsg.data)

        self.n_pub += 1

    # ------------------------------------------------------------------------
    def report(self):
        """Measure what you actually got. The requested rate is a wish; the
        Pi 3B, the camera, and the JPEG encoder all get a vote."""
        now = time.time()
        dt = now - self.last_report
        if dt <= 0:
            return
        hz = self.n_pub / dt
        kbs_raw = self.bytes_raw / dt / 1024.0
        kbs_jpeg = self.bytes_jpeg / dt / 1024.0
        self.get_logger().info(
            f'{hz:5.1f} Hz published | raw {kbs_raw:7.1f} KB/s | '
            f'jpeg {kbs_jpeg:6.1f} KB/s | {self.n_stale} duplicate frames skipped')
        self.n_pub = self.n_stale = 0
        self.bytes_raw = self.bytes_jpeg = 0
        self.last_report = now

    # ------------------------------------------------------------------------
    def shutdown(self):
        try:
            self.Vilib.camera_close()
        except Exception:
            pass


def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
