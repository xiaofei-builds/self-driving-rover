#!/usr/bin/env python3
"""imu_node.py — BNO085 (gyro-only) -> sensor_msgs/Imu on /imu/data"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import board, busio
from adafruit_bno08x import BNO_REPORT_GYROSCOPE
from adafruit_bno08x.i2c import BNO08X_I2C


class ImuNode(Node):
    def __init__(self):
        super().__init__("imu_node")
        self.declare_parameter("imu_yaw_sign", 1.0)      # S29 measured: +1.0
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("i2c_address", 0x4A)
        self.declare_parameter("yaw_rate_variance", 4.0e-4)   # (0.02 rad/s)^2

        self.yaw_sign = float(self.get_parameter("imu_yaw_sign").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        rate = float(self.get_parameter("rate_hz").value)
        addr = int(self.get_parameter("i2c_address").value)
        self.var = float(self.get_parameter("yaw_rate_variance").value)

        i2c = busio.I2C(board.SCL, board.SDA)     # board-default clock (clock-stretch safe)
        self.bno = BNO08X_I2C(i2c, address=addr)
        self.bno.enable_feature(BNO_REPORT_GYROSCOPE)
        self.get_logger().info(
            f"BNO085 gyro-only up @0x{addr:02x}; yaw_sign={self.yaw_sign:+.0f}, "
            f"frame={self.frame_id}, rate={rate:.0f}Hz")

        self.pub = self.create_publisher(Imu, "/imu/data", 10)
        self.bad = 0
        self.create_timer(1.0 / rate, self.tick)

    def tick(self):
        try:
            gx, gy, gz = self.bno.gyro
        except Exception as e:
            self.bad += 1
            if self.bad % 50 == 1:
                self.get_logger().warn(f"gyro read failed ({self.bad} total): {e}")
            return
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = self.yaw_sign * gz
        msg.angular_velocity_covariance[0] = 1.0e-2
        msg.angular_velocity_covariance[4] = 1.0e-2
        msg.angular_velocity_covariance[8] = self.var
        msg.linear_acceleration_covariance[0] = -1.0
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
