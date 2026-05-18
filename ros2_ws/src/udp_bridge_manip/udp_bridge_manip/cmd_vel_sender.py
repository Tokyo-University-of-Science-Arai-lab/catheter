#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket
import time

AMR_IP = "172.20.10.2"
TCP_PORT = 5065
RETRY_INTERVAL = 1.0


class CmdVelTcpSender(Node):

    def __init__(self):
        super().__init__('cmd_vel_tcp_sender')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.get_logger().info(
            f"Waiting for AMR TCP receiver {AMR_IP}:{TCP_PORT} ..."
        )

        # receiver が立つまで待つ
        while rclpy.ok():
            try:
                self.sock.connect((AMR_IP, TCP_PORT))
                break
            except (ConnectionRefusedError, OSError):
                self.get_logger().warn(
                    "AMR not ready yet, retrying..."
                )
                time.sleep(RETRY_INTERVAL)

        self.get_logger().info(
            f"Connected to AMR {AMR_IP}:{TCP_PORT}"
        )

        # cmd_vel subscriber
        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cb_cmd_vel,
            10
        )

    def cb_cmd_vel(self, msg: Twist):

        # 差動ロボットなので vx と wz のみ
        vx = msg.linear.x
        wz = msg.angular.z

        data = f"{vx:.3f},{wz:.3f}\n"

        try:
            self.sock.sendall(data.encode('utf-8'))
        except BrokenPipeError:
            self.get_logger().error("TCP connection lost")


def main():
    rclpy.init()
    node = CmdVelTcpSender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()