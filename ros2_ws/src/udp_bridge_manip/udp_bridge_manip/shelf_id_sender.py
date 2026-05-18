#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import time

AMR_IP = "172.20.10.2"
TCP_PORT = 5000
RETRY_INTERVAL = 1.0  # 秒


class ManipTcpSender(Node):
    def __init__(self):
        super().__init__('manip_tcp_sender')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.get_logger().info(
            f"Waiting for AMR TCP receiver {AMR_IP}:{TCP_PORT} ..."
        )

        # 🔴 ここが「receiver が立つまで待つ」本体
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

        # receiver が立ってから subscriber を作る
        self.sub = self.create_subscription(
            String,
            '/shelf_id',
            self.cb_shelf_id,
            10
        )

    def cb_shelf_id(self, msg: String):
        shelf_id = msg.data.strip()
        if shelf_id == "":
            return

        self.sock.sendall((shelf_id + "\n").encode('utf-8'))
        self.get_logger().info(
            f"Sent shelf_id='{shelf_id}'"
        )


def main():
    rclpy.init()
    node = ManipTcpSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
