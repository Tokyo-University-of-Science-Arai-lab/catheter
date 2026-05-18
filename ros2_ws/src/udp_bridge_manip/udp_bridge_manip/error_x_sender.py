#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import socket
import time

TARGET_IP = "172.20.10.2"   # 外部PCのIP
TCP_PORT = 5012
RETRY_INTERVAL = 1.0


class ErrorXSender(Node):
    def __init__(self):
        super().__init__('error_x_sender')

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.get_logger().info(
            f"Connecting to external PC {TARGET_IP}:{TCP_PORT}"
        )

        # Receiverが起動するまで待つ
        while rclpy.ok():
            try:
                self.sock.connect((TARGET_IP, TCP_PORT))
                break
            except (ConnectionRefusedError, OSError):
                self.get_logger().warn("Receiver not ready, retrying...")
                time.sleep(RETRY_INTERVAL)

        self.get_logger().info("Connected to external PC")

        # 🔹 ROSトピック購読
        self.sub = self.create_subscription(
            String,
            '/error_x',
            self.cb,
            10
        )

    def cb(self, msg: String):
        text = msg.data.strip()

        try:
            self.sock.sendall((text + "\n").encode('utf-8'))
            self.get_logger().info(f"Sent TCP: {text}")
        except BrokenPipeError:
            self.get_logger().error("TCP connection lost")


def main():
    rclpy.init()
    node = ErrorXSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()