#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
import socket
import threading

TCP_PORT = 5015
BUF_SIZE = 1024


class NavigationGoalReceiver(Node):
    def __init__(self):
        super().__init__('navigation_goal_final_receiver')

        # ROS publisher
        self.pub = self.create_publisher(Bool, '/navigation_goal_final', 10)

        # TCP server socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('', TCP_PORT))
        self.server.listen(1)

        self.get_logger().info(
            f"NavigationGoalReceiver listening on TCP port {TCP_PORT}"
        )

        # TCP accept thread
        thread = threading.Thread(target=self.accept_loop, daemon=True)
        thread.start()

    def accept_loop(self):
        while True:
            conn, addr = self.server.accept()
            self.get_logger().info(f"TCP connection from {addr}")
            self.handle_client(conn, addr)

    def handle_client(self, conn, addr):
        with conn:
            while rclpy.ok():
                data = conn.recv(BUF_SIZE)
                if not data:
                    self.get_logger().info(f"TCP disconnected {addr}")
                    break

                text = data.decode('utf-8').strip().lower()
                msg = Bool()

                if text in ['1', 'true', 'start', 'go']:
                    msg.data = True
                elif text in ['0', 'false', 'stop']:
                    msg.data = False
                else:
                    self.get_logger().warn(
                        f"Unknown TCP message '{text}' from {addr}"
                    )
                    continue

                self.pub.publish(msg)
                self.get_logger().info(
                    f"Received TCP '{text}' → publish /navigation_goal_final={msg.data}"
                )


def main():
    rclpy.init()
    node = NavigationGoalReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
