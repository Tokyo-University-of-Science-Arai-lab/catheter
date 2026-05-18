#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import socket
import threading

TCP_PORT = 5025
BUF_SIZE = 1024


class WallDistanceReceiver(Node):
    def __init__(self):
        super().__init__('wall_distance_receiver')

        # 🔹 ROS publisher
        self.pub = self.create_publisher(
            Float32,
            '/wall_distance',
            10
        )

        # 🔹 TCP server
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(('', TCP_PORT))
        self.server.listen(1)

        self.get_logger().info(
            f"WallDistanceReceiver listening on TCP port {TCP_PORT}"
        )

        thread = threading.Thread(
            target=self.accept_loop,
            daemon=True
        )
        thread.start()

    def accept_loop(self):
        while True:
            conn, addr = self.server.accept()
            self.get_logger().info(f"TCP connection from {addr}")
            self.handle_client(conn, addr)

    def handle_client(self, conn, addr):
        with conn:
            buffer = ""

            while rclpy.ok():
                data = conn.recv(BUF_SIZE)
                if not data:
                    self.get_logger().info(f"TCP disconnected {addr}")
                    break

                buffer += data.decode('utf-8')

                # 🔹 改行区切りで処理
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    text = line.strip()

                    try:
                        value = float(text)
                    except ValueError:
                        self.get_logger().warn(
                            f"Invalid float '{text}' from {addr}"
                        )
                        continue

                    msg = Float32()
                    msg.data = value
                    self.pub.publish(msg)

                    #self.get_logger().info(
                    #    f"Received TCP '{value}' → publish /wall_distance"
                    #)


def main():
    rclpy.init()
    node = WallDistanceReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()