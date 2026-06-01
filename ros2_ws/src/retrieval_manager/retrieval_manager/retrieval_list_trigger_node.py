#!/usr/bin/env python3
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


class RetrievalListTriggerNode(Node):
    def __init__(self):
        super().__init__("retrieval_list_trigger_node")

        self.declare_parameter(
            "master_path",
            "/home/book/pro_book/pro_hand_book_python/master_20260216.json"
        )
        self.declare_parameter("initial_wait_sec", 2.0)
        self.declare_parameter("after_shelf_id_wait_sec", 0.5)
        self.declare_parameter("after_navigation_goal_wait_sec", 0.5)

        self.master_path = Path(
            self.get_parameter("master_path").value
        )
        self.initial_wait_sec = float(
            self.get_parameter("initial_wait_sec").value
        )
        self.after_shelf_id_wait_sec = float(
            self.get_parameter("after_shelf_id_wait_sec").value
        )
        self.after_navigation_goal_wait_sec = float(
            self.get_parameter("after_navigation_goal_wait_sec").value
        )

        with open(self.master_path, "r", encoding="utf-8") as f:
            self.books = json.load(f)

        self.shelf_id_pub = self.create_publisher(String, "/shelf_id", 10)
        self.navigation_goal_pub = self.create_publisher(Bool, "/navigation_goal", 10)
        self.navigation_goal_final_pub = self.create_publisher(Bool, "/navigation_goal_final", 10)

        self.retrieval_done_sub = self.create_subscription(
            Bool,
            "/retrieval_done",
            self.retrieval_done_callback,
            10
        )

        self.index = 0
        self.waiting_done = False
        self.all_done_logged = False

        self.start_time = time.time()
        self.timer = self.create_timer(0.5, self.timer_callback)

        self.get_logger().info(
            f"Loaded {len(self.books)} books from {self.master_path}"
        )

    def retrieval_done_callback(self, msg: Bool):
        if not msg.data:
            return

        if not self.waiting_done:
            self.get_logger().warn(
                "Received /retrieval_done, but node was not waiting. Ignored."
            )
            return

        book = self.books[self.index]

        self.index += 1
        self.waiting_done = False

    def publish_bool(self, publisher, value: bool):
        msg = Bool()
        msg.data = value
        publisher.publish(msg)

    def publish_current_book_request(self):
        book = self.books[self.index]

        book_name = book.get("book_name", "")
        shelf_id = book.get("bookshelf_ID", "")

        if not shelf_id:
            self.get_logger().error(
                f"bookshelf_ID is empty. index={self.index}, book_name={[book_name]}"
            )
            self.index += 1
            return

        self.get_logger().info(
            f"Start retrieval request: {self.index + 1}/{len(self.books)} "
        )

        self.get_logger().info(
            f"book_name={[book_name]}"
        )
        
        shelf_msg = String()
        shelf_msg.data = shelf_id
        self.shelf_id_pub.publish(shelf_msg)
        self.get_logger().info(f"Published /shelf_id: {shelf_id}")

        time.sleep(self.after_shelf_id_wait_sec)
        self.publish_bool(self.navigation_goal_pub, True)
        self.get_logger().info("Published /navigation_goal: true")

        time.sleep(self.after_navigation_goal_wait_sec)
        self.publish_bool(self.navigation_goal_final_pub, True)
        self.get_logger().info("Published /navigation_goal_final: true")

        self.waiting_done = True

    def timer_callback(self):
        if time.time() - self.start_time < self.initial_wait_sec:
            return

        if self.waiting_done:
            return

        if self.index >= len(self.books):
            if not self.all_done_logged:
                self.get_logger().info("All retrieval requests completed.")
                self.all_done_logged = True
            return

        self.publish_current_book_request()


def main(args=None):
    rclpy.init(args=args)
    node = RetrievalListTriggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
