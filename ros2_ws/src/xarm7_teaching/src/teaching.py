
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import yaml
import math
import threading
import sys
import termios
import tty
import os


# ==========================
# 設定
# ==========================
TARGET_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

SAVE_DIR = os.path.expanduser("~/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config" \
"")


class WaypointTeaching(Node):
    def __init__(self):
        super().__init__("waypoint_teaching")

        self.latest_joint_state = None
        self.waypoints = []
        self.wp_count = 0

        # keyboard control
        self.running = True
        self.termios_old = None

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_cb,
            10
        )

        # キーボード入力スレッド
        self.key_thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True
        )
        self.key_thread.start()

        self.get_logger().info(
            "Teaching started:\n"
            "  [Enter]  save waypoint\n"
            "  Ctrl+C   finish & save YAML"
        )

    # ------------------------
    # JointState callback
    # ------------------------
    def joint_cb(self, msg: JointState):
        self.latest_joint_state = msg

    # ------------------------
    # Save one waypoint
    # ------------------------
    def save_waypoint(self):
        if self.latest_joint_state is None:
            print("No JointState received yet")
            return

        pos_map = dict(
            zip(self.latest_joint_state.name,
                self.latest_joint_state.position)
        )

        try:
            q_rad = [pos_map[j] for j in TARGET_JOINTS]
        except KeyError:
            print("JointState incomplete")
            return

        q_deg = [round(math.degrees(q), 2) for q in q_rad]

        wp = {
            "name": f"p{self.wp_count}",
            "q": q_deg
        }
        self.wp_count += 1

        self.waypoints.append(wp)
        print(f"Saved waypoint {wp['name']}")

    # ------------------------
    # Keyboard thread
    # ------------------------
    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        self.termios_old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        try:
            while self.running:
                ch = sys.stdin.read(1)
                if ch == "\n":
                    self.save_waypoint()
        finally:
            # 必ず端末を元に戻す
            if self.termios_old is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, self.termios_old)

    # ------------------------
    # Save YAML
    # ------------------------
    def save_yaml(self):
        # keyboard停止 & termios復元
        self.running = False
        if self.key_thread.is_alive():
            self.key_thread.join(timeout=0.5)

        if not self.waypoints:
            print("No waypoints recorded. Nothing saved.")
            return

        os.makedirs(SAVE_DIR, exist_ok=True)

        # ファイル名入力（ここはROS非依存）
        filename = input("\nEnter YAML filename (without .yaml): ").strip()
        if not filename:
            filename = "waypoints"

        path = os.path.join(SAVE_DIR, f"{filename}.yaml")

        data = {
            "joint_order": TARGET_JOINTS,
            "waypoints": self.waypoints
        }

        with open(path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

        print(f"\nSaved {len(self.waypoints)} waypoints to:\n  {path}")


def main():
    rclpy.init()
    node = WaypointTeaching()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Ctrl+C はここで1回だけ捕まえる
        pass

    # ---- ROSはまだ生きている ----
    node.save_yaml()

    # ---- shutdownは1回だけ ----
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

