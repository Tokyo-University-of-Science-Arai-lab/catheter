#!/usr/bin/env python3
# ==========================
# xarm7 モジュールのパスを明示的に追加（最重要）
# ==========================
import sys
sys.path.append("/home/book/pro_book/pro_hand_book_python")

# ==========================
# 通常import
# ==========================
import rclpy
from rclpy.node import Node
import yaml
import math
import time
import os

from xarm7.control.xarm7 import XArm7


# ==========================
# 設定
# ==========================
XARM_HOST = "192.168.2.197"

BASE_DIR = os.path.expanduser(
    "~/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config"
)

DEFAULT_SPEED = 0.5     # rad/s
DEFAULT_ACCEL = 1.0     # rad/s^2


def deg2rad_list(deg_list):
    return [math.radians(d) for d in deg_list]


class WaypointPlayer(Node):
    def __init__(self, yaml_path: str):
        super().__init__("waypoint_player")

        self.yaml_path = yaml_path

        self.get_logger().info("Connecting to xArm...")
        self.arm = XArm7(self, host=XARM_HOST)

        time.sleep(0.3)  # SDK ready wait

        self.play()

    # ------------------------
    # 再生本体
    # ------------------------
    def play(self):
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        waypoints = data["waypoints"]

        self.get_logger().info(
            f"Loaded {len(waypoints)} waypoints from:\n  {self.yaml_path}"
        )

        for wp in waypoints:
            name = wp["name"]
            q_rad = deg2rad_list(wp["q"])

            self.get_logger().info(f"Move to {name}")

            ret = self.arm.arm.set_servo_angle(
                angle=q_rad,
                speed=DEFAULT_SPEED,
                mvacc=DEFAULT_ACCEL,
                is_radian=True,
                wait=False
            )

            if ret == 0:
                time.sleep(1.0)
                continue

            elif ret == 3:
                self.get_logger().warn(
                    f"Motion warning at {name}, code=3 (continue async)"
                )
                time.sleep(1.2)  # 少し長め
                continue

            else:
                self.get_logger().error(
                    f"Fatal motion error at {name}, code={ret}"
                )
                break


    # ------------------------
    # 終了処理
    # ------------------------
    def shutdown(self):
        try:
            self.arm.disconnect()
        except Exception:
            pass


# ==========================
# YAML選択
# ==========================
def select_yaml():
    if not os.path.isdir(BASE_DIR):
        print(f"Config directory not found:\n  {BASE_DIR}")
        sys.exit(1)

    yamls = sorted(
        [f for f in os.listdir(BASE_DIR) if f.endswith(".yaml")]
    )

    if not yamls:
        print("No YAML files found in config/")
        sys.exit(1)

    print("\nAvailable YAML files:")
    for f in yamls:
        print(f"  - {f}")

    name = input("\nEnter YAML filename to play (without .yaml): ").strip()
    if not name:
        print("No filename entered.")
        sys.exit(1)

    if not name.endswith(".yaml"):
        name += ".yaml"

    path = os.path.join(BASE_DIR, name)

    if not os.path.exists(path):
        print(f"File not found:\n  {path}")
        sys.exit(1)

    return path


# ==========================
# main
# ==========================
def main():
    # --- ROS初期化前にYAML選択 ---
    yaml_path = select_yaml()

    rclpy.init()
    node = None

    try:
        node = WaypointPlayer(yaml_path)

    except KeyboardInterrupt:
        pass

    finally:
        if node:
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
