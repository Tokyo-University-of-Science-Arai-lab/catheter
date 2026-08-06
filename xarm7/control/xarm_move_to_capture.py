#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xArm7 を撮影位置（capture pose）まで移動させるスクリプト。

流れ:
  1. xArm 接続 + XArmMonitor 起動（異常検知で緊急停止）
  2. shelf_id から side（left/right）と TCP Z オフセットを決定
  3. WaypointPlayerNode.trigger_goal() で撮影姿勢へ移動
     （同時に昇降機構の目標高さを /target_mm に publish）
  4. 撮影姿勢到達後、棚段ごとの TCP Z オフセット(mm)を moveL で適用

設定は Retrieval_integration.yaml を共有する（host, shelf_id, waypoint パス）。
shelf_id を変えたいだけならこのファイルの SHELF_ID を上書きすればよい
（None のままなら yaml の shelf_id を使う）。
"""
import sys
sys.path.append("/home/book/pro_book_SAM3/pro_hand_book_python")

import time
from pathlib import Path

import yaml
import rclpy
from rclpy.node import Node

from xarm7.control.xarm7 import XArm7
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion
from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode


# ==========================
# 設定
# ==========================
SCRIPT_DIR = Path(__file__).resolve().parents[2]  # pro_hand_book_python/
CONFIG_PATH = SCRIPT_DIR / "Retrieval_integration.yaml"

# None にすると yaml の shelf_id を使う。上書きしたい場合はここに文字列を入れる。
# 書式: "棚番号-列番号-段番号-位置番号" 例: "2-5-1-1"
SHELF_ID: str | None = None

WAYPOINT_SPEED = 1.0
WAYPOINT_ACCEL = 1.0


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_dir"] = str(config_path.parent)
    return config


def resolve_config_relative_path(config: dict, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def main():
    config = load_config(CONFIG_PATH)
    xarm_host = config["robot"]["xarm"]["host"]
    shelf_id = SHELF_ID or config.get("shelf_id")
    if not shelf_id:
        raise RuntimeError(
            "shelf_id が指定されていません。"
            "SHELF_ID 定数か Retrieval_integration.yaml の shelf_id を設定してください。"
        )
    waypoint_path = resolve_config_relative_path(
        config, config["paths"]["waypoint"]["init_to_capture"]
    )

    rclpy.init()
    node = Node("xarm_move_to_capture")

    arm = None
    monitor = None
    waypoint_node = None

    try:
        node.get_logger().info(f"Connecting to xArm at {xarm_host} ...")
        arm = XArm7(node=node, host=xarm_host)
        time.sleep(1.5)  # enable 安定待ち

        monitor = XArmMonitor(arm)

        waypoint_node = WaypointPlayerNode(
            node_name="xarm_move_to_capture_waypoint",
            arm=arm,
            monitor=monitor,
            yaml_path=waypoint_path,
            speed=WAYPOINT_SPEED,
            accel=WAYPOINT_ACCEL,
        )

        # -------------------------
        # shelf_id → side / height / tcp_z_offset を決定
        # -------------------------
        node.get_logger().info(f"Using shelf_id: {shelf_id}")
        waypoint_node.shelf_manager.set_from_string(str(shelf_id))

        if not waypoint_node.shelf_manager.is_received():
            raise RuntimeError(f"Invalid shelf_id: {shelf_id}")

        side = waypoint_node.shelf_manager.get_side()
        tcp_offset = waypoint_node.shelf_manager.get_tcp_z_offset()
        node.get_logger().info(
            f"side={side}, tcp_z_offset={tcp_offset}mm"
        )

        # -------------------------
        # 撮影姿勢へ移動（昇降機構の目標高さも同時に publish）
        # -------------------------
        node.get_logger().info("Moving to capture pose...")
        waypoint_node.trigger_goal()

        while rclpy.ok() and not waypoint_node.is_finished():
            rclpy.spin_once(node, timeout_sec=0.1)
            rclpy.spin_once(waypoint_node, timeout_sec=0.0)

        if waypoint_node.is_failed():
            raise RuntimeError(
                f"Waypoint failed: {waypoint_node.error_message()}"
            )

        node.get_logger().info("Capture pose reached.")

        # -------------------------
        # Z方向オフセット適用（棚段ごとのTCP微調整）
        # -------------------------
        if tcp_offset:
            node.get_logger().info(f"Applying TCP Z offset: {tcp_offset}mm")
            safe_motion(
                lambda: arm.moveL_tcp_z_offset(tcp_offset),
                monitor,
                "tcp_z_offset",
            )
        else:
            node.get_logger().info("TCP Z offset is 0, skip.")

        node.get_logger().info("Done: xArm is at capture position (with Z offset applied).")

    except KeyboardInterrupt:
        pass

    finally:
        if monitor:
            monitor.stop()
        if arm:
            try:
                arm.disconnect()
            except Exception:
                pass
        if waypoint_node:
            waypoint_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
