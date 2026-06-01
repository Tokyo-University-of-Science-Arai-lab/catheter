from xarm7.control.xarm7 import XArm7
from rs_d435i.get_book_position import GetBookSpinePosition
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook_retrieval 
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook_storage
from pathlib import Path
from detection.pro_handbook.sam_py_demo.get_book_points_revised import run_capture_and_pca
from xarm7.control.move_to_container_test import Move_to_Container
from xarm7.control.shelf_id_manager import ShelfIDManager
from detection.pro_handbook.sam_py_demo.bar_code.book_barcode import book_barcode_sequence
from detection.pro_handbook.sam_py_demo.bar_code.bookshelf_barcode import bookshelf_barcode_sequence
from xarm7.control.book_return_sequence import storage_sequence
from detection.pro_handbook.sam_py_demo.Storage import run_capture_and_pca_depth_space
from linear_lift import TargetPublisher
import rclpy
import cv2
import numpy as np
import json
from xarm7.control.robot_base_coordinate import PoseChain
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
import traceback
import time
from rclpy.executors import MultiThreadedExecutor
from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode
import signal, os
import sys
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion
import csv
from datetime import datetime
import yaml
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic_ros2_editing import (
    capture_barcode_and_x_offset,
    WallDistanceWatcher,
    BoolPulseWatcher,
    BoolLatchWatcher
)

def write_log(config, book_name, shelf_id, roll_deg, book_width, side, height, result, shot_dir, memo):

    log_file = config["paths"]["log"]["retrieval"]

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        book_name,
        shelf_id,
        roll_deg,
        book_width,
        side,
        height,
        result,
        str(shot_dir),
        memo
    ]

    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def sigint_handler(sig, frame):
    print("Ctrl+C detected → FORCE KILL")
    try:
        arm = globals().get("arm", None)
        if arm:
            arm.emergency_stop()
    except Exception:
        pass
    os._exit(1)

signal.signal(signal.SIGINT, sigint_handler)

def hard_disconnect(arm):
    print("disconnect xArm NOW")
    try:
        try:
            arm.emergency_stop()
        except Exception:
            pass

        arm.disconnect()
    except Exception as e:
        print(f"disconnect failed: {e}")
        
def main_sequence(
    config,
    book_name: str,
    barcode_number: str,
    bookshelf_ID: str,
    book_shelf_height: str,
    book_width_offset: float,
    tp: TargetPublisher,
    node,
    arm: XArm7,                     
    executor,
    waypoint_node: WaypointPlayerNode,
    monitor: XArmMonitor,
):
    
    HandMotors = None
    #  1. ハンドの初期化
    try:
        print("ハンドを初期化します...")
        HandMotors = HandBook_retrieval.init_dynamixels()
    except Exception as e:
        print(f"【重要】ハンドの初期化で通信エラーが発生しましたが、無視してテストを続行します: {e}")
        HandMotors = None # 失敗してもNoneのまま続行
    
    try:
        # --- 2. メイン動作 ---
        print(f"\n=== 収納(Storage) 移動テスト開始: {book_name} ===")
        # ... (中略) ...

        # 3. コンテナへの収納動作 (HandMotorsがNoneの場合はスキップ)
        if HandMotors is not None:
            print("3. Move_to_Container を実行します")
            safe_motion(
                lambda: Move_to_Container(book_width_offset, arm, waypoint_node, HandMotors), 
                monitor, 
                "Move_to_container"
            )
        else:
            print("3. 【警告】ハンドが未接続のため、収納動作(Move_to_Container)をスキップします")

        # 4. 確認と終了動作
        print("=== 収納動作(またはスキップ)が完了しました ===")
        input("Enterを押すと初期姿勢に戻り、リフトが降下します...")

        # ... (以降の姿勢移動処理はそのまま) ...
        # (中略)

        print("=== 収納移動テスト完了 ===")
        return 30.0 

    except Exception as e:
        print(f"!!! メイン処理で予期せぬエラーが発生しました: {e} !!!")
        traceback.print_exc()
        # 必要であれば emergency_stop を外してもよい
        return None
        
    finally:
        # 終了処理：ハンドが初期化されていた場合のみ閉じる
        if HandMotors is not None:
            try:
                print("ハンドの通信ポートをクローズします...")
                HandBook_retrieval.disable_torque(HandBook_retrieval.GRIPPER_ID)
                HandMotors.close_port()
            except Exception as e:
                print(f"ポートクローズ中のエラー: {e}")
        
def main():
    config = load_config("Retrieval_integration.yaml")
    # ==============================
    # マスターデータ読み込み
    # ==============================
    with open(config["books"]["master_file"], "r", encoding="utf-8") as f:
        books_master = json.load(f)

    retrieved_book_width_list = [0.0]

    # ==============================
    # ROS2 初期化
    # ==============================
    rclpy.init()

    # メイン制御ノード（publish / log / trigger 用）
    node = rclpy.create_node("book_retrieval_main")

    XARM_HOST = config["robot"]["xarm"]["host"]

    arm = XArm7(
        node=node,
        host=XARM_HOST,
    )
    globals()["arm"] = arm
    # ★ MultiThreadedExecutor 推奨
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    monitor = XArmMonitor(arm)
    # ==============================
    # リニアリフト
    # ==============================
    tp = TargetPublisher()
    executor.add_node(tp)

    # ==============================
    # WaypointPlayerNode（初期姿勢・撮影姿勢）
    # ==============================
    waypoint_node = WaypointPlayerNode(
        node_name="xarm_init_to_capture",
        arm=arm,
        monitor=monitor,
        yaml_path=config["paths"]["waypoint"]["init_to_capture"],
        speed=1.0,
        accel=1.0,
    )

    executor.add_node(waypoint_node)

    # ==============================
    # メインループ
    # ==============================
    try:
        for b in books_master:
            waypoint_node.reset()
            book_width_offset = sum(retrieved_book_width_list)

            retrieved_book_width = main_sequence(
                config=config,
                book_name=b["book_name"],
                barcode_number=b["ISBN_number"],
                bookshelf_ID=b["bookshelf_ID"],
                book_shelf_height=b.get("book_shelf_height", ""),
                book_width_offset=book_width_offset,
                tp=tp,
                node=node,
                arm=arm, 
                monitor=monitor,
                executor=executor,
                waypoint_node=waypoint_node,   
            )

            if retrieved_book_width is None:
                node.get_logger().error(
                    "Fatal error detected. Stop processing further books."
                )
                break

            retrieved_book_width_list.append(retrieved_book_width)

    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted by user")

    finally:
        # ==============================
        # 終了処理（順番大事）
        # ==============================
        node.get_logger().info("Shutting down nodes...")

        try:
            waypoint_node.destroy_node()
        except Exception:
            pass

        try:
            tp.destroy_node()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        rclpy.shutdown()


if __name__ == '__main__':

    main()
