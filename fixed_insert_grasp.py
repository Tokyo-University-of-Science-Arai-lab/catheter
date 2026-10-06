"""撮影姿勢から 挿入→把持→引き抜き までを固定動作で行う（認識なし・昇降機構なし）。

Retrieval_integration.py の取り出し動作から認識部分を外し、
アプローチ位置を「撮影姿勢TCPからの固定の相対移動」に置き換えたもの。
動作の再現性確認や教示値の確認用。設定は fixed_insert_grasp.yaml。

    python fixed_insert_grasp.py
    python fixed_insert_grasp.py --config 別の設定.yaml
"""
import argparse
import os
import signal
import time

import numpy as np
import rclpy  # type: ignore
import yaml
from rclpy.executors import MultiThreadedExecutor

import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook_retrieval
from xarm7.control.xarm7 import XArm7, INSERT_DX, RETRIEVAL_DX
from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sigint_handler(sig, frame):
    print("Ctrl+C / Emergency detected → close hand then FORCE KILL")

    hand = globals().get("HandMotors_retrieval", None)
    if hand is not None:
        try:
            HandBook_retrieval.grasp(hand)
        except Exception as e:
            print(f"[EMG] Failed to close hand: {e}")

    arm = globals().get("arm", None)
    if arm is not None:
        try:
            arm.emergency_stop()
        except Exception as e:
            print(f"[EMG] xArm emergency_stop failed: {e}")

    os._exit(1)


signal.signal(signal.SIGINT, sigint_handler)


def confirm(config, message):
    if config.get("confirm_each_step", True):
        input(f"{message}: Enter / 中止: Ctrl+C ")


def play_waypoint(waypoint_node, executor, yaml_path):
    """waypoint を再生し、終わるまで待つ。失敗したら RuntimeError。"""
    waypoint_node.reset()
    waypoint_node.play_direct(yaml_path)
    while rclpy.ok() and not waypoint_node.is_finished():
        executor.spin_once(timeout_sec=0.1)
    if waypoint_node.is_failed():
        raise RuntimeError(f"Waypoint failed: {waypoint_node.error_message()}")


def run_once(config, arm, hand, monitor, waypoint_node, executor, from_init):
    side = config["side"]
    wp = config["paths"]["waypoint"]
    ap = config["approach"]
    approach_rel = [
        float(ap["dx_mm"]),
        float(ap["dy_mm"]),
        float(ap["dz_mm"]),
        float(np.radians(ap["droll_deg"])),
        0.0,
        0.0,
    ]
    open_width = float(config["gripper_open_width_mm"])

    # ===== init → 撮影姿勢 =====
    if from_init:
        confirm(config, "init → 撮影姿勢へ移動")
        play_waypoint(waypoint_node, executor, wp["init_to_capture"][side])

    # ===== 撮影姿勢の高さ調整 =====
    capture_dz = float(config.get("capture_dz_mm", 0.0))
    if capture_dz != 0.0:
        confirm(config, f"撮影姿勢の高さを {capture_dz:+.1f} mm 調整")
        safe_motion(lambda: arm.moveL_tcp_z_offset(capture_dz), monitor, "capture_dz")
        time.sleep(1.0)

    capture_pose = arm.get_tcp_pose(is_radian=True)
    print(f"""
    ===== 固定動作 =====
    side          : {side}
    撮影姿勢 TCP  : X={capture_pose[0]:.2f}, Y={capture_pose[1]:.2f}, Z={capture_pose[2]:.2f} mm, roll={np.degrees(capture_pose[3]):.2f} deg
    アプローチ    : dx={approach_rel[0]:.1f}, dy={approach_rel[1]:.1f}, dz={approach_rel[2]:.1f} mm, droll={ap["droll_deg"]:.1f} deg
    開口幅        : {open_width:.1f} mm
    挿入量        : {INSERT_DX:.1f} mm / 引き抜き量: {RETRIEVAL_DX:.1f} mm
    ====================
    """)

    # アームが対象に近づく前に、ハンドを安全な幅(GRIPPER_CLOSE)まで閉じておく
    HandBook_retrieval.close_to_home(hand)

    # ===== 撮影姿勢 → アプローチ位置 =====
    confirm(config, "アプローチ位置へ移動")
    safe_motion(lambda: arm.moveL_relative(approach_rel), monitor, "approach")

    print(f"[DEBUG] open_until_width: target={open_width:.1f} mm")
    HandBook_retrieval.open_until_width(hand, open_width, gravity=False)
    time.sleep(2.0)  # グリッパーが開き終わるまで待機

    # ===== 挿入 → 把持 → 引き抜き =====
    confirm(config, "挿入")
    if side == "right":
        safe_motion(lambda: arm.moveL_to_insert_right(), monitor, "insert_right")
        HandBook_retrieval.grasp(hand)
        safe_motion(lambda: arm.moveL_post_grasp_right(), monitor, "retreave_right")
    else:
        safe_motion(lambda: arm.moveL_to_insert_left(), monitor, "insert_left")
        HandBook_retrieval.grasp(hand)
        safe_motion(lambda: arm.moveL_post_grasp_left(), monitor, "retreave_left")

    # ===== 解放 → ハンドリセット → 初期姿勢へ =====
    confirm(config, "ハンドを開いて解放し、初期姿勢へ戻る")
    HandBook_retrieval.open_until_full(hand)
    time.sleep(1.5)
    HandBook_retrieval.close_to_home(hand)
    play_waypoint(waypoint_node, executor, wp["capture_to_init"][side])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="fixed_insert_grasp.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    if config["side"] not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")

    rclpy.init()
    node = rclpy.create_node("fixed_insert_grasp")
    arm = XArm7(node=node, host=config["robot"]["xarm"]["host"])
    globals()["arm"] = arm

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    monitor = XArmMonitor(arm)

    waypoint_node = WaypointPlayerNode(
        node_name="fixed_insert_grasp_waypoint",
        arm=arm,
        monitor=monitor,
        yaml_path=config["paths"]["waypoint"]["init_to_capture"][config["side"]],
        speed=0.7,
        accel=1.0,
    )
    executor.add_node(waypoint_node)

    hand = HandBook_retrieval.init_dynamixels()
    globals()["HandMotors_retrieval"] = hand
    print("xarm ready")

    try:
        repeat = int(config.get("repeat", 1))
        for i in range(repeat):
            print(f"\n########## {i + 1} / {repeat} ##########")
            # 2回目以降は初期姿勢に戻っているので、必ず init → capture から始める
            from_init = config.get("start_from_init", True) or i > 0
            run_once(config, arm, hand, monitor, waypoint_node, executor, from_init)
        print("done")

    except Exception:
        import traceback
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGINT)

    finally:
        for n in (waypoint_node, node):
            try:
                n.destroy_node()
            except Exception:
                pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
