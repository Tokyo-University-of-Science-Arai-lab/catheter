from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
import time
import numpy as np
import traceback

from detection.pro_handbook.sam_py_demo.bar_code.web_camera_capture import capture_one_depstech
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic import barcode_perception
from xarm7.control.xarm7 import XArm7


#BOOK_BARCODE_0 = [86, -60, 173, 63, 84, 9.4, 44]
BOOK_BARCODE_1 = [52.4, -82, 178, 78, 204, 4.6, -61.2]
#BOOK_BARCODE_1 = [39.8, -62.9, 177.6, 55.4, 163, 12.6, -63.8](昔のやつ)
#BOOK_BARCODE_2 = [-16.1, -80.3, 150.4, 89.3, 73.8, 30.2, -4.4]
BOOK_BARCODE_2 = [-36.9, -75.2, 159.3, 80.5, 83.8, 18.8, -34.9]
def decoded_to_dict_safe(d) -> dict:
    """barcode SDKのdecoded要素を、安全にJSON化できるdictへ変換する"""
    # data: bytes -> str
    raw_data = getattr(d, "data", None)
    if isinstance(raw_data, (bytes, bytearray)):
        data = raw_data.decode("utf-8", errors="replace")
    else:
        data = raw_data

    # rect
    rect = getattr(d, "rect", None)
    rect_dict = None
    if rect is not None:
        # rect.left/top/width/height が無い/Noneでも落ちないようにする
        rect_dict = {
            "left": int(getattr(rect, "left", 0) or 0),
            "top": int(getattr(rect, "top", 0) or 0),
            "width": int(getattr(rect, "width", 0) or 0),
            "height": int(getattr(rect, "height", 0) or 0),
        }

    # polygon
    poly = []
    for p in (getattr(d, "polygon", None) or []):
        poly.append({"x": int(getattr(p, "x", 0) or 0), "y": int(getattr(p, "y", 0) or 0)})

    # quality: int(None) で落ちるのを防ぐ
    q = getattr(d, "quality", -1)
    quality = int(q) if isinstance(q, (int, np.integer)) else -1

    return {
        "data": data,
        "type": getattr(d, "type", None),
        "quality": quality,
        "orientation": getattr(d, "orientation", None),
        "rect": rect_dict,
        "polygon": poly,
    }


def book_barcode_sequence(barcode_number_input: str, shot_dir: Path, arm: XArm7) -> bool:
    """
    - JSON保存で落ちても例外ログを shot_dir に残す
    - 途中で落ちてもロボットをできるだけ初期姿勢へ戻す
    - 戻り値は bool
    """
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    barcode_identification = False
    barcode_data_output = None
    log_path = shot_dir / f"book_barcode_result_{ts}.json"
    err_path = shot_dir / f"book_barcode_error_{ts}.txt"

    try:
        #arm.moveJ_to_capture_right()
        arm.switch_gripper_pose(BOOK_BARCODE_1)
        arm.moveL_relative([-50.0, 630.0, 50.0, 0.0, 0.0, 0.0])
        #arm.switch_gripper_pose(BOOK_BARCODE_2)

        time.sleep(2.0)

        # ---- Capture ----
        frame = capture_one_depstech(shot_dir / "book_barcode_capture.png")
        color_frame = np.asanyarray(frame)

        # ---- Perception ----
        barcode_identification, barcode_data_output = barcode_perception(barcode_number_input, color_frame)
        print("[book barcode] barcode_identification:", barcode_identification)
        print("[book barcode] barcode_data_output:", barcode_data_output)

        # ---- Save JSON (safe) ----
        payload = {
            "timestamp": ts,
            "barcode_number_input": barcode_number_input,
            "barcode_identification": bool(barcode_identification),
            "barcode_data_output": [decoded_to_dict_safe(d) for d in (barcode_data_output or [])],
        }

        # dumpsが通るか先に確認（ここで落ちるケースも拾う）
        s = json.dumps(payload, ensure_ascii=False, indent=2)
        log_path.write_text(s, encoding="utf-8")
        print("[book barcode] saved:", str(log_path))

        if barcode_identification:
            return "success"
        elif not barcode_data_output:
            return "no_barcode"
        else :
            return "wrong_barcode"
        

    except Exception:
        # 例外は shot_dir に必ず残す（原因特定用）
        try:
            err_path.write_text(traceback.format_exc(), encoding="utf-8")
            print("[book barcode] ERROR saved:", str(err_path))
        except Exception as e:
            print("[book barcode] ERROR while saving error log:", repr(e))
        # main側で拾いたければ raise、止めたくなければ return False
        return "error"

    # finally:
    #     # ここは必ず通る：ロボットをできるだけ元に戻す
    #     try:
    #         arm.switch_gripper_pose(BOOK_BARCODE_1)
    #         #arm.moveJ_to_capture_right()
    #     except Exception as e:
    #         print("[WARN] arm cleanup failed:", repr(e))


import rclpy
from rclpy.node import Node

class DummyNode(Node):
    def __init__(self):
        super().__init__("book_barcode_node")


def main():
    rclpy.init()

    node = DummyNode()
    arm = XArm7(node)   # ←ここ修正

    barcode_number_input = "1234567890"
    shot_dir = Path("./logs")
    shot_dir.mkdir(parents=True, exist_ok=True)

    result = book_barcode_sequence(barcode_number_input, shot_dir, arm)
    time.sleep(5.0)
    #arm.switch_gripper_pose(BOOK_BARCODE_2)
    arm.moveL_relative([0.0, -630.0, -50.0, 0.0, 0.0, 0.0])
    arm.switch_gripper_pose(BOOK_BARCODE_1)
    arm.moveJ_to_capture_right()
    print("RESULT:", result)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()