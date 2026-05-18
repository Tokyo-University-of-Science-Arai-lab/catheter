from xarm7.control.xarm7 import XArm7
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook
from detection.pro_handbook.sam_py_demo.Storage_rev import run_capture_and_pca_depth_space
from xarm7.control.robot_base_coordinate import cam_mm_to_robot_mm
from xarm7.control.shelf_id_manager import ShelfIDManager
import rclpy
import time
import numpy as np
from linear_lift import TargetPublisher

def detect_storage_target(Xarm7):
    """
    収納スペースを認識し、
    ロボット座標系の目標位置・角度・水平移動量dyを返す
    """
    shelf_manager = ShelfIDManager()
    height = shelf_manager.get_height()
    tp = TargetPublisher()

    tp.publish_target_mm(height)

    ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
    time.sleep(1.0)

    angle_rad, first_target_cam, res = run_capture_and_pca_depth_space(
        crop_y_min=120,
        crop_y_max=550,
        min_area_px=800,
        min_height_px=100,
        max_space_width_px=420,
        target_y_from_space_top_px=240,
    )

    angle_deg = float(np.degrees(angle_rad))
    oblique_line = float(res.get("guide_edge_length_mm", 0.0))

    # cosにはdegreeではなくradを入れる
    dy = -oblique_line * np.cos(angle_rad) / 2

    print(f"斜辺の長さ：L={oblique_line}mm")
    print(f"マニピュレー水平移動：dy={dy}mm")

    print("========== run_capture_and_pca result ==========")
    print(f"[RESULT] angle_rad = {angle_rad:.6f} rad")
    print(f"[RESULT] angle_deg = {angle_deg:.2f} deg")

    print(
        "[RESULT] first_target_cam [m] = "
        f"X={first_target_cam[0]:.4f}, "
        f"Y={first_target_cam[1]:.4f}, "
        f"Z={first_target_cam[2]:.4f}"
    )

    print(
        "[RESULT] first_target_cam [mm] = "
        f"X={first_target_cam[0] * 1000:.1f}, "
        f"Y={first_target_cam[1] * 1000:.1f}, "
        f"Z={first_target_cam[2] * 1000:.1f}"
    )

    print(f"[RESULT] pair_indices = {res.get('pair_indices', None)}")
    print(f"[RESULT] line_p0 = {res.get('line_p0', None)}")
    print(f"[RESULT] line_p1 = {res.get('line_p1', None)}")
    print(f"[RESULT] sub_line_p0 = {res.get('sub_line_p0', None)}")
    print(f"[RESULT] sub_line_p1 = {res.get('sub_line_p1', None)}")
    print("================================================")

    first_target_cam_mm = 1000.0 * first_target_cam
    print("[DEBUG] first_target_cam_mm =", first_target_cam_mm)

    first_target_robot_mm = cam_mm_to_robot_mm(Xarm7, first_target_cam_mm)
    print("[DEBUG] first_target_robot_mm =", first_target_robot_mm)

    return first_target_robot_mm, angle_deg, dy


def storage_sequence(Xarm7, Hand):
    """
    書籍収納シーケンス本体。
    xArm7とDynamixelは外で初期化して渡す。
    """

    first_target_robot_mm, angle_deg, dy = detect_storage_target(Xarm7)


    try:
        input("Enter: execute reaching / Ctrl+D: retract and return : ")
    except EOFError:
        print("[INFO] Ctrl+D detected before reaching")
        print("[INFO] retract spacer and return to capture pose")

        try:
            print("[DEBUG] before contract_sp_lin_2")
            HandBook.contract_sp_lin_2(Hand, asynchronous=False)
            print("[DEBUG] after contract_sp_lin_2")
        except Exception as e:
            print("[WARN] contract_sp_lin_2 failed:", e)

        try:
            print("[DEBUG] before moveJ_to_capture_right")
            ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
            print("[DEBUG] after moveJ_to_capture_right ret =", ret)
        except Exception as e:
            print("[WARN] moveJ_to_capture_right failed:", e)

        return False

    print("xArm7 moves to recognized storage target")
    print("[DEBUG] before move_to_storage_target_xyz_and_roll")

    ret = Xarm7.move_to_storage_target_xyz_and_roll(
        p_robot_mm=first_target_robot_mm,
        d_roll_rad=0.0,
        side="right",
        y_offset_mm=0.0,
        z_offset_mm=0.0,
        roll_scale=1.0,
    )

    print("[DEBUG] after move_to_storage_target_xyz_and_roll ret =", ret)

    try:
        input("After reaching: Enter: expand spacer / Ctrl+D: retract and return : ")
    except EOFError:
        print("[INFO] Ctrl+D detected after reaching")
        print("[INFO] return to capture pose")

        try:
            print("[DEBUG] before moveJ_to_capture_right")
            ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
            print("[DEBUG] after moveJ_to_capture_right ret =", ret)
        except Exception as e:
            print("[WARN] moveJ_to_capture_right failed:", e)

        return False

    print("[DEBUG] before expand_sp_lin")
    HandBook.expand_sp_lin(dxl=Hand, asynchronous=True)
    print("[DEBUG] after expand_sp_lin")

    print("waiting expansion")
    time.sleep(14)

    if angle_deg < 90:
        HandBook.rotate_spacer(Hand, angle_deg - 180)

        ret = Xarm7.moveL_to_insert_book_full(asynchronous=True)
        time.sleep(3.0)

        HandBook.reset_rot(Hand)

    else:
        HandBook.rotate_spacer(Hand, angle_deg)

        Xarm7.moveL_y_offset(y_offset=dy)

        ret = Xarm7.moveL_to_insert_book_full(
            velocity=15,
            acceleration=15,
            asynchronous=True
        )
        time.sleep(6.0)

        HandBook.reset_rot(Hand)

    time.sleep(2.0)

    HandBook.contract_sp_lin_1(Hand, asynchronous=False)
    print("[DEBUG] after contract_sp_lin_1")

    ret = Xarm7.move_L_to_insert_book_tip(
        velocity=15,
        acceleration=15,
        asynchronous=True
    )

    print("[DEBUG] before contract_sp_lin_2")
    HandBook.contract_sp_lin_2(Hand, asynchronous=False)
    print("[DEBUG] after contract_sp_lin_2")

    print("[DEBUG] before ungrasp")
    HandBook.ungrasp_auto(Hand)
    print("[DEBUG] after ungrasp")

    print("[DEBUG] before post_storage")
    ret = Xarm7.moveL_to_post_storage(asynchronous=True)
    print("[DEBUG] after post_storage ret =", ret)

    time.sleep(4.0)

    print("[DEBUG] before moveJ_to_capture_right")
    ret = Xarm7.moveJ_to_capture_right(asynchronous=False)
    print("[DEBUG] after moveJ_to_capture_right ret =", ret)

    HandBook.grasp(dxl=Hand, keep_torque=True)

    print("sequence done")

    return True


def shutdown_devices(Xarm7=None, Hand=None, node=None):
    """
    終了処理をまとめる。
    途中でエラーが出ても、できるだけ安全に止める。
    """

    if Hand is not None:
        try:
            Hand.disable_torque(HandBook.GRIPPER_ID)
            Hand.disable_torque(HandBook.SP_LIN_ID)
            Hand.disable_torque(HandBook.SP_ROT_ID)
            time.sleep(0.2)
        except Exception as e:
            print("[WARN] torque disable failed:", e)

        try:
            Hand.close_port()
        except Exception as e:
            print("[WARN] close_port failed:", e)

    if Xarm7 is not None:
        try:
            Xarm7.disconnect()
        except Exception as e:
            print("[WARN] xArm disconnect failed:", e)

    if node is not None:
        try:
            node.destroy_node()
        except Exception as e:
            print("[WARN] node destroy failed:", e)

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as e:
        print("[WARN] rclpy shutdown failed:", e)


def main():
    Hand = None
    Xarm7 = None
    node = None

    try:
        print("start storage sequence")

        Hand = HandBook.init_dynamixels()

        rclpy.init()
        node = rclpy.create_node("storage_node")

        Xarm7 = XArm7(
            node=node,
            host="192.168.2.197"
        )

        storage_sequence(Xarm7, Hand)

    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected")

    except Exception as e:
        print("[ERROR] storage sequence failed:", e)

    finally:
        shutdown_devices(Xarm7=Xarm7, Hand=Hand, node=node)
        print("dynamixel deactivated")


if __name__ == "__main__":
    main()