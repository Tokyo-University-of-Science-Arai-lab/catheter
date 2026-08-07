from xarm7.control.xarm7 import (
    XArm7,
    CAPTURE_RIGHT,
    TCP_VEL_1,
    TCP_ACC_2,
)
from rs_d435i.get_book_position import GetBookSpinePosition
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook_retrieval 
import Dynamixel_win_pro_hand_book.HandBook_Storage as HandBook_storage
from pathlib import Path
from detection.pro_handbook.sam3_runtime.integration_service_manager import (
    Sam3ServiceSession,
)
from detection.pro_handbook.sam_py_demo.get_book_points_no_mask_merge_no_side_filter import (
    run_capture_and_pca_no_mask_merge_no_side_filter,
)
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
import math
import sys
from xarm7.control.xarm_monitor import XArmMonitor, safe_motion
from datetime import datetime
from std_msgs.msg import Bool
import yaml
import ezodf
from xarm7.control.robot_base_coordinate import print_camera_debug_info
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic_ros2_editing import (
    capture_barcode_and_x_offset,
    WallDistanceWatcher,
    BoolPulseWatcher,
    BoolLatchWatcher
)


class XArm7C(XArm7):
    """R_I_SAM3_C専用のXArm7拡張。

    moveL_to_insert_right_with_current_monitor()の失敗時リトライ動作を、
    xarm7.py本体を変更せずにこのファイル内だけで上書きする。
    元の実装はJ1電流閾値超過を検知すると、挿入途中の位置から
    撮影姿勢(CAPTURE_RIGHT)へ関節空間で直接moveJ相当の動作(set_servo_angle)
    を行うため、TCPが斜めに移動し隣接する書籍に接触する恐れがある。
    ここでは撮影姿勢へ戻る前に、挿入方向(+X)と逆向きにX軸方向だけの
    直線移動(moveL)で書架から一度まっすぐ抜けてから関節姿勢へ戻す。
    """

    def moveL_to_insert_right_with_current_monitor(
        self,
        velocity: float = TCP_VEL_1,
        acceleration: float = TCP_ACC_2,
        j1_threshold: float = 2.7,
        required_count: int = 1,
        sample_interval: float = 0.02,
        timeout: float = 10.0,
        return_joint_angles=None,
        return_speed: float = 10.0,
        return_acceleration: float = 50.0,
    ):
        import math
        import time

        arm = self.arm

        if arm is None or not arm.connected:
            raise RuntimeError("xArmに接続されていません")

        if required_count < 1:
            raise ValueError("required_countは1以上にしてください")

        def check_result(label, result):
            if result is None:
                return

            if isinstance(result, int):
                code = result
            elif isinstance(result, (tuple, list)):
                code = int(result[0])
            else:
                code = 0

            if code != 0:
                raise RuntimeError(
                    f"{label}失敗: code={code}, result={result}"
                )

        def get_j1_current():
            currents = arm.currents

            if currents is None or len(currents) < 7:
                raise RuntimeError(
                    f"関節電流を取得できません: {currents}"
                )

            j1 = float(currents[0])

            if not math.isfinite(j1):
                raise RuntimeError(
                    f"J1電流値が不正です: {j1}"
                )

            return j1

        def stop_motion():
            result = arm.set_state(4)
            check_result("挿入停止", result)
            time.sleep(0.2)

        def return_to_capture_pose():
            if return_joint_angles is None:
                return

            if len(return_joint_angles) != 7:
                raise ValueError(
                    "return_joint_anglesは7要素にしてください"
                )

            # ==================================================
            # 変更点: J1電流が急上昇してstop_motion()に至った場合、
            # xArm本体の衝突検知(error=31, Collision Caused Abnormal
            # Current)がラッチされていることがある。motion_enable/
            # set_mode/set_state(0)だけではこのerror_codeは解除
            # されないため、先にclean_warn/clean_errorで掃除する。
            # ==================================================
            if arm.warn_code != 0:
                arm.clean_warn()
            if arm.error_code != 0:
                arm.clean_error()

            # STOP状態から通常の位置制御状態へ戻す
            check_result(
                "motion_enable",
                arm.motion_enable(enable=True),
            )
            check_result(
                "set_mode",
                arm.set_mode(0),
            )
            check_result(
                "set_state",
                arm.set_state(0),
            )

            time.sleep(0.2)

            if arm.error_code != 0:
                raise RuntimeError(
                    "停止後にxArmエラーが発生しました: "
                    f"error_code={arm.error_code}"
                )

            # ==================================================
            # X軸retreatは廃止。引っかかった状態でX軸方向へ動かすと
            # xArm側の衝突検知(error/warn=31)を誘発し、その残留警告を
            # XArmMonitorの独立監視スレッドが後から拾って
            # os._exit(1)でプロセスごと落ちる事象を確認したため、
            # Retrieval_integration_SAM3.py（xarm7.py本来の実装）と
            # 同じ、直接moveJでの復帰に戻す。
            # ==================================================
            result = arm.set_servo_angle(
                angle=list(return_joint_angles),
                speed=return_speed,
                mvacc=return_acceleration,
                wait=True,
                is_radian=False,
            )

            check_result(
                "撮影姿勢への復帰",
                result,
            )

        # 電流をレポートする設定
        check_result(
            "電流レポート設定",
            arm.set_report_tau_or_i(1),
        )

        time.sleep(0.2)

        # 非同期で挿入開始
        result = self.moveL_to_insert_right(
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=True,
        )
        check_result("右挿入", result)

        start_time = time.monotonic()
        trigger_count = 0
        moving_detected = False

        while True:
            elapsed = time.monotonic() - start_time

            if not arm.connected:
                raise RuntimeError(
                    "xArmとの接続が切断されました"
                )

            if arm.error_code != 0:
                stop_motion()

                raise RuntimeError(
                    "挿入中にxArmエラーが発生しました: "
                    f"error_code={arm.error_code}"
                )

            # J1電流監視
            j1_current = get_j1_current()

            if j1_current > j1_threshold:
                trigger_count += 1
            else:
                trigger_count = 0

            if trigger_count >= required_count:
                print(
                    "[INSERT] J1電流閾値超過: "
                    f"current={j1_current:+.3f}, "
                    f"threshold={j1_threshold:+.3f}"
                )

                stop_motion()
                return_to_capture_pose()

                return {
                    "ok": False,
                    "reason": "current_threshold",
                    "j1_current": j1_current,
                }

            # 移動完了確認
            moving_result = arm.get_is_moving()

            if isinstance(moving_result, bool):
                moving = moving_result
            elif (
                isinstance(moving_result, (tuple, list))
                and len(moving_result) >= 2
            ):
                check_result(
                    "get_is_moving",
                    moving_result,
                )
                moving = bool(moving_result[1])
            else:
                moving = None

            if moving is True:
                moving_detected = True

            if (
                moving_detected
                and moving is False
            ):
                return {
                    "ok": True,
                    "reason": "completed",
                }

            # 短い動作でmoving=Trueを取得できなかった場合
            if (
                elapsed >= 1.0
                and not moving_detected
                and moving is False
            ):
                return {
                    "ok": True,
                    "reason": "completed",
                }

            if elapsed >= timeout:
                stop_motion()

                return {
                    "ok": False,
                    "reason": "timeout",
                }

            time.sleep(sample_interval)


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "Retrieval_integration.yaml"

# J1電流閾値を検知した場合の最大試行回数（初回を含む）
# 4回すべて失敗した場合は、その本を失敗扱いにして次の本へ進む。
MAX_CURRENT_INSERT_ATTEMPTS = 4

LOG_HEADER = [
    "timestamp",
    "book_name",
    "shelf_id",
    "roll_deg",
    "estimated_book_width_mm",
    "master_book_width_mm",
    "width_error_mm",
    "camera_x_mm",
    "camera_y_mm",
    "camera_z_mm",
    "robot_x_mm",
    "robot_y_mm",
    "robot_z_mm",
    "side",
    "height_mm",
    "result",
    "shot_dir",
    "memo",
]


def _optional_float(value):
    """Return a finite float, or None when the value is blank/invalid."""
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _csv_value(value):
    """Write missing values as an empty CSV cell instead of the text 'None'."""
    return "" if value is None else value


def _xyz_csv_values(point_mm):
    """Return XYZ as three CSV-safe floats, or three empty cells."""
    if point_mm is None:
        return ["", "", ""]
    try:
        xyz = np.asarray(point_mm, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        return ["", "", ""]
    if not np.all(np.isfinite(xyz)):
        return ["", "", ""]
    return [float(xyz[0]), float(xyz[1]), float(xyz[2])]


def _resolve_config_relative_path(config, raw_path):
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        config_dir = Path(config.get("_config_dir", SCRIPT_DIR))
        path = config_dir / path
    return path.resolve()


def _width_error_values(estimated_width_mm, master_width_mm):
    estimated = _optional_float(estimated_width_mm)
    master = _optional_float(master_width_mm)
    if estimated is None or master is None:
        return estimated, master, None
    error = estimated - master
    return estimated, master, error


def _short_shot_dir(shot_dir):
    """Return only the path beginning with /captures for CSV output."""
    if shot_dir is None or str(shot_dir).strip() in {"", "None"}:
        return ""

    path = Path(str(shot_dir)).expanduser()
    parts = path.parts

    # Keep the suffix from the first 'captures' component.
    try:
        captures_index = parts.index("captures")
    except ValueError:
        return str(path)

    suffix = Path(*parts[captures_index:]).as_posix()
    return f"/{suffix}"

def _ods_safe_value(value):
    """ODSへ書き込める値に変換する。"""
    if value is None:
        return ""

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float) and not math.isfinite(value):
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def _find_next_ods_row(sheet, column_count):
    """
    データが入っている最終行の次の行番号を返す。
    行番号は0始まり。
    """
    check_columns = min(column_count, sheet.ncols())

    for row_index in range(sheet.nrows() - 1, -1, -1):
        for col_index in range(check_columns):
            value = sheet[row_index, col_index].value

            if value is not None and str(value).strip() != "":
                return row_index + 1

    return 0

def write_log(
    config,
    book_name,
    shelf_id,
    roll_deg,
    estimated_book_width_mm,
    master_book_width_mm,
    camera_point_mm,
    robot_point_mm,
    side,
    height,
    result,
    shot_dir,
    memo,
):
    """retrieval_log.odsへ出庫結果を1行追記する。"""

    log_file = _resolve_config_relative_path(
        config,
        config["paths"]["log"]["retrieval_ods"],
    )

    if not log_file.exists():
        raise FileNotFoundError(
            f"ODSログファイルが見つかりません: {log_file}"
        )

    estimated, master, error = _width_error_values(
        estimated_book_width_mm,
        master_book_width_mm,
    )

    camera_xyz = _xyz_csv_values(camera_point_mm)
    robot_xyz = _xyz_csv_values(robot_point_mm)

    # 現在のCSVと同じ18列
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        book_name,
        shelf_id,
        _csv_value(_optional_float(roll_deg)),
        _csv_value(estimated),
        _csv_value(master),
        _csv_value(error),
        *camera_xyz,
        *robot_xyz,
        side,
        height,
        result,
        _short_shot_dir(shot_dir),
        "" if memo is None else str(memo),
    ]

    # ODSを開く
    document = ezodf.opendoc(str(log_file))

    if len(document.sheets) == 0:
        raise RuntimeError(
            f"ODS内にシートがありません: {log_file}"
        )

    # YAMLにretrieval_sheetがあればそのシートを使用
    sheet_name = (
        config.get("paths", {})
        .get("log", {})
        .get("retrieval_sheet")
    )

    if sheet_name:
        try:
            sheet = document.sheets[sheet_name]
        except KeyError:
            raise KeyError(
                f"ODS内にシート '{sheet_name}' がありません。"
                f"シート一覧: {list(document.sheets.names())}"
            )
    else:
        # 指定がない場合は一番左のシート
        sheet = document.sheets[0]

    required_columns = len(LOG_HEADER)

    # 18列未満なら列を追加
    if sheet.ncols() < required_columns:
        sheet.append_columns(
            required_columns - sheet.ncols()
        )

    # 行が存在しない場合
    if sheet.nrows() == 0:
        sheet.append_rows(1)

    # シートが完全に空なら1行目へヘッダを書く
    first_row_values = [
        sheet[0, col].value
        for col in range(required_columns)
    ]

    if not any(
        value is not None and str(value).strip() != ""
        for value in first_row_values
    ):
        for col, header in enumerate(LOG_HEADER):
            sheet[0, col].set_value(header)

    # 最後のデータ行の次を取得
    target_row = _find_next_ods_row(
        sheet,
        required_columns,
    )

    # 足りない行を追加
    if target_row >= sheet.nrows():
        sheet.append_rows(
            target_row - sheet.nrows() + 1
        )

    # 18列を書き込む
    for col, value in enumerate(row):
        sheet[target_row, col].set_value(
            _ods_safe_value(value)
        )

    # 元ファイルへ保存
    document.save()

    print(f"[ODS LOG] saved: result={result}")
    print(f"[ODS LOG] path: {log_file}")
    print(f"[ODS LOG] sheet: {sheet.name}")
    print(f"[ODS LOG] row: {target_row + 1}")
    print(
        "[ODS LOG] width: "
        f"estimated={estimated}, master={master}, error={error}"
    )
    print(f"[ODS LOG] camera XYZ [mm]: {camera_xyz}")
    print(f"[ODS LOG] robot XYZ [mm]: {robot_xyz}")

def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    return config
    
def sigint_handler(sig, frame):
    print("Ctrl+C detected → FORCE KILL")
    try:
        arm = globals().get("arm", None)
        if arm:
            arm.emergency_stop()
    except Exception:
        pass
    service_session = globals().get("_sam3_service_session")
    if service_session is not None:
        try:
            service_session.stop_if_owned()
        except Exception as e:
            print(f"[SAM3] owned service cleanup failed: {e}")
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
    book_width_offset: float,
    master_book_width_mm: float | None,
    tp: TargetPublisher,
    node,
    arm: XArm7,                     
    executor,
    waypoint_node: WaypointPlayerNode,
    shelf_manager: ShelfIDManager,
    monitor: XArmMonitor,
    done_pub,
):

    runtime_log = {
        "roll_deg": None,
        "estimated_book_width_mm": None,
        "camera_point_mm": None,
        "robot_point_mm": None,
        "shot_dir": None,
        "insert_attempt": 0,
        "safe_stop_logged": False,
        "error_logged": False,
    }
    side = None
    height = None
    shelf_id = bookshelf_ID
    shot_dir = None
    roll = None
    p_xmax = None
    book_width = None
    HandMotors_retrieval = None
    successful_insert_attempt = None
    
    try:
        print('start sequence')
        shelf_manager.received = False
        waypoint_node.reset()
        # ==============================
        # shelf_id 取得（YAML設定優先、なければトピック待ち）
        # ==============================
        yaml_shelf_id = config.get("shelf_id")
        if yaml_shelf_id:
            node.get_logger().info(f"Using shelf_id from config: {yaml_shelf_id}")
            shelf_manager.set_from_string(str(yaml_shelf_id))
        else:
            node.get_logger().info("Waiting for /shelf_id ...")
            while rclpy.ok() and not shelf_manager.is_received():
                executor.spin_once(timeout_sec=0.1)

        side = shelf_manager.get_side()
        height = shelf_manager.get_height()
        shelf_id = shelf_manager.get_shelf_id()
        tcp_offset = shelf_manager.get_tcp_z_offset()

        def on_abnormal(msg):
            # Some motion errors can call the callback more than once.
            if runtime_log["safe_stop_logged"]:
                return
            try:
                write_log(
                    config=config,
                    book_name=book_name,
                    shelf_id=shelf_id,
                    roll_deg=runtime_log["roll_deg"],
                    estimated_book_width_mm=runtime_log[
                        "estimated_book_width_mm"
                    ],
                    master_book_width_mm=master_book_width_mm,
                    camera_point_mm=runtime_log["camera_point_mm"],
                    robot_point_mm=runtime_log["robot_point_mm"],
                    side=side,
                    height=height,
                    result="safe_stop",
                    shot_dir=runtime_log["shot_dir"],
                    memo=msg,
                )
                runtime_log["safe_stop_logged"] = True
                runtime_log["error_logged"] = True
            except Exception as log_exc:
                print(f"[CSV LOG ERROR] safe_stop logging failed: {log_exc}")

        monitor.on_abnormal = on_abnormal

        print("Shelf side:", side)
        print("Lift height:", height)
        # ==============================
        HandMotors_retrieval = HandBook_retrieval.init_dynamixels() 
        print("xarm ready")
        safe_motion(lambda: arm.moveJ_to_init_Q_DEG(), monitor, "init_pose")
        bar_dir = Path(config["paths"]["capture"]["bookshelf_barcode"])
        bar_dir.mkdir(parents=True, exist_ok=True)

        wall_watcher = WallDistanceWatcher(node)

        # 少し待って最新値を受信
        timeout_sec = 2.0
        start = time.time()
        while rclpy.ok() and wall_watcher.get_distance() is None:
            executor.spin_once(timeout_sec=0.1)
            if time.time() - start > timeout_sec:
                break

        wall_distance = wall_watcher.get_distance()

        if wall_distance is None:
            node.get_logger().warn("wall_distance not received. Using default 0.25m")
            wall_distance = 0.25

        # ==============================
        # self-localization
        # ==============================
        # detected2, code_str, info = capture_barcode_and_x_offset(
        #     node=node,
        #     executor=executor,
        #     shot_dir=bar_dir,
        #     barcode_number=None,
        #     fx_px=2500.0,
        #     depth_m= 0.8 - wall_distance - 0.13 , # 0.8 : 通路幅， 0.13 : AMR中心からカメラまでの距離
        #     v_x=0.01,        # 好きな低速
        #     cmd_sign_x=1.0,  # 逆なら -1.0
        # )

        # ==============================
        # wall_distance watcher
        # ==============================
        wall_watcher = WallDistanceWatcher(node)

        # ==============================
        # navigation_goal / navigation_goal_final watchers
        # ==============================
        nav_goal_pulse = BoolPulseWatcher(node, "/navigation_goal")
        final_goal = BoolLatchWatcher(node, "/navigation_goal_final")

        node.get_logger().info("Start self-localization loop: wait /navigation_goal pulses until /navigation_goal_final==True")

        # デバッグ上書き回避したいならサブディレクトリを毎回作るのがおすすめ
        # 例: bar_dir / time.strftime("%Y%m%d_%H%M%S")
        bar_dir = Path("/home/book/pro_book_SAM3/pro_hand_book_python/captures/bookshelf_barcode")
        bar_dir.mkdir(parents=True, exist_ok=True)

        detected2 = False
        label_str = None
        info = None

        # while rclpy.ok() and not final_goal.is_true():
        #     executor.spin_once(timeout_sec=0.1)

        #     # /navigation_goal=True を受けるまで待つ
        #     if not nav_goal_pulse.consume():
        #         continue

        #     node.get_logger().info(
        #         "[bookshelf barcode] /navigation_goal received -> start self-localization"
        #     )

        #     wall_distance = wall_watcher.get_distance()
        #     if wall_distance is None:
        #         node.get_logger().warn("wall_distance not received. Using default 0.25m")
        #         wall_distance = 0.25

        #     detected2, label_str, info = capture_barcode_and_x_offset(
        #         node=node,
        #         executor=executor,
        #         shot_dir=bar_dir,
        #         fx_px=2500.0,
        #         depth_m=0.8 - wall_distance - 0.13,
        #         v_x=0.05,
        #         min_search_vx=0.02,
        #         max_search_vx=0.07,
        #         k_p_search=0.00008,
        #         cmd_sign_x=1.0,
        #         align_thresh_px=10.0,
        #         wait_navigation_goal=False,   # 外で待っているので False
        #         total_timeout_sec=1000.0,
        #         ocr_interval_sec=0.25,
        #         bbox_lost_grace_sec=0.5,
        #     )

        #     if detected2 and info is not None:
        #         node.get_logger().info(
        #             f"[bookshelf barcode] self-localization OK: X_offset={info['X_m']:.3f} [m]"
        #         )
        #     else:
        #         node.get_logger().warn(
        #             "[bookshelf barcode] self-localization failed"
        #         )

        # ==============================
        # init → capture 姿勢へ（Waypoint）
        # AMR不使用のため trigger_goal() で直接起動
        node.get_logger().info("Triggering waypoint directly (AMR not used)")
        waypoint_node.trigger_goal()

        while rclpy.ok() and not waypoint_node.is_finished():
            executor.spin_once(timeout_sec=0.1)
            
        if waypoint_node.is_failed():
            raise RuntimeError(
                f"Waypoint failed: {waypoint_node.error_message()}"
            )

        node.get_logger().info("Waypoint succeeded → start recognition")

        try:
            # 右側書架では、J1電流閾値を検知するとCAPTURE_RIGHTへ戻し、
            # 同じ本を再認識して最大4回まで再試行する。
            for insert_attempt in range(1, MAX_CURRENT_INSERT_ATTEMPTS + 1):
                runtime_log["insert_attempt"] = insert_attempt
                print(
                    f"[RETRIEVAL] recognition/insert attempt "
                    f"{insert_attempt}/{MAX_CURRENT_INSERT_ATTEMPTS}"
                )

                # 電流検知後はCAPTURE_RIGHTへ戻るため、
                # 初回・再試行のどちらでもTCP高さ補正を適用する。
                print("TCP調整開始")
                safe_motion(
                    lambda: arm.moveL_tcp_z_offset(tcp_offset),
                    monitor,
                    "tcp_z_offset",
                )
                time.sleep(1.0)

                # 前回の認識結果を残さない
                runtime_log["roll_deg"] = None
                runtime_log["estimated_book_width_mm"] = None
                runtime_log["camera_point_mm"] = None
                runtime_log["robot_point_mm"] = None
                runtime_log["shot_dir"] = None

                # ================== 認識 ==================
                print("認識開始")
                start = time.perf_counter()

                try:
                    roll, p_xmax, book_width, shot_dir = (
                        run_capture_and_pca_no_mask_merge_no_side_filter(
                            query=book_name,
                            sam_device="gpu",
                        )
                    )

                    shot_dir = Path(shot_dir)

                    runtime_log["roll_deg"] = float(np.degrees(roll))
                    runtime_log["estimated_book_width_mm"] = float(book_width)
                    runtime_log["shot_dir"] = shot_dir

                    print(
                        f"""
                        ===== PCA RESULT =====
                        roll        : {roll}
                        p_xmax      : {p_xmax}
                        book_width  : {book_width}
                        ======================
                        """
                    )

                    if p_xmax is None:
                        raise RuntimeError(
                            "Recognition failed: p_xmax is None"
                        )

                    # p_xmaxはカメラ座標系[m]。ログでは[mm]にする。
                    runtime_log["camera_point_mm"] = (
                        np.asarray(
                            p_xmax,
                            dtype=np.float64,
                        ).reshape(3)
                        * 1000.0
                    )

                except Exception as e:
                    print(
                        f" recognition failed -> "
                        f"skip this book: {e}"
                    )
                    write_log(
                        config=config,
                        book_name=book_name,
                        shelf_id=shelf_id,
                        roll_deg=runtime_log["roll_deg"],
                        estimated_book_width_mm=runtime_log[
                            "estimated_book_width_mm"
                        ],
                        master_book_width_mm=master_book_width_mm,
                        camera_point_mm=runtime_log[
                            "camera_point_mm"
                        ],
                        robot_point_mm=runtime_log[
                            "robot_point_mm"
                        ],
                        side=side,
                        height=height,
                        result="recognition_fail",
                        shot_dir=runtime_log["shot_dir"],
                        memo=(
                            f"attempt={insert_attempt}/"
                            f"{MAX_CURRENT_INSERT_ATTEMPTS}; "
                            f"retry_count={max(0, insert_attempt - 1)}; "
                            f"{type(e).__name__}: {e}"
                        ),
                    )
                    traceback.print_exc()

                    # 認識失敗時はコンテナ収納へ進まず、
                    # 初期姿勢へ戻して次の本へ進む。
                    tp.publish_target_mm(
                        config["linear_lift"]["home_mm"]
                    )
                    rclpy.spin_once(tp, timeout_sec=0.1)

                    waypoint_node.reset()
                    waypoint_path = config["paths"]["waypoint"][
                        "capture_to_init"
                    ][side]
                    waypoint_node.play_direct(waypoint_path)

                    while (
                        rclpy.ok()
                        and not waypoint_node.is_finished()
                    ):
                        executor.spin_once(timeout_sec=0.1)

                    if waypoint_node.is_failed():
                        raise RuntimeError(
                            "認識失敗後の初期姿勢復帰に失敗: "
                            f"{waypoint_node.error_message()}"
                        )

                    shelf_manager.received = False

                    done_msg = Bool()
                    done_msg.data = True
                    done_pub.publish(done_msg)
                    rclpy.spin_once(node, timeout_sec=0.1)
                    return 0.0

                end = time.perf_counter()
                print(f"{end - start} sec.")

                print("roll (deg) =", np.degrees(roll))

                if np.degrees(roll) > 90.0:
                    roll = -(roll - np.radians(90.0))
                elif np.degrees(roll) < -90.0:
                    roll = -(roll + np.radians(90.0))
                else:
                    roll = 0.0

                out = {
                    "adjusted_roll_rad": float(roll),
                    "adjusted_roll_deg": float(
                        np.degrees(roll)
                    ),
                }
                (shot_dir / "adjusted_roll.json").write_text(
                    json.dumps(
                        out,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                print(
                    "adjusted roll (deg) =",
                    np.degrees(roll),
                )
                runtime_log["roll_deg"] = float(
                    np.degrees(roll)
                )

                p_max = 1000 * p_xmax

                print_camera_debug_info(
                    arm,
                    p_max,
                )

                # ロボット座標系での対象点[mm]
                p_robot_mm = cam_mm_to_robot_mm(
                    arm,
                    p_max,
                )
                runtime_log["camera_point_mm"] = np.asarray(
                    p_max,
                    dtype=np.float64,
                ).reshape(3).copy()
                runtime_log["robot_point_mm"] = np.asarray(
                    p_robot_mm,
                    dtype=np.float64,
                ).reshape(3).copy()

                safe_motion(
                    lambda: arm.move_to_target_xyz_and_roll(
                        p_robot_mm=p_robot_mm,
                        d_roll_rad=roll,
                        side=side,
                    ),
                    monitor,
                    "insertz_before",
                )

                HandBook_retrieval.open_until_width(
                    HandMotors_retrieval,
                    book_width - 10.0,
                    gravity=False,
                )

                # ================== 挿入・取り出し ==================
                #input("insert: Enter / ""return to capture: Ctrl+D / ""exit: Ctrl+C")
                time.sleep(3.0)

                if side == "right":
                    # 通常と同じmoveL挿入を行い、挿入中だけJ1電流を監視する。
                    insert_result = (
                        arm.moveL_to_insert_right_with_current_monitor(
                            return_joint_angles=CAPTURE_RIGHT,
                        )
                    )

                    if not insert_result["ok"]:
                        reason = insert_result.get(
                            "reason",
                            "unknown",
                        )

                        if reason == "current_threshold":
                            print(
                                "[RETRY] J1電流閾値を検知。"
                                "撮影姿勢へ戻ったため、"
                                "同じ本を再認識します。"
                            )

                            # この試行で使った認識結果を、次の試行で上書きする前に
                            # ODSへ1行保存する。1回目失敗、2回目成功なら、
                            # 同じ本がODS上で2行連続して記録される。
                            is_last_attempt = (
                                insert_attempt >= MAX_CURRENT_INSERT_ATTEMPTS
                            )
                            retry_result = (
                                "current_threshold_fail"
                                if is_last_attempt
                                else "current_threshold_retry"
                            )
                            retry_memo = (
                                f"attempt={insert_attempt}/"
                                f"{MAX_CURRENT_INSERT_ATTEMPTS}; "
                                f"retry_count={insert_attempt}; "
                                "J1 current threshold detected; "
                                + (
                                    "container storage skipped"
                                    if is_last_attempt
                                    else "retry same book"
                                )
                            )
                            write_log(
                                config=config,
                                book_name=book_name,
                                shelf_id=shelf_id,
                                roll_deg=runtime_log["roll_deg"],
                                estimated_book_width_mm=runtime_log[
                                    "estimated_book_width_mm"
                                ],
                                master_book_width_mm=master_book_width_mm,
                                camera_point_mm=runtime_log[
                                    "camera_point_mm"
                                ],
                                robot_point_mm=runtime_log[
                                    "robot_point_mm"
                                ],
                                side=side,
                                height=height,
                                result=retry_result,
                                shot_dir=runtime_log["shot_dir"],
                                memo=retry_memo,
                            )
                            print(
                                "[ODS RETRY LOG] "
                                f"attempt={insert_attempt}/"
                                f"{MAX_CURRENT_INSERT_ATTEMPTS}, "
                                f"result={retry_result}"
                            )

                            # アームはCAPTURE_RIGHTへ戻っているが、
                            # ハンドは本幅まで開いたままなので閉じ直す。
                            print(
                                "[HAND RETRY] 再認識前に"
                                "ハンドを閉じます。"
                            )
                            try:
                                HandBook_retrieval.grasp(
                                    HandMotors_retrieval
                                )
                            except Exception as hand_exc:
                                raise RuntimeError(
                                    "電流検知後のハンド閉鎖に失敗しました: "
                                    f"{hand_exc}"
                                ) from hand_exc

                            time.sleep(1.0)
                            print(
                                "[HAND RETRY] ハンド閉鎖指令完了。"
                            )

                            if (
                                insert_attempt
                                >= MAX_CURRENT_INSERT_ATTEMPTS
                            ):
                                print(
                                    "[SKIP] J1電流閾値を"
                                    f"{MAX_CURRENT_INSERT_ATTEMPTS}"
                                    "回連続で検知しました。"
                                    "コンテナ収納を行わず、"
                                    "次の本へ進みます。"
                                )

                                # 監視付き挿入関数によってCAPTURE_RIGHTへ
                                # 戻っているので、リフトとアームを初期状態へ戻す。
                                tp.publish_target_mm(
                                    config["linear_lift"]["home_mm"]
                                )
                                rclpy.spin_once(tp, timeout_sec=0.1)

                                waypoint_node.reset()
                                waypoint_path = config["paths"]["waypoint"][
                                    "capture_to_init"
                                ][side]
                                waypoint_node.play_direct(waypoint_path)

                                while (
                                    rclpy.ok()
                                    and not waypoint_node.is_finished()
                                ):
                                    executor.spin_once(timeout_sec=0.1)

                                if waypoint_node.is_failed():
                                    raise RuntimeError(
                                        "電流閾値失敗後の初期姿勢復帰に失敗: "
                                        f"{waypoint_node.error_message()}"
                                    )

                                shelf_manager.received = False

                                done_msg = Bool()
                                done_msg.data = True
                                done_pub.publish(done_msg)
                                rclpy.spin_once(node, timeout_sec=0.1)

                                # 0.0はコンテナ内の使用幅へ加算されない。
                                return 0.0

                            # ループ先頭へ戻り、
                            # TCP補正→認識→把持位置移動をやり直す。
                            continue

                        raise RuntimeError(
                            "右挿入に失敗しました: "
                            f"{insert_result}"
                        )

                    # 挿入成功後に本を把持し、書架から引き抜く。
                    HandBook_retrieval.grasp(
                        HandMotors_retrieval
                    )
                    safe_motion(
                        lambda: arm.moveL_post_grasp_right(),
                        monitor,
                        "retreave_right",
                    )

                else:
                    safe_motion(
                        lambda: arm.moveL_to_insert_left(),
                        monitor,
                        "insert_left",
                    )
                    HandBook_retrieval.grasp(
                        HandMotors_retrieval
                    )
                    safe_motion(
                        lambda: arm.moveL_post_grasp_left(),
                        monitor,
                        "retreave_left",
                    )

                # 挿入・取り出し成功。最終成功ログに試行回数を残す。
                successful_insert_attempt = insert_attempt
                break

            # ==========================
            # 引き抜き成功後: pro_book の Retrieval_integration.py と同じく、
            # コンテナ収納(Move_to_Container)・バーコード照合は行わず、
            # ハンドを開いて（書籍解放）次の本へ進む。
            # ==========================
            HandBook_retrieval.open_until_full(
                HandMotors_retrieval,
                asynchronous=False,
            )
            time.sleep(1.5)  # 開き終わるまで待機
            HandBook_retrieval.close_to_home(
                HandMotors_retrieval
            )  # 完全に閉じる（次書籍に備えてリセット）

            node.get_logger().info(
                "[HAND] 引き抜き成功後にハンドを開き、次の本へ進みます"
            )


        except EOFError:
            #ctrl + D によってその書籍出庫はスキップし初期姿勢に戻る
            print("Ctrl+D detected → return to capture")
    
            if side == "right":
                safe_motion(lambda: arm.moveL_post_grasp_right() , monitor, "retreave_right")   #書籍を引き抜く   
            else:
                safe_motion(lambda: arm.moveL_post_grasp_left() , monitor, "retreave_left")   #書籍を引き抜く

            HandBook_retrieval.grasp(HandMotors_retrieval)
            tp.publish_target_mm(config["linear_lift"]["home_mm"])
            #初期姿勢へ戻る
            waypoint_node.reset()

            waypoint_node.play_direct(
                config["paths"]["waypoint"]["capture_to_init"][side]
            )

            while rclpy.ok() and not waypoint_node.is_finished():
                executor.spin_once(timeout_sec=0.1)

            memo = "manual_skip_before_insertion"

            write_log(
                config=config,
                book_name=book_name,
                shelf_id=shelf_id,
                roll_deg=runtime_log["roll_deg"],
                estimated_book_width_mm=runtime_log[
                    "estimated_book_width_mm"
                ],
                master_book_width_mm=master_book_width_mm,
                camera_point_mm=runtime_log["camera_point_mm"],
                robot_point_mm=runtime_log["robot_point_mm"],
                side=side,
                height=height,
                result="ctrl+d",
                shot_dir=runtime_log["shot_dir"],
                memo=memo,
            )
            rclpy.spin_once(tp, timeout_sec=0.1)

            done_msg = Bool()
            done_msg.data = True
            done_pub.publish(done_msg)
            rclpy.spin_once(node, timeout_sec=0.1)

            return 0.0   # ← 次の本へ


        except Exception as e:
            print("xArm7 error")
            if not runtime_log["error_logged"]:
                write_log(
                    config=config,
                    book_name=book_name,
                    shelf_id=shelf_id,
                    roll_deg=runtime_log["roll_deg"],
                    estimated_book_width_mm=runtime_log[
                        "estimated_book_width_mm"
                    ],
                    master_book_width_mm=master_book_width_mm,
                    camera_point_mm=runtime_log["camera_point_mm"],
                    robot_point_mm=runtime_log["robot_point_mm"],
                    side=side,
                    height=height,
                    result="motion_error",
                    shot_dir=runtime_log["shot_dir"],
                    memo=f"{type(e).__name__}: {e}",
                )
                runtime_log["error_logged"] = True
            raise
        
        
        waypoint_node.reset()
        waypoint_node.play_direct(
            "/home/book/pro_book_SAM3/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/init.yaml"
        )

        wait_start_time = time.time()
        while rclpy.ok() and (time.time() - wait_start_time) < 1.5:  # 待つ（この間にアームの移動が終わることが多い）
            executor.spin_once(timeout_sec=0.1)  # ROSの通信を維持
            #もしアームの移動が終わってしまったら待機を終了
            if waypoint_node.is_finished():
                break

        tp.publish_target_mm(config["linear_lift"]["home_mm"])
        rclpy.spin_once(tp, timeout_sec=0.1)
        while rclpy.ok() and not waypoint_node.is_finished():
            executor.spin_once(timeout_sec=0.1)

        shelf_manager.received = False
        if successful_insert_attempt is None:
            successful_insert_attempt = max(
                1,
                int(runtime_log.get("insert_attempt", 1)),
            )

        memo = (
            f"attempt={successful_insert_attempt}/"
            f"{MAX_CURRENT_INSERT_ATTEMPTS}; "
            f"retry_count={max(0, successful_insert_attempt - 1)}; "
            "container_storage=skipped (pro_book-style flow)"
        )
        write_log(
            config=config,
            book_name=book_name,
            shelf_id=shelf_id,
            roll_deg=runtime_log["roll_deg"],
            estimated_book_width_mm=runtime_log[
                "estimated_book_width_mm"
            ],
            master_book_width_mm=master_book_width_mm,
            camera_point_mm=runtime_log["camera_point_mm"],
            robot_point_mm=runtime_log["robot_point_mm"],
            side=side,
            height=height,
            result="success",
            shot_dir=runtime_log["shot_dir"],
            memo=memo,
        )
        print('sequence done')

        done_msg = Bool()
        done_msg.data = True
        done_pub.publish(done_msg)
        rclpy.spin_once(node, timeout_sec=0.1)

        return book_width   

    except Exception as e:
        print("Abort sequence due to exception")
        traceback.print_exc()

        # Avoid a duplicate fatal row when XArmMonitor already wrote safe_stop.
        if not runtime_log["error_logged"]:
            try:
                write_log(
                    config=config,
                    book_name=book_name,
                    shelf_id=shelf_id,
                    roll_deg=runtime_log["roll_deg"],
                    estimated_book_width_mm=runtime_log[
                        "estimated_book_width_mm"
                    ],
                    master_book_width_mm=master_book_width_mm,
                    camera_point_mm=runtime_log["camera_point_mm"],
                    robot_point_mm=runtime_log["robot_point_mm"],
                    side=side,
                    height=height,
                    result="fatal_error",
                    shot_dir=runtime_log["shot_dir"],
                    memo=f"{type(e).__name__}: {e}",
                )
            except Exception as log_exc:
                print(f"[CSV LOG ERROR] fatal logging failed: {log_exc}")

        if HandMotors_retrieval is not None:
            try:
                HandMotors_retrieval.disable_torque(
                    HandBook_retrieval.GRIPPER_ID
                )
                HandMotors_retrieval.close_port()
            except Exception:
                pass
        os.kill(os.getpid(), signal.SIGINT)
        return None
        
def main():
    config = load_config(CONFIG_PATH)
    print(f"[CONFIG] loaded: {config['_config_path']}")
    print(
        "[CONFIG] retrieval log: "
        f"{_resolve_config_relative_path(config, config['paths']['log']['retrieval'])}"
    )

    # ==============================
    # マスターデータ読み込み
    # ==============================
    master_file = _resolve_config_relative_path(
        config,
        config["books"]["master_file"],
    )
    print(f"[MASTER] loaded: {master_file}")
    with master_file.open("r", encoding="utf-8") as f:
        books_master = json.load(f)

    retrieved_book_width_list = [0.0]

    # ==============================
    # ROS2 初期化
    # ==============================
    rclpy.init()

    # メイン制御ノード（publish / log / trigger 用）
    node = rclpy.create_node("book_retrieval_main")
    done_pub = node.create_publisher(Bool, "/retrieval_done", 10)
    XARM_HOST = config["robot"]["xarm"]["host"]

    arm = XArm7C(
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
        speed=0.7,
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
                book_width_offset=book_width_offset,
                master_book_width_mm=_optional_float(b.get("book_width")),
                tp=tp,
                node=node,
                arm=arm, 
                monitor=monitor,
                executor=executor,
                waypoint_node=waypoint_node,   
                shelf_manager=waypoint_node.shelf_manager,
                done_pub=done_pub,
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
    _sam3_service_session = Sam3ServiceSession()
    with _sam3_service_session:
        main()