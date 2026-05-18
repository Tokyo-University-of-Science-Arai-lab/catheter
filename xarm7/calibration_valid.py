#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs

from control.ar_marker_pose import rs_color_K_dist, aruco_marker_pose_target2cam


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def to_h(p3: np.ndarray) -> np.ndarray:
    """3次元ベクトルを同次座標 [x,y,z,1]^T に拡張"""
    return np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float64)


def detect_marker(
    bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_len_m: float,
    dict_name: str,
    marker_id: int,
) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    """ArUco マーカー検出 -> (id, rvec(3,), tvec(3,)) を返す"""
    ret = aruco_marker_pose_target2cam(
        bgr=bgr,
        K=K,
        dist=dist,
        marker_len_m=marker_len_m,
        dict_name=dict_name,
        target_id=marker_id,
    )
    if ret is None:
        return None
    mid, rvec, tvec = ret
    return int(mid), np.array(rvec, dtype=np.float64).reshape(3), np.array(tvec, dtype=np.float64).reshape(3)


def measure_marker_center_cam_m(
    pipe: rs.pipeline,
    K: np.ndarray,
    dist: np.ndarray,
    marker_len_m: float,
    dict_name: str,
    marker_id: int,
    avg_n: int = 10,
    timeout_s: float = 8.0,
    preview: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    マーカー中心Oのカメラ座標 O_cam[m] を avg_n 回平均して返す
    """
    t0 = time.time()
    ts: List[np.ndarray] = []
    last_vis: Optional[np.ndarray] = None

    while len(ts) < avg_n:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(
                f"Marker not detected enough times (got {len(ts)}/{avg_n}) within {timeout_s:.1f}s"
            )

        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        bgr = np.asanyarray(color.get_data())
        vis = bgr.copy()

        det = detect_marker(bgr, K, dist, marker_len_m, dict_name, marker_id)
        if det is None:
            if preview:
                cv2.putText(
                    vis,
                    "Marker: not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("handeye_reach", vis)
                cv2.waitKey(1)
            continue

        mid, rvec, tvec = det
        ts.append(tvec)
        last_vis = vis

        if preview:
            cv2.putText(
                vis,
                f"Marker ok {len(ts)}/{avg_n}  t[m]=({tvec[0]:.3f},{tvec[1]:.3f},{tvec[2]:.3f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            try:
                cv2.drawFrameAxes(vis, K, dist, rvec.reshape(3, 1), tvec.reshape(3, 1), 0.05)
            except Exception:
                pass
            cv2.imshow("handeye_reach", vis)
            cv2.waitKey(1)

    o_cam_m = np.mean(np.stack(ts, axis=0), axis=0)
    return o_cam_m, last_vis


def get_T_tcp_cam_from_json(handeye_json: Path, swap: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    JSONから T_tcp_cam（tcp<-cam）を得る。
    swap=True ならキーの意味が逆として取り扱う。
    """
    d = load_json(handeye_json)

    T_cam_tcp = np.array(d["T_cam_tcp"], dtype=np.float64)
    T_tcp_cam = np.array(d["T_tcp_cam"], dtype=np.float64)

    # 通常: T_tcp_cam を使う
    # swap: 「実は逆だった」場合に T_cam_tcp を T_tcp_cam として扱う
    T_tcp_cam_use = T_cam_tcp if swap else T_tcp_cam

    if T_tcp_cam_use.shape != (4, 4):
        raise ValueError(f"T_tcp_cam must be 4x4, got {T_tcp_cam_use.shape}")

    meta = {
        "created_at": d.get("created_at"),
        "settings": d.get("settings"),
        "source_path": str(handeye_json.resolve()),
        "swap_used": bool(swap),
    }
    return T_tcp_cam_use, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handeye-json", type=str, required=True,
                    help="hand-eye result json (contains T_cam_tcp & T_tcp_cam)")
    ap.add_argument("--xarm-host", type=str, default="192.168.2.197")
    ap.add_argument("--trials", type=int, default=10,
                    help="リーチ→手計測→戻る を何回行うか")
    ap.add_argument("--marker-id", type=int, default=0)
    # ★ サイズを 0.050 → 0.100
    ap.add_argument("--marker-len-m", type=float, default=0.100)
    # ★ dict を DICT_4X4_50 → DICT_5X5_100
    ap.add_argument("--aruco-dict", type=str, default="DICT_5X5_100")
    ap.add_argument("--avg-n", type=int, default=10,
                    help="マーカー位置平均に使うフレーム数")
    ap.add_argument("--x-offset-mm", type=float, default=0.0,
                    help="ベース座標系 X 方向オフセット [mm]")
    ap.add_argument("--y-offset-mm", type=float, default=0.0,
                    help="ベース座標系 Y 方向オフセット [mm]")
    ap.add_argument("--z-offset-mm", type=float, default=30.0,
                    help="ベース座標系 Z 方向に、マーカー中心からどれだけ上に TCP を置くか [mm]")
    ap.add_argument("--swap", action="store_true",
                    help="T_tcp_cam と T_cam_tcp の意味が逆のとき指定")
    ap.add_argument("--preview", action="store_true",
                    help="検出中にウィンドウで確認する")
    args = ap.parse_args()

    # ---- hand-eye 行列読み込み (tcp<-cam)
    T_tcp_cam, he_meta = get_T_tcp_cam_from_json(Path(args.handeye_json), swap=args.swap)

    # ---- RealSense start
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    intr, K, dist = rs_color_K_dist(profile)

    # ---- xArm connect
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(args.xarm_host)
    arm.connect()
    try:
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
    except Exception:
        pass

    # ---- 最初の撮影姿勢（基準ポーズ）を記録
    code_init, pose_init = arm.get_position(is_radian=False)  # mm, deg
    if code_init != 0 or pose_init is None:
        raise RuntimeError(f"failed to get initial xArm pose: code={code_init}, pose={pose_init}")
    x0, y0, z0, roll0, pitch0, yaw0 = [float(v) for v in pose_init]
    print("[INFO] initial pose (mm,deg):", pose_init)

    if args.preview:
        cv2.namedWindow("handeye_reach", cv2.WINDOW_NORMAL)

    print("\n[INFO] ===== Hand-Eye Reaching (manual error measurement) =====")
    print(
        f"[INFO] trials={args.trials}, avg_n={args.avg_n}, "
        f"offsets_mm=(x={args.x_offset_mm}, y={args.y_offset_mm}, z={args.z_offset_mm}), "
        f"swap={args.swap}"
    )
    print("[INFO] each trial:")
    print("       1) robot moves to initial pose")
    print("       2) measure marker center in camera")
    print("       3) reach around marker with given base-frame offsets")
    print("       4) YOU measure error, then press Enter to return to initial pose\n")

    try:
        for k in range(args.trials):
            print(f"\n===== TRIAL {k+1}/{args.trials} =====")

            # 0) 念のため初期姿勢へ戻す
            print("[STEP] move to initial pose...")
            ret = arm.set_position(
                x=x0, y=y0, z=z0,
                roll=roll0, pitch=pitch0, yaw=yaw0,
                speed=80, mvacc=500,
                wait=True
            )
            if ret != 0:
                print(f"[WARN] set_position to initial returned {ret}")
            time.sleep(0.3)

            input("[INFO] 初期姿勢に到達しました。マーカーが見えることを確認したら Enter で計測開始 > ")

            # 1) マーカー中心（カメラ座標）を平均取得
            print("[STEP] measuring marker center in camera frame...")
            o_cam_m, _ = measure_marker_center_cam_m(
                pipe, K, dist,
                marker_len_m=args.marker_len_m,
                dict_name=args.aruco_dict,
                marker_id=args.marker_id,
                avg_n=args.avg_n,
                timeout_s=8.0,
                preview=args.preview
            )

            # 2) 現在の TCP 姿勢 (base frame) 取得
            code, pose = arm.get_position(is_radian=False)
            if code != 0 or pose is None:
                raise RuntimeError(f"xArm get_position failed: code={code}, pose={pose}")
            x, y, z, roll, pitch, yaw = [float(v) for v in pose]
            print(f"[INFO] current pose (mm,deg): {pose}")

            # 3) roll,pitch,yaw を ZYX (yaw-pitch-roll) で回転行列に変換
            rr = np.deg2rad(roll)
            rp = np.deg2rad(pitch)
            ry = np.deg2rad(yaw)

            cz, sz = np.cos(ry), np.sin(ry)
            cy, sy = np.cos(rp), np.sin(rp)
            cx, sx = np.cos(rr), np.sin(rr)

            Rz = np.array(
                [[cz, -sz, 0],
                 [sz,  cz, 0],
                 [ 0,   0, 1]], dtype=np.float64
            )
            Ry = np.array(
                [[ cy, 0, sy],
                 [  0, 1,  0],
                 [-sy, 0, cy]], dtype=np.float64
            )
            Rx = np.array(
                [[1,  0,   0],
                 [0, cx, -sx],
                 [0, sx,  cx]], dtype=np.float64
            )

            R_base_tcp = Rz @ Ry @ Rx  # zyx
            p_base_tcp_m = np.array([x, y, z], dtype=np.float64) / 1000.0  # mm -> m

            T_base_tcp = np.eye(4, dtype=np.float64)
            T_base_tcp[:3, :3] = R_base_tcp
            T_base_tcp[:3, 3] = p_base_tcp_m

            # 4) T_base_cam = T_base_tcp * T_tcp_cam
            T_base_cam = T_base_tcp @ T_tcp_cam

            # 5) マーカー中心の base 座標 (m, mm) を求める
            Po_m = (T_base_cam @ to_h(o_cam_m))[:3]
            Po_mm = Po_m * 1000.0

            # ベース座標系でのオフセットを適用
            Po_mm_move = Po_mm.copy()
            Po_mm_move[0] += float(args.x_offset_mm)  # base X 方向
            Po_mm_move[1] += float(args.y_offset_mm)  # base Y 方向
            Po_mm_move[2] += float(args.z_offset_mm)  # base Z 方向

            print(f"[INFO] marker center in base (mm): {Po_mm.tolist()}")
            print(
                "[INFO] target TCP position with base offsets "
                f"(x+={args.x_offset_mm}, y+={args.y_offset_mm}, z+={args.z_offset_mm}): "
                f"{Po_mm_move.tolist()}"
            )

            # 6) その位置に TCP を移動（姿勢はそのまま）
            print("[STEP] moving TCP to target around marker...")
            ret = arm.set_position(
                x=float(Po_mm_move[0]),
                y=float(Po_mm_move[1]),
                z=float(Po_mm_move[2]),
                roll=roll, pitch=pitch, yaw=yaw,
                speed=80, mvacc=500,
                wait=True
            )
            if ret != 0:
                print(f"[WARN] arm.set_position returned {ret} (move to marker)")
            else:
                print("[INFO] reached target around marker.")

            print("\n[MEASURE] 今の状態で誤差を実測してください。")
            input("計測が終わったら Enter を押すと、初期姿勢へ戻ります > ")

            # 7) 初期姿勢に戻る
            print("[STEP] returning to initial pose...")
            ret = arm.set_position(
                x=x0, y=y0, z=z0,
                roll=roll0, pitch=pitch0, yaw=yaw0,
                speed=80, mvacc=500,
                wait=True
            )
            if ret != 0:
                print(f"[WARN] set_position to initial returned {ret}")
            else:
                print("[INFO] back to initial pose.")

        print("\n[INFO] all trials finished.")

    finally:
        try:
            pipe.stop()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
