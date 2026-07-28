#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ハンドアイキャリブレーション用サンプル自動収集（save_pose_pair.py の自動化版）。

使い方:
  1. 手動（ジョグ）でアームを「マーカーが画面中央・距離0.3〜0.6mで写る基準姿勢」へ動かす
  2. リポジトリルートから実行:
       python xarm7/save_pose_pair_auto.py --num-samples 40
  3. スクリプトが基準姿勢の周囲に並進±trans-range-mm・回転±rot-range-degの
     目標姿勢を自動生成して順に移動し、マーカーが検出できた姿勢だけ保存する

出力JSONは save_pose_pair.py と同一形式なので、そのまま
calculate_Homogeneous_transformation_matrix.py の --in-json に渡せる。

安全上の注意:
  - 初回は必ず小さい範囲・低速・少数で試すこと:
      python xarm7/save_pose_pair_auto.py --trans-range-mm 30 --rot-range-deg 8 --speed 30 --num-samples 5
  - 実行中は非常停止ボタンに手が届く位置から離れないこと
  - 基準姿勢の周囲（±範囲＋余裕）に障害物（書架・治具）がないことを確認すること
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as Rot

from control.ar_marker_pose import rs_color_K_dist, aruco_marker_pose_target2cam


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def json_dump(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def recover_arm(arm, wait_sec: float = 0.5) -> None:
    """エラー・停止状態からモーション可能な状態へ復帰させる。"""
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(wait_sec)


def generate_targets(center: list[float], n: int,
                     trans_range_mm: float, rot_range_deg: float,
                     seed: int) -> list[list[float]]:
    """基準姿勢 center=[x,y,z,roll,pitch,yaw] の周囲に目標姿勢を n 個生成する。
    先頭は基準姿勢そのもの（オフセットゼロ）。"""
    rng = np.random.default_rng(seed)
    scale = np.array([trans_range_mm] * 3 + [rot_range_deg] * 3, dtype=np.float64)
    targets = [list(center)]
    for _ in range(n - 1):
        offset = rng.uniform(-1.0, 1.0, size=6) * scale
        targets.append([float(c + o) for c, o in zip(center, offset)])
    return targets


def generate_lookat_targets(center: list[float], rvec0, tvec0, handeye_json: str | Path,
                            n: int, cone_deg: float, roll_range_deg: float,
                            dist_jitter: float, seed: int) -> list[list[float]]:
    """マーカー注視姿勢の生成。

    基準姿勢で検出したマーカーの位置をロボット座標へ変換し、マーカーを頂点とする
    円錐内のランダムな方向・距離から「カメラがマーカー中心を向く」TCP姿勢を n 個作る。
    ハンドアイの並進推定には回転の多様性が不可欠なため、光軸まわりのロールも振る。
    既存のハンドアイ推定（handeye_json）を照準用に使う。数cmの誤差があっても
    マーカーが画角から外れない程度に写るので照準用途には十分。
    """
    d = json.loads(Path(handeye_json).read_text(encoding="utf-8"))
    # 歴史的経緯でキー名が実体と逆: "T_cam_tcp" キーが カメラ→TCP の変換行列 [m]
    HE = np.array(d["T_cam_tcp"], dtype=np.float64)
    HE_inv = np.linalg.inv(HE)

    x, y, z, roll, pitch, yaw = center
    T_base_tcp = np.eye(4)
    T_base_tcp[:3, :3] = Rot.from_euler("ZYX", [yaw, pitch, roll], degrees=True).as_matrix()
    T_base_tcp[:3, 3] = np.array([x, y, z]) / 1000.0

    Rct, _ = cv2.Rodrigues(np.asarray(rvec0, dtype=np.float64).reshape(3, 1))
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3] = Rct
    T_cam_marker[:3, 3] = np.asarray(tvec0, dtype=np.float64).reshape(3)

    T_base_cam0 = T_base_tcp @ HE
    p_m = (T_base_cam0 @ T_cam_marker)[:3, 3]   # マーカー中心（base座標, m）
    c0 = T_base_cam0[:3, 3]                     # 現在のカメラ位置（base座標, m）
    y0 = T_base_cam0[:3, 1]                     # 現在のカメラy軸（roll=0の基準）
    u0 = c0 - p_m
    d0 = float(np.linalg.norm(u0))
    u0 /= d0
    print(f"[INFO] marker position (base) = {np.round(p_m * 1000.0, 1)} mm, "
          f"camera distance = {d0 * 1000.0:.0f} mm")

    # u0（マーカー→カメラ方向）まわりの局所座標系
    tmp = np.array([0.0, 0.0, 1.0]) if abs(u0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u0, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(u0, e1)

    rng = np.random.default_rng(seed)
    cone = np.deg2rad(cone_deg)
    targets = [list(center)]
    while len(targets) < n:
        # 円錐内で一様な視点方向
        cos_t = rng.uniform(np.cos(cone), 1.0)
        sin_t = float(np.sqrt(1.0 - cos_t ** 2))
        phi = rng.uniform(0.0, 2 * np.pi)
        u = cos_t * u0 + sin_t * (np.cos(phi) * e1 + np.sin(phi) * e2)
        dist_m = d0 * (1.0 + rng.uniform(-dist_jitter, dist_jitter))
        c = p_m + dist_m * u

        # カメラ姿勢: z軸(光軸)をマーカーへ向け、y軸はなるべく現状に近く
        zax = p_m - c
        zax /= np.linalg.norm(zax)
        xax = np.cross(y0, zax)
        nx = float(np.linalg.norm(xax))
        if nx < 1e-6:
            continue
        xax /= nx
        yax = np.cross(zax, xax)
        # 光軸まわりのロール（回転多様性の主成分。マーカーは画面中央のまま）
        psi = np.deg2rad(rng.uniform(-roll_range_deg, roll_range_deg))
        R_base_cam = Rot.from_rotvec(psi * zax).as_matrix() @ np.column_stack([xax, yax, zax])

        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = R_base_cam
        T_base_cam[:3, 3] = c
        T_bt = T_base_cam @ HE_inv
        yaw_d, pitch_d, roll_d = Rot.from_matrix(T_bt[:3, :3]).as_euler("ZYX", degrees=True)
        p = T_bt[:3, 3] * 1000.0
        targets.append([float(p[0]), float(p[1]), float(p[2]),
                        float(roll_d), float(pitch_d), float(yaw_d)])
    return targets


def detect_marker_fresh(pipe, K, dist, args, n_flush: int = 5, n_tries: int = 5):
    """バッファに残った古いフレームを捨ててから検出を試みる。
    戻り値: (bgr, (marker_id, rvec, tvec)) / 検出失敗時は (bgr, None)"""
    for _ in range(n_flush):
        pipe.wait_for_frames()
    bgr = None
    for _ in range(n_tries):
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        bgr = np.asanyarray(color.get_data())
        ret = aruco_marker_pose_target2cam(
            bgr=bgr, K=K, dist=dist,
            marker_len_m=args.marker_len_m,
            dict_name=args.aruco_dict,
            target_id=args.marker_id,
        )
        if ret is not None:
            return bgr, ret
    return bgr, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xarm-host", type=str, default="192.168.2.221",
                    help="xArm IP (Retrieval_integration.yaml の robot.xarm.host と合わせる)")
    ap.add_argument("--out-dir", type=str, default="xarm7/handeye_pairs", help="output directory")
    ap.add_argument("--num-samples", type=int, default=40, help="number of samples to save")
    ap.add_argument("--marker-len-m", type=float, default=0.15, help="marker side length [m]")
    ap.add_argument("--marker-id", type=int, default=0, help="use specific marker id")
    ap.add_argument("--aruco-dict", type=str, default="DICT_4X4_1000", help="ArUco dict name")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--save-images", action="store_true", help="also save RGB images")
    # --- 自動化用パラメータ ---
    ap.add_argument("--mode", choices=["lookat", "perturb"], default="lookat",
                    help="lookat: マーカー注視姿勢を生成（推奨、回転多様性が大きい）/ "
                         "perturb: 基準姿勢に小オフセットを加える旧方式")
    ap.add_argument("--handeye-json", type=str,
                    default="xarm7/handeye_pairs/handeye_T_tcp_cam_20260604_204926 copy.json",
                    help="lookatモードの照準に使う既存ハンドアイ結果JSON")
    ap.add_argument("--cone-deg", type=float, default=30.0,
                    help="lookat: マーカーを頂点とする視点円錐の半頂角 [deg]")
    ap.add_argument("--roll-range-deg", type=float, default=45.0,
                    help="lookat: 光軸まわりロールの範囲 [±deg]")
    ap.add_argument("--dist-jitter", type=float, default=0.15,
                    help="lookat: マーカーまでの距離の変動割合 (0.15 = ±15%%)")
    ap.add_argument("--z-min-mm", type=float, default=None,
                    help="TCPの高さ下限 [mm]（既定: 基準姿勢のz - 150）")
    ap.add_argument("--trans-range-mm", type=float, default=60.0,
                    help="perturb: 基準姿勢からの並進オフセット範囲 [±mm]")
    ap.add_argument("--rot-range-deg", type=float, default=15.0,
                    help="perturb: 基準姿勢からの回転オフセット範囲 [±deg]")
    ap.add_argument("--speed", type=float, default=50.0, help="移動速度 [mm/s]")
    ap.add_argument("--mvacc", type=float, default=500.0,
                    help="移動加速度 [mm/s^2]（SDK既定2000は特異点近傍で速度超過エラーになりやすい）")
    ap.add_argument("--settle-sec", type=float, default=0.8, help="移動後の静定待ち時間 [s]")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="生成する候補姿勢数（0 なら num_samples の3倍）")
    ap.add_argument("--seed", type=int, default=0, help="姿勢生成の乱数シード（再現用）")
    ap.add_argument("--no-gui", action="store_true", help="プレビューウィンドウを出さない")
    ap.add_argument("--no-return", action="store_true", help="終了時に基準姿勢へ戻らない")
    args = ap.parse_args()

    n_candidates = args.max_candidates if args.max_candidates > 0 else args.num_samples * 3

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.save_images:
        img_dir.mkdir(parents=True, exist_ok=True)

    # --- RealSense start ---
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipe.start(cfg)
    intr, K, dist = rs_color_K_dist(profile)

    # --- xArm connect ---
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(args.xarm_host)
    arm.connect()
    if arm.error_code != 0:
        # 前回実行のエラーが残っているとき（例: 運動学エラー21）はクリアして開始
        print(f"[WARN] controller error {arm.error_code} remains -> clean_error")
        arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)

    session_id = now_stamp()
    json_path = out_dir / f"handeye_pairs_{session_id}.json"

    meta = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "marker_len_m": float(args.marker_len_m),
        "aruco_dict": args.aruco_dict,
        "marker_id_target": args.marker_id,
        "realsense": {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "model": str(intr.model),
            "coeffs": [float(x) for x in intr.coeffs],
            "K": K.tolist(),
            "dist5": dist.tolist(),
        },
        "xarm": {
            "host": args.xarm_host,
            "is_radian": False,
            "pos_unit": "mm",
            "angle_unit": "deg",
            "pose_order": ["x", "y", "z", "roll", "pitch", "yaw"],
        },
        "auto_collect": {
            "mode": args.mode,
            "cone_deg": args.cone_deg,
            "roll_range_deg": args.roll_range_deg,
            "dist_jitter": args.dist_jitter,
            "handeye_json_for_aiming": args.handeye_json if args.mode == "lookat" else None,
            "trans_range_mm": args.trans_range_mm,
            "rot_range_deg": args.rot_range_deg,
            "speed_mm_s": args.speed,
            "settle_sec": args.settle_sec,
            "seed": args.seed,
            "n_candidates": n_candidates,
        },
        "note": "Each sample has (tcp_pose in base frame) and (marker pose target->camera: rvec,tvec).",
    }

    samples = []
    center = None
    aborted = False

    try:
        # --- 基準姿勢の取得とマーカー確認 ---
        code, center = arm.get_position(is_radian=False)
        if code != 0 or center is None:
            raise RuntimeError(f"xArm get_position failed: code={code}")
        center = [float(v) for v in center]
        print(f"[INFO] center pose (base) = {['%.1f' % v for v in center]}")

        bgr, ret = detect_marker_fresh(pipe, K, dist, args)
        if ret is None:
            raise RuntimeError("基準姿勢でマーカーが検出できません。"
                               "マーカーが画面に写る姿勢へ動かしてから再実行してください。")
        print(f"[INFO] marker id={ret[0]} detected at center pose. start auto collection.")

        if args.mode == "lookat":
            mid0, rvec0, tvec0 = ret
            targets = generate_lookat_targets(
                center, rvec0, tvec0, args.handeye_json, n_candidates,
                cone_deg=args.cone_deg, roll_range_deg=args.roll_range_deg,
                dist_jitter=args.dist_jitter, seed=args.seed)
        else:
            targets = generate_targets(center, n_candidates,
                                       args.trans_range_mm, args.rot_range_deg, args.seed)

        # 高さ下限より低い候補は捨てる（机・治具への接近を防ぐ）
        z_min = args.z_min_mm if args.z_min_mm is not None else center[2] - 150.0
        n_before = len(targets)
        targets = [t for t in targets if t[2] >= z_min]
        if len(targets) < n_before:
            print(f"[INFO] {n_before - len(targets)} candidates below z_min={z_min:.0f}mm dropped")
        print(f"[INFO] mode={args.mode}  candidates={len(targets)}  goal={args.num_samples} samples")
        print("[INFO] abort: GUIで 'q' / ターミナルで Ctrl+C")

        for i, tgt in enumerate(targets):
            if len(samples) >= args.num_samples:
                break

            print(f"[MOVE] {i + 1}/{len(targets)} -> "
                  f"xyz=({tgt[0]:.1f},{tgt[1]:.1f},{tgt[2]:.1f}) "
                  f"rpy=({tgt[3]:.1f},{tgt[4]:.1f},{tgt[5]:.1f})")
            code = arm.set_position(x=tgt[0], y=tgt[1], z=tgt[2],
                                    roll=tgt[3], pitch=tgt[4], yaw=tgt[5],
                                    speed=args.speed, mvacc=args.mvacc,
                                    is_radian=False, wait=True)
            if code != 0:
                err = arm.error_code
                print(f"[WARN] set_position failed: code={code}, controller_error={err}, state={arm.state}")
                if err in (21, 24, 25):
                    # 21:運動学エラー(到達解なし) 24:速度超過(特異点近傍の経路)
                    # 25:計画エラー — いずれも「この候補姿勢が悪い」だけ。
                    # クリアして次の候補へ進む
                    print(f"[INFO] recoverable error {err} -> clean and skip this candidate")
                    recover_arm(arm)
                    continue
                if err != 0:
                    print(f"[ERROR] controller error {err} -> abort")
                    aborted = True
                    break
                if arm.state == 4:
                    # エラーコードなしの停止状態（速度超過停止の直後など）。
                    # 状態を戻さないと以降の指令が全て即失敗するため、ここで復帰させる
                    print("[INFO] arm in stop state -> recover and skip this candidate")
                    recover_arm(arm)
                    continue
                continue  # その他の失敗はスキップして次の候補へ

            time.sleep(args.settle_sec)
            bgr, ret = detect_marker_fresh(pipe, K, dist, args)

            if not args.no_gui and bgr is not None:
                vis = bgr.copy()
                if ret is not None:
                    mid, rvec, tvec = ret
                    cv2.drawFrameAxes(vis, K, dist, rvec.reshape(3, 1), tvec.reshape(3, 1), 0.05)
                    cv2.putText(vis, f"Marker: id={mid}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(vis, "Marker: not detected (skip)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(vis, f"Saved: {len(samples)}/{args.num_samples}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("handeye_auto_capture (q=quit)", vis)
                if (cv2.waitKey(10) & 0xFF) == ord("q"):
                    print("[INFO] aborted by user")
                    aborted = True
                    break

            if ret is None:
                print("[SKIP] marker not detected at this pose")
                continue

            # 実際に到達した姿勢を保存する（目標値ではなく）
            code, pose = arm.get_position(is_radian=False)
            if code != 0 or pose is None:
                print(f"[WARN] get_position failed: code={code} -> skip")
                continue

            mid, rvec, tvec = ret
            sid = now_stamp()
            img_path = None
            if args.save_images:
                img_path = str((img_dir / f"rgb_{len(samples):02d}_{sid}.png").resolve())
                cv2.imwrite(img_path, bgr)

            samples.append({
                "sample_id": sid,
                "marker": {
                    "id": int(mid),
                    "rvec_target2cam": [float(x) for x in rvec.reshape(3)],
                    "tvec_target2cam_m": [float(x) for x in tvec.reshape(3)],
                },
                "robot": {
                    "tcp_pose_base": [float(x) for x in pose],
                },
                "rgb_path": img_path,
            })
            # 逐次JSON保存（途中で落ちてもデータが残る）
            json_dump(json_path, {"meta": meta, "samples": samples})
            print(f"[SAVE] {len(samples)}/{args.num_samples}  id={mid}  json={json_path.name}")

        if len(samples) >= args.num_samples:
            print("[INFO] reached num_samples, done.")
        else:
            print(f"[INFO] finished with {len(samples)} samples "
                  f"(candidates exhausted or aborted)")

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt -> stop")
        aborted = True
    finally:
        try:
            if samples:
                json_dump(json_path, {"meta": meta, "samples": samples})
                print("[INFO] final json saved:", json_path.resolve())
        except Exception as e:
            print("[ERROR] final json save failed:", e)

        # 基準姿勢へ戻す（コントローラがエラー状態のときは動かさない）
        try:
            if (center is not None and not args.no_return
                    and arm.error_code == 0):
                print("[INFO] returning to center pose...")
                arm.set_position(x=center[0], y=center[1], z=center[2],
                                 roll=center[3], pitch=center[4], yaw=center[5],
                                 speed=args.speed, mvacc=args.mvacc,
                                 is_radian=False, wait=True)
        except Exception as e:
            print("[WARN] return-to-center failed:", e)

        try:
            pipe.stop()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        cv2.destroyAllWindows()

        if aborted:
            print("[INFO] aborted. 保存済みサンプルはそのまま利用できます:", json_path.name)


if __name__ == "__main__":
    main()
