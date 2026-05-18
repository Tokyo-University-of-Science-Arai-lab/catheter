#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RealSense 1ショット撮影 → SAM(infer_for_storage.infer_masks)で書籍領域認識
→ 対象書籍のRGB/Depth保存 → intrinsics+depthで点群化 → PLY保存
→ PCAで背表紙方向などの特徴量を計算

モジュールとしての使い方例:
  from rs_book_capture_and_pointcloud import run_capture_and_pca

  theta_rad, p_min, p_max = run_capture_and_pca(out_dir="captures")

スクリプトとしての使い方例:
  python rs_book_capture_and_pointcloud.py --out captures
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from PIL import Image
import json
import subprocess
import re
import traceback
from .OCR.only_one import find_similar_books
from .OCR.only_one_tilted import match_text_to_mask_main

# ===== RealSense =====
import pyrealsense2 as rs

# ===== SAM: infer_for_storage の infer_masks を使う =====
from .infer_for_storage import SamConfig, SamBatchInfer_storage, StageSaveCfg
from modules.overlay_io import _render_overlay_bgr, _save_points_and_overlay
#from bar_code.code_1_pic import detect_barcode
# 点群モジュール
from modules.pointcloud_utils import masked_depth_to_points, save_ply_ascii
from modules.calculate_3D_point_or_RANSAC import calculate_yaw
from modules.pca_vector import pca_axes_fix_dir
from modules.book_width import estimate_book_width
from modules.grip_point import find_target_point
from modules.open3d_view import visualize_points_and_target_open3d

ALLOWABLE_RANGE_Z = 0.07

def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    rs-viewer に近いフィルタ処理（あなたの depth_filter と同等）
    """
    decim = rs.decimation_filter()
    decim.set_option(rs.option.filter_magnitude, 2.0)

    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2.0)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20.0)

    hole_fill = rs.hole_filling_filter()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # viewer準拠: decimation は RGB 解像とズレる危険があるのでスキップ
    filtered = depth_to_disparity.process(raw_depth_frame)
    filtered = spat.process(filtered)
    filtered = disparity_to_depth.process(filtered)
    filtered = hole_fill.process(filtered)
    return filtered.as_depth_frame()


# def save_masked_and_cropped(
#     rgb_bgr: np.ndarray,
#     depth_u16: np.ndarray,
#     mask01: np.ndarray,
#     outdir: Path,
#     stem: str,
# ):
#     """
#     対象書籍のみの RGB/Depth を保存（背景0マスク）
#     """
#     outdir.mkdir(parents=True, exist_ok=True)

#     # --- マスク適用（背景=0） ---
#     rgb_masked = rgb_bgr.copy()
#     rgb_masked[mask01 == 0] = 0
#     depth_masked = depth_u16.copy()
#     depth_masked[mask01 == 0] = 0

#     cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
#     np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

#     # 深度可視化（0 以外の範囲を 0–255 に正規化）
#     nonzero = depth_masked[depth_masked > 0]
#     if nonzero.size > 0:
#         zmin, zmax = int(nonzero.min()), int(nonzero.max())
#         zrange = max(1, zmax - zmin)
#         depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
#         depth_vis[depth_masked > 0] = (
#             (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
#         ).astype(np.uint8)
#         cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

#     return depth_masked # 変更（追加）

def save_masked_and_cropped(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    outdir: Path,
    stem: str,
    z_tolerance_raw: int = 30,  # Z16値での許容幅（例: ±80カウント ≈ ±8cm 程度）
):
    """
    対象書籍のみの RGB/Depth を保存（背景0マスク ＋ 深度外れ値除去）
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # --- マスク適用（背景=0） ---
    rgb_masked = rgb_bgr.copy()
    rgb_masked[mask01 == 0] = 0

    depth_masked = depth_u16.copy()
    depth_masked[mask01 == 0] = 0  # まずマスク外を 0 にする

    # --- ここから: マスク内の depth の外れ値除去（1D簡易版 RANSAC 的なもの） ---
    # マスク内で depth が 0 でない画素だけ取り出す
    nonzero = depth_masked[depth_masked > 0]

    if nonzero.size > 0:
        # 主体（本）の「代表的な距離」を中央値で近似
        z_med = int(np.median(nonzero))

        # 中央値から外れすぎた depth を 0 にする
        #   -> 本より手前/奥にある別の物体を消す目的
        z_min_keep = z_med - z_tolerance_raw
        z_max_keep = z_med + z_tolerance_raw

        # 残したい領域（Trueが残す）
        keep = (depth_masked >= z_min_keep) & (depth_masked <= z_max_keep)

        # keep==False のところを 0 にする
        depth_masked[~keep] = 0

    # --- ファイル保存 ---
    cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
    np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

    # 深度可視化（0 以外の範囲を 0–255 に正規化） ※外れ値除去後の結果を可視化
    nonzero = depth_masked[depth_masked > 0]
    if nonzero.size > 0:
        zmin, zmax = int(nonzero.min()), int(nonzero.max())
        zrange = max(1, zmax - zmin)
        depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
        depth_vis[depth_masked > 0] = (
            (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
        ).astype(np.uint8)
        cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

    return depth_masked  # ここで外れ値除去済みの depth を返す

def capture_one_shot(pipe, cfg, align, shot_dir, *, stem: str, color_only: bool = False):
    profile = pipe.start(cfg)

    for _ in range(10):
        pipe.wait_for_frames()

    frames = pipe.wait_for_frames()

    if color_only:
        color_frame = frames.get_color_frame()
        color_np = np.asanyarray(color_frame.get_data())

        # 左右反転しないで保存
        cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)

        pipe.stop()
        return color_np, None, None, None

    # depthも使う場合（2回目）
    align_frames = align.process(frames)
    depth_frame = depth_filter_like_viewer(align_frames.get_depth_frame())
    color_frame = align_frames.get_color_frame()

    color_np = np.asanyarray(color_frame.get_data())
    depth_np_u16 = np.asanyarray(depth_frame.get_data())

    # 左右反転しないで保存
    cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)
    np.save(shot_dir / f"{stem}_depth.npy", depth_np_u16)

    dprof = rs.video_stream_profile(depth_frame.get_profile())
    intr = dprof.get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    pipe.stop()
    return color_np, depth_np_u16, intr, depth_scale #後で調べる
def run_ocr_subprocess(shot_dir: Path):
    # OCR用仮想環境の python へのパス（あなたの環境に合わせて変更）
    # Linux venv例: /home/book/venv/ocr/bin/python
    OCR_PY = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python"
    OCR_SCRIPT = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py"

    subprocess.run([OCR_PY, OCR_SCRIPT, str(shot_dir)], check=True)
    print(f"✔ OCR done: {shot_dir / 'ocr_result.json'}")
def start_ocr_subprocess(shot_dir: Path):
    """
    OCR を非同期で開始して Popen を返す。
    後で communicate() / wait() して終了を待つ。
    """
    OCR_PY = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python"
    OCR_SCRIPT = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py"

    proc = subprocess.Popen(
        [OCR_PY, OCR_SCRIPT, str(shot_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_ocr_subprocess(proc: subprocess.Popen, *, timeout: float | None = None) -> str:
    """
    OCR サブプロセスの終了待ち。
    正常終了しなければ例外を投げる。
    """
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        raise RuntimeError(f"OCR subprocess timeout.\n{stdout}")

    if proc.returncode != 0:
        raise RuntimeError(f"OCR subprocess failed (code={proc.returncode}).\n{stdout}")

    return stdout or ""


def merge_ocr_and_masks(
    query: str,
    masks,
    shot_dir: Path,
    interactive: bool = True,
    threshold: int = 40,
):
    """
    OCR 結果(json) と SAM マスクを統合して、選択マスクを返す。
    """
    results = match_text_to_mask_main(query, masks, shot_dir, threshold=threshold)

    book_name = results[0]["name"] if results else None

    sel_idx = None
    if not interactive:
        sel_idx = 1

    if book_name:
        m = re.search(r"(\d+)$", book_name)  # 末尾の連続数字
        if m:
            sel_idx = int(m.group(1))

    if sel_idx is None:
        raise RuntimeError("マスクIDが選べませんでした（OCR結果なし or 対応付け失敗）")

    sel_mask = masks[sel_idx - 1]
    mask01 = (np.asarray(sel_mask) > 0).astype(np.uint8)

    return {
        "results": results,
        "book_name": book_name,
        "sel_idx": sel_idx,
        "sel_mask": sel_mask,
        "mask01": mask01,
    }
def save_pointcloud_screenshot(
    pts_f: np.ndarray,
    target_point: np.ndarray,
    save_path: Path,
    show_window: bool = False,
) -> None:
    """
    点群とターゲット点を Open3D で描画し、そのスクリーンショットを保存する関数。
    既存の visualize_points_and_target_open3d には影響を与えない。

    Parameters
    ----------
    pts_f : (N, 3) np.ndarray
        点群 [m]
    target_point : (3,) np.ndarray
        ターゲット点 [m]
    save_path : Path
        画像の保存先パス (png など)
    show_window : bool
        True の場合はウィンドウを表示、False の場合は非表示でレンダリングのみ
    """
    try:
        pts = np.asarray(pts_f).reshape(-1, 3)
        tgt = np.asarray(target_point).reshape(3)

        # 点群
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # ターゲット点を小さな球で可視化
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(tgt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show_window)
        vis.add_geometry(pcd)
        vis.add_geometry(sphere)

        # 一度レンダリングしてからキャプチャ
        vis.poll_events()
        vis.update_renderer()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(save_path), do_render=True)

        vis.destroy_window()
        print(f"✔ Saved pointcloud screenshot: {save_path}")

    except Exception:
        # ここで落ちてもメイン処理に影響が出ないようにする
        traceback.print_exc()
        print("⚠ 点群スクリーンショット保存に失敗しました（処理は続行します）")

def run_capture_and_pca(
    query: str,
    out_dir: str | Path = "captures",
    #1280x720 固定
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        shot_dir = out_dir / ts
        shot_dir.mkdir(parents=True, exist_ok=True)

        pipe2 = rs.pipeline()
        cfg2 = rs.config()
        cfg2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg2.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        align2 = rs.align(rs.stream.color)

        # ===== 1) 撮影 =====
        capture_start = time.perf_counter()
        color_np, depth_np_u16, intr, depth_scale = capture_one_shot(
            pipe2, cfg2, align2, shot_dir, stem="after_init"
        )
        capture_end = time.perf_counter()
        print("Realsense captured.")
        print(f"[TIME] capture            : {capture_end - capture_start:.3f} sec")

        # ===== 1.5) カメラパラメータ保存 =====
        camera_json = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "depth_scale": float(depth_scale),
            "fps": int(fps),
        }

        camera_json_path = shot_dir / "camera_params.json"
        camera_json_path.write_text(
            json.dumps(camera_json, indent=2),
            encoding="utf-8"
        )
        print(f":heavy_check_mark: Saved camera params: {camera_json_path}")

        # ===== 2) OCR を先に非同期開始 =====
        ocr_start = time.perf_counter()
        ocr_proc = start_ocr_subprocess(shot_dir)
        print("[PARALLEL] OCR subprocess started.")

        # ===== 3) SAM 実行（OCR と並列） =====
        sam_start = time.perf_counter()

        sam_cfg = SamConfig(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            device=sam_device,
        )
        sam_runner = SamBatchInfer_storage(sam_cfg)

        # BGR（numpy） → RGB の PIL.Image に変換
        rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))

        # ステージ保存設定（after_nms / before_smooth など）
        stage_cfg = StageSaveCfg(out_dir=shot_dir)

        masks, sam_data = sam_runner.infer_masks(
            rgb_pil,
            stage_save=stage_cfg,
            stem_for_save="rgb",
        )

        sam_end = time.perf_counter()
        print(f"[TIME] SAM total          : {sam_end - sam_start:.3f} sec")

        # ===== 4) OCR 終了待ち =====
        ocr_stdout = wait_ocr_subprocess(ocr_proc, timeout=120.0)
        ocr_end = time.perf_counter()

        if ocr_stdout.strip():
            print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")

        print(f"[TIME] OCR wall           : {ocr_end - ocr_start:.3f} sec")
        print(f"[TIME] capture->SAM end   : {sam_end - capture_start:.3f} sec")
        print(f"[TIME] OCR start->end     : {ocr_end - ocr_start:.3f} sec")
        print(f"[TIME] SAM & OCR end           : {ocr_end - capture_start:.3f} sec")

        # ===== 5) OCR結果とSAM結果を統合 =====
        merge_start = time.perf_counter()

        merged = merge_ocr_and_masks(
            query=query,
            masks=masks,
            shot_dir=shot_dir,
            interactive=interactive,
            threshold=40,
        )

        sel_idx = merged["sel_idx"]
        sel_mask = merged["sel_mask"]
        mask01 = merged["mask01"]

        merge_end = time.perf_counter()
        print(f"[TIME] merge OCR+SAM      : {merge_end - merge_start:.3f} sec")
        print(f"[SAM] selected id = {sel_idx}, mask shape = {mask01.shape}")

        # ===== 6) 選択結果のオーバーレイ保存 =====
        _save_points_and_overlay(
            rgb_pil,
            [sel_mask],
            shot_dir,
            f"rgb_mask{sel_idx}_selected",
            draw_ids=False,
        )

        # ===== 7) 対象書籍のみの RGB/Depth を保存 =====
        depth_masked = save_masked_and_cropped(
            color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}"
        )

        # ===== 8) マスク + depth → カメラ座標点群へ変換 =====
        _3D_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
        yaw = _3D_info["yaw"]
        pts_f = _3D_info["points"]  # (N,3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_f)

        # ===== 9) 点群を保存 =====
        save_ply_ascii(shot_dir / "pointcloud.ply", pts_f, None)

        # ===== 10) PCA =====
        mean, pc1, pc2 = pca_axes_fix_dir(pts_f)

        vx, vy = float(pc1[0]), float(pc1[1])
        norm_xy = float(np.hypot(vx, vy))
        if norm_xy < 1e-8:
            theta_rad = 0.0
        else:
            theta_rad = float(np.arctan2(vy, vx))  # 基準は +x 軸

        book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
        book_width = book_width_info.get("av_book_width_m")

        # ===== 11) 把持位置導出 =====
        target_point_info = find_target_point(pts_f)
        target_point = target_point_info.get("target_m")

        # ===== 12) 可視化 =====
        visualize_points_and_target_open3d(pts_f, target_point)

        # ===== 12.5) 点群ビューのスクリーンショット保存 =====
        vis_img_path = shot_dir / "pointcloud_view.png"
        save_pointcloud_screenshot(
            pts_f=pts_f,
            target_point=target_point,
            save_path=vis_img_path,
            show_window=False,
        )

        # ===== 13) PCA結果保存 =====
        print(":heavy_check_mark: PCA result:")
        print(f"  theta_rad = {theta_rad:.6f}")
        print(f"  p_min = {target_point}")
        print(":heavy_check_mark: Files saved under:", shot_dir)

        pca_json = {
            "theta_rad": float(theta_rad),
            "theta_deg": float(np.degrees(theta_rad)),
            "p_min_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
            "book_width_mm": float(book_width * 1000.0),  # book_width は [m] 想定
        }

        json_path = Path(shot_dir) / "pca_result.json"
        json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f":heavy_check_mark: Saved PCA JSON: {json_path}")

        return theta_rad, target_point, book_width * 1000.0, shot_dir  # m → mm

    except Exception:
        traceback.print_exc()
        print("認識失敗！！")
        return None, None, None, None
    
def run_capture_and_pca_offline(
    query: str,
    shot_dir: str | Path,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, Path]:
    """
    すでに撮影済みの画像・深度を用いて、
    run_capture_and_pca と同じ処理フロー（SAM→OCR→点群→PCA）を行うオフライン版。
    """
    shot_dir = Path(shot_dir)

    rgb_path = shot_dir / "after_init_rgb.png"
    depth_path = shot_dir / "after_init_depth.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(f"{rgb_path} がありません")
    if not depth_path.exists():
        raise FileNotFoundError(f"{depth_path} がありません")

    color_np = cv2.imread(str(rgb_path))  # BGR
    depth_np_u16 = np.load(depth_path)    # (H,W) uint16

    intr = rs.intrinsics()
    intr.width = 1280
    intr.height = 720
    intr.fx = 908.1617431640625
    intr.fy = 906.4829711914062
    intr.ppx = 637.79833984375
    intr.ppy = 371.0213928222656

    depth_scale = 0.0010000000474974513

    # ===== 1) OCR を先に非同期開始 =====
    ocr_start = time.perf_counter()
    ocr_proc = start_ocr_subprocess(shot_dir)
    print("[PARALLEL][OFFLINE] OCR subprocess started.")

    # ===== 2) SAM 実行（OCR と並列） =====
    sam_start = time.perf_counter()

    sam_cfg = SamConfig(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        device=sam_device,
    )
    sam_runner = SamBatchInfer_storage(sam_cfg)

    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    stage_cfg = StageSaveCfg(out_dir=shot_dir)

    masks, sam_data = sam_runner.infer_masks(
        rgb_pil,
        stage_save=stage_cfg,
        stem_for_save="rgb",
    )

    sam_end = time.perf_counter()
    print(f"[TIME][OFFLINE] SAM total     : {sam_end - sam_start:.3f} sec")

    # ===== 3) OCR 終了待ち =====
    ocr_stdout = wait_ocr_subprocess(ocr_proc, timeout=120.0)
    ocr_end = time.perf_counter()

    if ocr_stdout.strip():
        print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")

    print(f"[TIME][OFFLINE] OCR wall      : {ocr_end - ocr_start:.3f} sec")

    # ===== 4) 統合 =====
    merge_start = time.perf_counter()

    merged = merge_ocr_and_masks(
        query=query,
        masks=masks,
        shot_dir=shot_dir,
        interactive=interactive,
        threshold=40,
    )

    sel_idx = merged["sel_idx"]
    sel_mask = merged["sel_mask"]
    mask01 = merged["mask01"]

    merge_end = time.perf_counter()
    print(f"[TIME][OFFLINE] merge OCR+SAM: {merge_end - merge_start:.3f} sec")
    print(f"[SAM OFFLINE] selected id = {sel_idx}, mask shape = {mask01.shape}")

    # ===== 5) 選択結果保存 =====
    _save_points_and_overlay(
        rgb_pil,
        [sel_mask],
        shot_dir,
        f"rgb_mask{sel_idx}_selected_offline",
        draw_ids=False,
    )

    # ===== 6) マスクDepth保存 =====
    depth_masked = save_masked_and_cropped(
        color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}_offline"
    )

    # ===== 7) 点群化 =====
    _3D_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
    yaw = _3D_info["yaw"]
    pts_f = _3D_info["points"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_f)

    save_ply_ascii(shot_dir / "pointcloud_offline.ply", pts_f, None)

    # ===== 8) PCA =====
    mean, pc1, pc2 = pca_axes_fix_dir(pts_f)
    vx, vy = float(pc1[0]), float(pc1[1])
    norm_xy = float(np.hypot(vx, vy))
    if norm_xy < 1e-8:
        theta_rad = 0.0
    else:
        theta_rad = float(np.arctan2(vy, vx))

    book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
    book_width = book_width_info.get("av_book_width_m")

    # ===== 9) 把持位置 =====
    target_point_info = find_target_point(pts_f)
    target_point = target_point_info.get("target_m")

    # ===== 10) 可視化 =====
    visualize_points_and_target_open3d(pts_f, target_point)

    vis_img_path = shot_dir / "pointcloud_view_offline.png"
    save_pointcloud_screenshot(
        pts_f=pts_f,
        target_point=target_point,
        save_path=vis_img_path,
        show_window=False,
    )

    print(":heavy_check_mark: [OFFLINE] PCA result:")
    print(f"  theta_rad = {theta_rad:.6f}")
    print(f"  p_min = {target_point}")
    print(":heavy_check_mark: Files saved under:", shot_dir)

    pca_json = {
        "theta_rad": float(theta_rad),
        "theta_deg": float(np.degrees(theta_rad)),
        "p_min_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
        "book_width_mm": float(book_width * 1000.0),
    }
    json_path = shot_dir / "pca_result_offline.json"
    json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f":heavy_check_mark: Saved PCA JSON (offline): {json_path}")

    return theta_rad, target_point, book_width * 1000.0, shot_dir


    
def main():
    import pyrealsense2 as rs
    import cv2
    from pathlib import Path
    from datetime import datetime

    # ========= 設定 =========
    width = 1280
    height = 720
    fps = 1

    # 保存先: captures/YYYY-MM-DD
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = Path("./captures") / today_str
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========= RealSense 初期化 =========
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipe.start(cfg)

    try:
        print("カメラ起動中...")
        print("s: 保存")
        print("q or ESC: 終了")

        # 最初の数フレームは捨てる（露出安定用）
        for _ in range(30):
            pipe.wait_for_frames()

        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = cv2.cvtColor(
                cv2.imread("/dev/null") if False else
                __import__("numpy").asanyarray(color_frame.get_data()),
                cv2.COLOR_RGB2BGR
            ) if False else __import__("numpy").asanyarray(color_frame.get_data())

            # 表示用
            preview = color_image.copy()
            cv2.putText(
                preview,
                f"{width}x{height}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                preview,
                "Press 's' to save / 'q' or ESC to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("RealSense Capture", preview)
            key = cv2.waitKey(1) & 0xFF

            # sキーで保存
            if key == ord("s"):
                now = datetime.now().strftime("%H-%M-%S")
                filename = f"{width}x{height}_{now}.png"
                filepath = save_dir / filename

                cv2.imwrite(str(filepath), color_image)
                print(f"保存: {filepath}")

            # q or ESCで終了
            elif key == ord("q") or key == 27:
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", type=str, default="captures", help="撮影と結果保存のベースフォルダ")
#     ap.add_argument("--w", type=int, default=1280)
#     ap.add_argument("--h", type=int, default=720)
#     ap.add_argument("--fps", type=int, default=6)
#     ap.add_argument("--sam_device", choices=["gpu", "cpu", "auto"], default="gpu")
#     ap.add_argument(
#         "--encoder",
#         default="/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
#     )
#     ap.add_argument(
#         "--decoder",
#         default="/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
#     )
#     args = ap.parse_args()

#     theta_rad, p_min, book_width, yaw = run_capture_and_pca(
#         query="独立行政法人",
#         out_dir=args.out,
#         width=args.w,
#         height=args.h,
#         fps=args.fps,
#         sam_device=args.sam_device,
#         encoder_path=args.encoder,
#         decoder_path=args.decoder,
#         interactive=True,
#     )

#     print("\n=== Summary ===")
#     print(f"book width = {book_width}")
#     print(f"roll (deg) = {np.degrees(theta_rad):.6f}")
#     print(f"p_min = {p_min}")

#     #print("yaw (deg) = {:.3f}".format(np.degrees(yaw)))
#     print("===============")
    

if __name__ == "__main__":
    main()
