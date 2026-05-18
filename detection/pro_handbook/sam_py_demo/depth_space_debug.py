#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    RealSense Viewerに近いDepthフィルタ処理．
    RGBとDepthの画素対応を崩したくないため，decimationは使わない．
    """
    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2.0)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20.0)

    hole_fill = rs.hole_filling_filter()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    filtered = depth_to_disparity.process(raw_depth_frame)
    filtered = spat.process(filtered)
    filtered = disparity_to_depth.process(filtered)
    filtered = hole_fill.process(filtered)

    return filtered.as_depth_frame()


def save_depth_visualization(
    depth_u16: np.ndarray,
    depth_scale: float,
    save_path: Path,
    min_m: float | None = None,
    max_m: float | None = None,
) -> None:
    """
    Depth画像を見やすい疑似カラー画像として保存する．
    """
    depth_m = depth_u16.astype(np.float32) * depth_scale

    valid = depth_m > 0
    if not np.any(valid):
        print("[WARN] valid depth is empty.")
        return

    if min_m is None:
        min_m = float(np.percentile(depth_m[valid], 2))
    if max_m is None:
        max_m = float(np.percentile(depth_m[valid], 98))

    depth_clip = np.clip(depth_m, min_m, max_m)
    depth_norm = ((depth_clip - min_m) / max(max_m - min_m, 1e-6) * 255.0).astype(np.uint8)

    depth_norm[~valid] = 0
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    cv2.imwrite(str(save_path), depth_color)
    print(f"[SAVE] depth visualization: {save_path}")


def save_pointcloud_ply_ascii(
    save_path: Path,
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
) -> None:
    """
    点群をASCII PLYとして保存する．
    points_xyz: (N, 3) [m]
    colors_rgb: (N, 3) uint8, RGB
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)

    if colors_rgb is not None:
        colors_rgb = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
        assert len(points_xyz) == len(colors_rgb)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors_rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        if colors_rgb is None:
            for x, y, z in points_xyz:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(points_xyz, colors_rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    print(f"[SAVE] pointcloud: {save_path}")


def depth_to_pointcloud(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    stride: int = 2,
    z_min_m: float = 0.10,
    z_max_m: float = 1.50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    aligned RGB-Dから点群を生成する．
    strideを大きくすると点数が減って軽くなる．
    """
    h, w = depth_u16.shape[:2]

    points = []
    colors = []

    for v in range(0, h, stride):
        for u in range(0, w, stride):
            d = int(depth_u16[v, u])
            if d == 0:
                continue

            z = float(d) * depth_scale
            if z < z_min_m or z > z_max_m:
                continue

            x, y, z = rs.rs2_deproject_pixel_to_point(
                intr,
                [float(u), float(v)],
                z,
            )

            b, g, r = color_bgr[v, u]
            points.append([x, y, z])
            colors.append([r, g, b])

    if len(points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(colors, dtype=np.uint8),
    )


def compute_depth_profile(
    depth_u16: np.ndarray,
    depth_scale: float,
    *,
    y_center: int = 230,
    scan_radius: int = 80,
    min_valid_ratio: float = 0.20,
) -> dict:
    """
    書籍収納スペース検証用．
    指定した高さ帯 y_center ± scan_radius に対して，
    x方向のDepth中央値プロファイルを作る．

    戻り値：
      depth_profile_m[x] : 各x列のDepth中央値 [m]
      valid_ratio[x]     : 各x列でDepthが取れている画素割合
      y0, y1             : 使用した帯領域
    """
    h, w = depth_u16.shape[:2]

    y0 = max(0, y_center - scan_radius)
    y1 = min(h, y_center + scan_radius)

    band = depth_u16[y0:y1, :]
    band_valid = band > 0

    depth_profile_m = np.full((w,), np.nan, dtype=np.float32)
    valid_ratio = np.zeros((w,), dtype=np.float32)

    band_h = band.shape[0]

    for x in range(w):
        col = band[:, x]
        valid_col = col[col > 0]

        valid_ratio[x] = len(valid_col) / max(band_h, 1)

        if valid_ratio[x] >= min_valid_ratio:
            depth_profile_m[x] = float(np.median(valid_col)) * depth_scale

    return {
        "depth_profile_m": depth_profile_m,
        "valid_ratio": valid_ratio,
        "y0": y0,
        "y1": y1,
    }


def save_depth_profile_plot_image(
    rgb_bgr: np.ndarray,
    profile_info: dict,
    save_path: Path,
    *,
    delta_depth_m: float = 0.03,
) -> None:
    """
    Depthプロファイルを簡易的に画像として保存する．
    matplotlibを使わず，OpenCVで見やすい確認画像を作る．
    """
    depth_profile = profile_info["depth_profile_m"]
    valid_ratio = profile_info["valid_ratio"]
    y0 = profile_info["y0"]
    y1 = profile_info["y1"]

    h, w = rgb_bgr.shape[:2]

    vis = rgb_bgr.copy()

    # 使用した帯領域を描画
    cv2.rectangle(vis, (0, y0), (w - 1, y1), (0, 255, 255), 2)

    valid = np.isfinite(depth_profile)
    if not np.any(valid):
        cv2.imwrite(str(save_path), vis)
        print("[WARN] profile has no valid depth.")
        return

    # 書籍面の基準Depthを暫定的に中央値で置く
    book_depth_ref = float(np.nanmedian(depth_profile[valid]))

    # 奥に抜けている候補
    far = valid & (depth_profile > book_depth_ref + delta_depth_m)

    # 欠損が多い候補
    invalid_like = valid_ratio < 0.10

    # far領域を画像上に描画
    for x in np.where(far)[0]:
        cv2.line(vis, (x, y0), (x, y1), (0, 0, 255), 1)

    # 欠損っぽい領域を青で描画
    for x in np.where(invalid_like)[0]:
        cv2.line(vis, (x, y0), (x, y1), (255, 0, 0), 1)

    text1 = f"book_depth_ref={book_depth_ref:.3f} m"
    text2 = f"red: depth > ref + {delta_depth_m:.3f} m, blue: low valid ratio"

    cv2.putText(
        vis,
        text1,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        vis,
        text2,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] depth profile overlay: {save_path}")


def save_depth_profile_csv(
    profile_info: dict,
    save_path: Path,
) -> None:
    depth_profile = profile_info["depth_profile_m"]
    valid_ratio = profile_info["valid_ratio"]

    xs = np.arange(len(depth_profile), dtype=np.int32)

    data = np.stack(
        [
            xs.astype(np.float32),
            depth_profile.astype(np.float32),
            valid_ratio.astype(np.float32),
        ],
        axis=1,
    )

    header = "x,depth_m,valid_ratio"
    np.savetxt(
        save_path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt=["%d", "%.6f", "%.6f"],
    )
    print(f"[SAVE] depth profile csv: {save_path}")


def capture_realsense_once(
    *,
    out_dir: str | Path = "captures_depth_debug",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    y_center: int = 230,
    scan_radius: int = 80,
    rotate_180: bool = True,
    pointcloud_stride: int = 2,
) -> None:
    out_dir = Path(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    shot_dir = out_dir / ts
    shot_dir.mkdir(parents=True, exist_ok=True)

    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)

        # 起動直後の不安定フレームを捨てる
        for _ in range(30):
            pipe.wait_for_frames()

        frames = pipe.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("color_frame or depth_frame is empty.")

        depth_frame = depth_filter_like_viewer(depth_frame)

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_u16 = np.asanyarray(depth_frame.get_data())

        # intrinsicsはalign後のcolor座標系で使う
        color_prof = rs.video_stream_profile(
            profile.get_stream(rs.stream.color)
        )
        intr = color_prof.get_intrinsics()
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

        print(
            f"[INFO] intrinsics: fx={intr.fx:.3f}, fy={intr.fy:.3f}, "
            f"ppx={intr.ppx:.3f}, ppy={intr.ppy:.3f}"
        )
        print(f"[INFO] depth_scale={depth_scale}")

        if rotate_180:
            color_bgr = cv2.rotate(color_bgr, cv2.ROTATE_180)
            depth_u16 = cv2.rotate(depth_u16, cv2.ROTATE_180)

            # 注意：
            # ここでは検証用として回転後画像を保存・可視化する．
            # 点群化には，回転後Depthと同じ画素座標を使うため，
            # 本来はintrinsicsの主点も回転後に合わせる必要がある．
            # ただし，簡易確認なら画像として見やすい向き優先でOK．
            # 厳密な3D点群確認をする場合は rotate_180=False を推奨．
            print("[WARN] rotate_180=True: pointcloud geometry is approximate unless intrinsics are adjusted.")
            print("[WARN] For strict 3D verification, use rotate_180=False.")

        # 保存
        rgb_path = shot_dir / "rgb.png"
        depth_npy_path = shot_dir / "depth_u16.npy"
        depth_csv_path = shot_dir / "depth_u16.csv"
        depth_vis_path = shot_dir / "depth_vis.png"

        cv2.imwrite(str(rgb_path), color_bgr)
        np.save(depth_npy_path, depth_u16)
        np.savetxt(depth_csv_path, depth_u16, delimiter=",", fmt="%d")

        print(f"[SAVE] rgb: {rgb_path}")
        print(f"[SAVE] depth npy: {depth_npy_path}")
        print(f"[SAVE] depth csv: {depth_csv_path}")

        save_depth_visualization(
            depth_u16,
            depth_scale,
            depth_vis_path,
        )

        # Depth profile
        profile_info = compute_depth_profile(
            depth_u16,
            depth_scale,
            y_center=y_center,
            scan_radius=scan_radius,
        )

        save_depth_profile_csv(
            profile_info,
            shot_dir / "depth_profile.csv",
        )

        save_depth_profile_plot_image(
            color_bgr,
            profile_info,
            shot_dir / "depth_profile_overlay.png",
            delta_depth_m=0.03,
        )

        # 点群保存
        points_xyz, colors_rgb = depth_to_pointcloud(
            color_bgr,
            depth_u16,
            intr,
            depth_scale,
            stride=pointcloud_stride,
            z_min_m=0.10,
            z_max_m=1.50,
        )

        save_pointcloud_ply_ascii(
            shot_dir / "pointcloud.ply",
            points_xyz,
            colors_rgb,
        )

        print("\n========== SUMMARY ==========")
        print(f"save dir       : {shot_dir}")
        print(f"rgb            : {rgb_path.name}")
        print(f"depth vis      : {depth_vis_path.name}")
        print(f"profile image  : depth_profile_overlay.png")
        print(f"profile csv    : depth_profile.csv")
        print(f"pointcloud     : pointcloud.ply")
        print(f"points         : {len(points_xyz)}")
        print("=============================\n")

    finally:
        try:
            pipe.stop()
        except Exception:
            pass


def main():
    capture_realsense_once(
        out_dir="captures_depth_debug",
        width=1280,
        height=720,
        fps=6,
        y_center=230,
        scan_radius=80,
        rotate_180=False,
        pointcloud_stride=2,
    )


if __name__ == "__main__":
    main()