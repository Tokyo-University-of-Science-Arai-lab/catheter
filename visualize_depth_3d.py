#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
captures/日時/ フォルダ内の対象マスク深度を open3d で3D可視化するスクリプト。
mask*_depth_prefilter_depth_masked.npy + after_init_rgb.png から点群を生成する。

使い方:
    python visualize_depth_3d.py captures/20260707_125829
    python visualize_depth_3d.py   # 最新フォルダを自動選択

操作:
    マウスドラッグ        : 回転
    Ctrl + ドラッグ      : 平行移動
    スクロール           : ズーム
    q                   : 終了
"""

import sys
import json
import numpy as np
from pathlib import Path
import cv2
import open3d as o3d


def find_book_depth(shot_dir: Path) -> Path:
    """mask*_depth_prefilter_depth_masked.npy を探して返す。なければ None。"""
    candidates = sorted(shot_dir.glob("mask*_depth_prefilter_depth_masked.npy"))
    return candidates[0] if candidates else None


def build_pointcloud(shot_dir: Path) -> o3d.geometry.PointCloud:
    with open(shot_dir / "camera_params.json") as f:
        p = json.load(f)
    fx  = p["fx"]
    fy  = p["fy"]
    ppx = p["ppx"]
    ppy = p["ppy"]
    ds  = p["depth_scale"]

    npy_path = find_book_depth(shot_dir)
    if npy_path is None:
        print("警告: mask*_depth_prefilter_depth_masked.npy が見つかりません。全体深度で代替します。")
        npy_path = shot_dir / "after_init_depth.npy"
    else:
        print(f"深度ファイル: {npy_path.name}")

    depth_raw = np.load(npy_path)          # uint16, shape (H, W)
    depth_m   = (depth_raw.astype(np.float32) * ds)  # メートル単位 float32

    rgb_path = shot_dir / "after_init_rgb.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB画像が見つかりません: {rgb_path}")
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

    H, W = depth_raw.shape

    color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_m)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,            # depth_m はすでにメートル
        depth_trunc=2.0,            # 2m 以内を有効とする
        convert_rgb_to_intensity=False,
    )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=W, height=H,
        fx=fx, fy=fy,
        cx=ppx, cy=ppy,
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return pcd


def main():
    captures_root = Path(__file__).parent / "captures"

    if len(sys.argv) > 1:
        shot_dir = Path(sys.argv[1])
    else:
        dirs = sorted([d for d in captures_root.iterdir()
                       if d.is_dir() and (d / "after_init_depth.npy").exists()])
        if not dirs:
            print("キャプチャフォルダが見つかりません")
            return
        shot_dir = dirs[-1]

    print(f"読み込み: {shot_dir}")
    pcd = build_pointcloud(shot_dir)
    print(f"点数: {len(pcd.points)}")

    print()
    print("===== 操作方法 =====")
    print("マウスドラッグ        : 回転")
    print("Ctrl + ドラッグ      : 平行移動")
    print("スクロール            : ズーム")
    print("q                    : 終了")
    print("====================")
    print()

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=f"点群: {shot_dir.name}",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
