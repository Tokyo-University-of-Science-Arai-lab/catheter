#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d
from pathlib import Path
import numpy as np


def load_and_pick_ply(ply_path: str):
    """
    pointcloud.ply を表示し，クリックした点の3次元座標を表示する．

    操作:
      Shift + 左クリック : 点を選択
      Shift + 右クリック : 選択解除
      q                 : 終了して選択点を表示
    """
    ply_path = Path(ply_path)

    if not ply_path.exists():
        print(f"❌ ファイルが見つかりません: {ply_path}")
        return

    print(f"📂 Loading: {ply_path}")

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)

    print(f"✔ Loaded point cloud, num points = {len(points)}")
    print("")
    print("===== 操作方法 =====")
    print("Shift + 左クリック : 点を選択")
    print("Shift + 右クリック : 選択解除")
    print("q                 : 終了")
    print("====================")
    print("")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick points from pointcloud")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    picked_indices = vis.get_picked_points()

    print("")
    print("===== 選択された点 =====")

    if len(picked_indices) == 0:
        print("点は選択されませんでした．")
        return

    for i, idx in enumerate(picked_indices):
        x, y, z = points[idx]

        print(f"[{i}] index = {idx}")
        print(f"    x = {x:.6f} m")
        print(f"    y = {y:.6f} m")
        print(f"    z = {z:.6f} m")
        print(f"    z = {z * 1000:.2f} mm")
        print("")

    print("=======================")


if __name__ == "__main__":
    ply_file = "/home/book/pro_book/pro_hand_book_python/captures/20260409_165212/pointcloud.ply"
    load_and_pick_ply(ply_file)