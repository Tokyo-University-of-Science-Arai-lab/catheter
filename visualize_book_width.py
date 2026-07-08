#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
captures フォルダ内の書籍マスク点群と幅推定方向を open3d で3D表示するスクリプト。

色の凡例:
  ライムグリーン : pc1  本の長手方向（高さ）
  オレンジ      : pc2  Method 2 の幅方向ベクトル
  オレンジ スパン: Method 2 の測定幅（端点＝球マーカー）
  水色          : Method 1 の幅方向ベクトル（画像平面X → 3D変換）
  水色 スパン   : Method 1 の測定幅（端点＝球マーカー）
  黄色球        : 点群の重心（mean）

使い方:
    python visualize_book_width.py captures/20260707_112626
    python visualize_book_width.py   # 最新フォルダを自動選択
"""

import sys
import json
import numpy as np
from pathlib import Path
import cv2
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent / "detection/pro_handbook/sam_py_demo"))
from modules.pca_vector import pca_axes_fix_dir
from modules.book_width import estimate_book_width

# ── 色定義 ──────────────────────────────────────────────────────────────────
C_PC1_FACE   = [0.30, 1.00, 0.30]   # ライムグリーン
C_PC1_EDGE   = [0.00, 0.55, 0.00]
C_PC2_FACE   = [1.00, 0.55, 0.00]   # オレンジ
C_PC2_EDGE   = [0.70, 0.30, 0.00]
C_PC2S_FACE  = [1.00, 0.75, 0.00]   # 薄オレンジ（スパン）
C_PC2S_EDGE  = [0.70, 0.45, 0.00]
C_M1_FACE    = [0.00, 0.80, 1.00]   # 水色
C_M1_EDGE    = [0.00, 0.40, 0.70]
C_M1S_FACE   = [0.40, 0.95, 1.00]   # 薄水色（スパン）
C_M1S_EDGE   = [0.00, 0.55, 0.75]
C_MEAN       = [1.00, 0.95, 0.00]   # 黄色（重心マーカー）
C_MEAN_EDGE  = [0.60, 0.55, 0.00]


# ── 回転行列（+Z → direction） ───────────────────────────────────────────────

def _rot_z_to(direction):
    z = np.array([0.0, 0.0, 1.0])
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    axis  = np.cross(z, d)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(z, d))
    if sin_a < 1e-8:
        return np.eye(3) if cos_a > 0 else -np.eye(3)
    axis /= sin_a
    K = np.array([[0, -axis[2],  axis[1]],
                  [axis[2],  0, -axis[0]],
                  [-axis[1], axis[0],  0]])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def _transform(mesh, R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    mesh.transform(T)
    return mesh


# ── メッシュ生成ヘルパー ──────────────────────────────────────────────────────

def mesh_with_edges(mesh, face_color, edge_color):
    """メッシュに塗りつぶし色を付け、ワイヤーフレームエッジと一緒に返す。"""
    mesh.paint_uniform_color(face_color)
    mesh.compute_vertex_normals()
    edges = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    edges.paint_uniform_color(edge_color)
    return mesh, edges


def arrow(origin, direction, length, face_color, edge_color, r=0.00025, res=20):
    """矢印（円柱＋コーン）+ エッジワイヤーフレーム。"""
    d = np.asarray(direction, dtype=np.float64)
    d /= np.linalg.norm(d)
    m = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=r,
        cone_radius=r * 3.0,
        cylinder_height=length * 0.82,
        cone_height=length * 0.18,
        resolution=res,
        cylinder_split=1,
        cone_split=1,
    )
    _transform(m, _rot_z_to(d), np.asarray(origin, dtype=np.float64))
    return mesh_with_edges(m, face_color, edge_color)


def span(start, end, face_color, edge_color, r=0.00060, res=20):
    """start-end 間を示す円柱スパン + エッジ + 端点球マーカー。"""
    s = np.asarray(start, dtype=np.float64)
    e = np.asarray(end,   dtype=np.float64)
    vec    = e - s
    length = float(np.linalg.norm(vec))
    if length < 1e-9:
        return []
    d = vec / length

    # 円柱（Z中心 → 平行移動で start-end の中間へ）
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=r, height=length, resolution=res, split=1)
    R = _rot_z_to(d)
    mid = (s + e) / 2.0
    _transform(cyl, R, mid)
    cyl_m, cyl_e = mesh_with_edges(cyl, face_color, edge_color)

    # 端点キャップ（スパン方向に垂直な薄いディスク）
    def _cap(center):
        disc = o3d.geometry.TriangleMesh.create_cylinder(
            radius=r * 5.0, height=r * 0.8, resolution=res, split=1)
        _transform(disc, R, np.asarray(center, dtype=np.float64))
        return mesh_with_edges(disc, face_color, edge_color)

    cap1_m, cap1_e = _cap(s)
    cap2_m, cap2_e = _cap(e)
    return [cyl_m, cyl_e, cap1_m, cap1_e, cap2_m, cap2_e]


def sphere_marker(center, face_color, edge_color, r=0.0015, res=14):
    """点マーカー用の球 + エッジ。"""
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=res)
    sp.translate(np.asarray(center, dtype=np.float64))
    return mesh_with_edges(sp, face_color, edge_color)


# ── データ読み込み ────────────────────────────────────────────────────────────

def find_book_depth(shot_dir):
    cands = sorted(shot_dir.glob("mask*_depth_prefilter_depth_masked.npy"))
    return cands[0] if cands else None


def load_shot(shot_dir):
    with open(shot_dir / "camera_params.json") as f:
        p = json.load(f)
    npy = find_book_depth(shot_dir)
    if npy is None:
        raise FileNotFoundError("mask*_depth_prefilter_depth_masked.npy が見つかりません")
    depth_raw = np.load(npy)
    rgb = cv2.cvtColor(cv2.imread(str(shot_dir / "after_init_rgb.png")),
                       cv2.COLOR_BGR2RGB)
    return depth_raw, rgb, p["fx"], p["fy"], p["ppx"], p["ppy"], p["depth_scale"]


def depth_to_pts_rgb(depth_raw, rgb, fx, fy, ppx, ppy, ds):
    H, W = depth_raw.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    Z = depth_raw.astype(np.float32) * float(ds)
    v = Z > 0
    Zv = Z[v]
    pts = np.stack([(uu[v] - ppx) / fx * Zv,
                    (vv[v] - ppy) / fy * Zv, Zv], axis=1)
    cols = rgb[vv.astype(int)[v], uu.astype(int)[v]] / 255.0
    return pts, cols


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    captures_root = Path(__file__).parent / "captures"
    if len(sys.argv) > 1:
        shot_dir = Path(sys.argv[1])
    else:
        dirs = sorted([d for d in captures_root.iterdir()
                       if d.is_dir() and (d / "after_init_depth.npy").exists()])
        if not dirs:
            print("キャプチャフォルダが見つかりません"); return
        shot_dir = dirs[-1]

    print(f"読み込み: {shot_dir}")
    depth_raw, rgb, fx, fy, ppx, ppy, ds = load_shot(shot_dir)
    pts, colors = depth_to_pts_rgb(depth_raw, rgb, fx, fy, ppx, ppy, ds)
    print(f"点数: {len(pts):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors  = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # ── PCA ─────────────────────────────────────────────────────────────────
    mean, pc1, pc2 = pca_axes_fix_dir(pts)
    result = estimate_book_width(pts, mean, pc1, pc2)
    w2_mm  = result["av_book_width_m"] * 1000.0 if result["ok"] else None

    t1 = (pts - mean) @ pc1
    t2 = (pts - mean) @ pc2
    pc1_lo, pc1_hi = float(np.percentile(t1, 2)), float(np.percentile(t1, 98))
    t2_lo,  t2_hi  = float(np.percentile(t2, 2)), float(np.percentile(t2, 98))
    pc2_center = mean + pc2 * (t2_lo + t2_hi) / 2.0

    # ── Method 1 の幅方向 3D ────────────────────────────────────────────────
    stored_normal_img = None
    stored_w1_mm = None
    m1_used = False
    pca_path = shot_dir / "pca_result.json"
    if pca_path.exists():
        with open(pca_path) as f:
            pca_data = json.load(f)
        info = pca_data.get("book_width_info", {})
        stored_w1_mm = pca_data.get("book_width_mm")
        if info.get("used"):
            stored_normal_img = info.get("normal")
            m1_used = True

    if stored_normal_img is None:
        stored_normal_img = [1.0, 0.0]
        print("警告: Method 1 normal 未保存 → [1,0] で代替")

    nx, ny = stored_normal_img
    d1_3d = np.array([nx / fx, ny / fy, 0.0])
    d1_3d /= np.linalg.norm(d1_3d)
    w1_half = (stored_w1_mm / 2.0 / 1000.0) if (m1_used and stored_w1_mm) else None

    # Method 1 を保存フィールドから再計算
    w1_recalc_mm = None
    if m1_used and info:
        try:
            nx_i, ny_i = info["normal"]
            w1_recalc_mm = (info["width_px"] * info["z_median_m"]
                            * np.sqrt((nx_i / fx) ** 2 + (ny_i / fy) ** 2) * 1000.0)
        except (KeyError, TypeError):
            pass

    # ── ジオメトリ ───────────────────────────────────────────────────────────
    ALEN = 0.015   # 矢印の長さ [m]
    geoms = [pcd]

    # 黄色球: 重心 (mean)
    m, e = sphere_marker(mean, C_MEAN, C_MEAN_EDGE)
    geoms += [m, e]

    # ライムグリーン矢印: pc1 長手方向
    m, e = arrow(mean + pc1 * pc1_lo, pc1,
                 (pc1_hi - pc1_lo), C_PC1_FACE, C_PC1_EDGE)
    geoms += [m, e]

    # オレンジ矢印: pc2 幅方向 (Method 2)
    m, e = arrow(mean, pc2, ALEN, C_PC2_FACE, C_PC2_EDGE)
    geoms += [m, e]

    # 薄オレンジ スパン: Method 2 測定幅（estimate_book_width の実測値）
    if w2_mm is not None:
        w2_half = w2_mm / 2000.0  # mm → m の半分
        geoms += span(pc2_center - pc2 * w2_half,
                      pc2_center + pc2 * w2_half,
                      C_PC2S_FACE, C_PC2S_EDGE)

    # 水色矢印: Method 1 幅方向
    m, e = arrow(mean, d1_3d, ALEN, C_M1_FACE, C_M1_EDGE)
    geoms += [m, e]

    # 薄水色 スパン: Method 1 測定幅
    if w1_half is not None:
        geoms += span(mean - d1_3d * w1_half,
                      mean + d1_3d * w1_half,
                      C_M1S_FACE, C_M1S_EDGE)

    # ── 比較表・凡例 ─────────────────────────────────────────────────────────
    def _fmm(v):
        return f"{v:8.2f}" if v is not None else "     N/A"

    stored_mm     = pca_data.get("book_width_mm") if pca_path.exists() else None
    stored_label  = "M1採用" if m1_used else "M2採用"
    m1_label      = "再計算" if w1_recalc_mm is not None else "未使用"
    m2_label      = "再計算" if w2_mm  is not None else "計算失敗"

    print()
    print("┌──────────────────────────────────────────────────────────┐")
    print("│  幅推定結果の比較                                         │")
    print("├─────────────────────────────────┬──────────┬────────────┤")
    print("│  手法                           │  幅 [mm] │  備考      │")
    print("├─────────────────────────────────┼──────────┼────────────┤")
    print(f"│  Retrieval_integration (保存)   │{_fmm(stored_mm)} │ {stored_label:8s}   │")
    print(f"│  Method 1  画像平面 → 3D        │{_fmm(w1_recalc_mm)} │ {m1_label:8s}   │")
    print(f"│  Method 2  3D PCA スライス      │{_fmm(w2_mm)} │ {m2_label:8s}   │")
    print("└─────────────────────────────────┴──────────┴────────────┘")
    print()
    print("色の凡例")
    print("  黄色球        : 重心 (mean)")
    print("  ライムグリーン: pc1  本の長手方向（高さ）")
    print("  オレンジ矢印  : pc2  幅方向ベクトル（Method 2）")
    print("  薄オレンジ棒  : Method 2 測定幅スパン")
    print("  水色矢印      : Method 1 幅方向ベクトル（画像平面 → 3D）")
    print("  薄水色棒      : Method 1 測定幅スパン")
    print()
    print("  操作: ドラッグ=回転  Ctrl+ドラッグ=平行移動  スクロール=ズーム  q=終了")
    print()

    t_stored = f"保存={stored_mm:.1f}mm" if stored_mm else "保存=?"
    t_m1     = f"M1={w1_recalc_mm:.1f}mm" if w1_recalc_mm else "M1=N/A"
    t_m2     = f"M2={w2_mm:.1f}mm" if w2_mm else "M2=N/A"
    window_name = f"書籍幅 {shot_dir.name} | {t_stored} | {t_m1} | {t_m2}"

    o3d.visualization.draw_geometries(
        geoms,
        window_name=window_name,
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
