#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
captures フォルダ内の保存データから書籍幅を再計算するスクリプト。

【Method 1 の計算式】
    書籍幅(mm) = width_px × z_median × sqrt((nx/fx)^2 + (ny/fy)^2) × 1000
    ※ fx/fy : カメラ焦点距離（ピクセル単位）。camera_params.json に記載。
               = ピンホールカメラ投影 ΔX_world = Δu × Z / fx の fx に相当。
    ※ nx,ny  : 画像上の幅方向単位ベクトル
    ※ z_median: マスク内の深度中央値(m)

【注意】
    保存済み width_px は「apply_final_t_width_clip」後の最終マスクから計算。
    このスクリプトが読める mask*_depth_prefilter_depth_masked.npy は
    クリップ前の広いマスクのため、再推定値はずれる場合がある。

使い方:
    python calc_book_width.py captures/20260707_112626
    python calc_book_width.py   # 最新フォルダを自動選択
"""

import sys
import json
import numpy as np
from pathlib import Path
import open3d as o3d

sys.path.insert(0, str(Path(__file__).parent / "detection/pro_handbook/sam_py_demo"))

RANSAC_INLIER_THRESHOLD_MM = 5.0


# ──────────────────────────────────────────────────────
#  ファイル読み込み
# ──────────────────────────────────────────────────────

def find_book_depth(shot_dir: Path) -> Path | None:
    candidates = sorted(shot_dir.glob("mask*_depth_prefilter_depth_masked.npy"))
    return candidates[0] if candidates else None


def load_shot(shot_dir: Path):
    with open(shot_dir / "camera_params.json") as f:
        p = json.load(f)
    npy_path = find_book_depth(shot_dir)
    if npy_path is None:
        raise FileNotFoundError("mask*_depth_prefilter_depth_masked.npy が見つかりません")
    print(f"深度ファイル : {npy_path.name}")
    depth_raw = np.load(npy_path)
    return depth_raw, p["fx"], p["fy"], p["ppx"], p["ppy"], p["depth_scale"]


def load_stored_result(shot_dir: Path) -> dict | None:
    for name in ("pca_result.json", "pca_result_offline.json"):
        p = shot_dir / name
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return None


# ──────────────────────────────────────────────────────
#  Method 1 : 画像平面ピクセル幅法
# ──────────────────────────────────────────────────────

def method1_from_mask(depth_raw: np.ndarray, fx: float, fy: float,
                      normal_img,
                      lower_pct: float = 2.0, upper_pct: float = 98.0):
    """
    depth_raw から Method 1 を再推定する。
    normal_img が None の場合はマスク画素の 2D PCA で推定する。
    """
    H, W = depth_raw.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    valid = depth_raw > 0
    us = uu[valid]
    vs = vv[valid]
    zs = depth_raw[valid].astype(np.float64)

    if normal_img is None:
        pts2d = np.stack([us, vs], axis=1)
        _, _, vt = np.linalg.svd(pts2d - pts2d.mean(axis=0), full_matrices=False)
        normal_img = vt[1]  # 第二主成分 = 幅方向
        if normal_img[0] < 0:
            normal_img = -normal_img
        axis_source = "2D PCA 推定"
    else:
        normal_img  = np.asarray(normal_img, dtype=np.float64)
        axis_source = "pca_result.json から取得"

    n = normal_img / np.linalg.norm(normal_img)

    pts2d   = np.stack([us, vs], axis=1)
    center  = pts2d.mean(axis=0)
    t       = (pts2d - center) @ n
    t_lo    = float(np.percentile(t, lower_pct))
    t_hi    = float(np.percentile(t, upper_pct))
    width_px = max(0.0, t_hi - t_lo)

    z_med_m       = float(np.median(zs)) * 1e-3
    nx, ny        = float(n[0]), float(n[1])
    scale_m_per_px = z_med_m * np.sqrt((nx / fx) ** 2 + (ny / fy) ** 2)
    width_mm      = width_px * scale_m_per_px * 1000.0

    return width_mm, {
        "width_px"      : width_px,
        "t_lo_px"       : t_lo,
        "t_hi_px"       : t_hi,
        "z_median_mm"   : z_med_m * 1000.0,
        "scale_mm_per_px": scale_m_per_px * 1000.0,
        "normal_img"    : n.tolist(),
        "n_valid_px"    : int(valid.sum()),
        "axis_source"   : axis_source,
    }


# ──────────────────────────────────────────────────────
#  RANSAC 平面推定
# ──────────────────────────────────────────────────────

def depth_to_points(depth_raw, fx, fy, ppx, ppy, ds):
    H, W = depth_raw.shape
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    Z     = depth_raw.astype(np.float32) * float(ds)
    valid = Z > 0
    Zv    = Z[valid]
    X     = (uu[valid] - ppx) / fx * Zv
    Y     = (vv[valid] - ppy) / fy * Zv
    return np.stack([X, Y, Zv], axis=1)


def ransac_plane(pts, threshold_m=0.005, num_iterations=2000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=threshold_m,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    norm = np.sqrt(a**2 + b**2 + c**2)
    a, b, c, d = a/norm, b/norm, c/norm, d/norm
    distances   = np.abs(pts @ np.array([a, b, c]) + d).astype(np.float32)
    inlier_mask = distances <= threshold_m
    return np.array([a, b, c, d]), distances, inlier_mask


def sep(char="─", width=62):
    print(char * width)


# ──────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────

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

    sep("=")
    print(f"  フォルダ : {shot_dir}")
    sep("=")

    # ── 0. 保存済み結果 ────────────────────────────────────────
    stored      = load_stored_result(shot_dir)
    stored_mm   = None
    stored_info = {}
    if stored:
        stored_mm   = stored.get("book_width_mm")
        stored_info = stored.get("book_width_info", {})

    stored_method  = (stored_info.get("method") or
                      stored_info.get("fallback_method") or "不明")
    stored_used_m1 = bool(stored_info.get("used", False))
    stored_normal  = stored_info.get("normal")   # Method 1 なら存在
    stored_axis    = stored_info.get("axis")
    stored_wpx     = stored_info.get("width_px")
    stored_zmed_m  = stored_info.get("z_median_m")
    stored_scale   = stored_info.get("scale_m_per_px")

    # ── 1. データ読み込み ─────────────────────────────────────
    depth_raw, fx, fy, ppx, ppy, ds = load_shot(shot_dir)
    nonzero = depth_raw[depth_raw > 0]

    print(f"\n【カメラ内部パラメータ】")
    print(f"  fx={fx:.2f} px  fy={fy:.2f} px")
    print(f"  ppx={ppx:.2f} px  ppy={ppy:.2f} px")

    print(f"\n【深度情報 (クリップ前マスク)】")
    print(f"  有効ピクセル数  : {len(nonzero):,} px")
    print(f"  最小/中央/最大 : "
          f"{nonzero.min()*ds*1000:.1f} / "
          f"{np.median(nonzero)*ds*1000:.1f} / "
          f"{nonzero.max()*ds*1000:.1f} mm")
    print(f"  深度範囲幅      : {(nonzero.max()-nonzero.min())*ds*1000:.1f} mm")

    # ── 2. RANSAC 平面推定 ────────────────────────────────────
    pts  = depth_to_points(depth_raw, fx, fy, ppx, ppy, ds)
    thr_m = RANSAC_INLIER_THRESHOLD_MM / 1000.0
    plane, distances, inlier_mask = ransac_plane(pts, threshold_m=thr_m)
    a, b, c, d = plane
    dist_mm   = distances * 1000.0
    n_inlier  = int(inlier_mask.sum())
    tilt_deg  = float(np.degrees(np.arccos(min(1.0, abs(c)))))

    print(f"\n【RANSAC 平面推定  (しきい値={RANSAC_INLIER_THRESHOLD_MM:.1f} mm)】")
    print(f"  平面法線       : [{a:+.4f}, {b:+.4f}, {c:+.4f}]")
    print(f"  Z軸からの傾き  : {tilt_deg:.2f}°  ← 大きいほど本が斜めを向いている")
    sep()
    print(f"  平均距離       : {dist_mm.mean():.2f} mm")
    print(f"  RMS距離        : {float(np.sqrt(np.mean(distances**2)))*1000:.2f} mm")
    print(f"  中央値距離     : {float(np.median(dist_mm)):.2f} mm")
    print(f"  95%距離        : {float(np.percentile(dist_mm, 95)):.2f} mm")
    print(f"  最大距離       : {dist_mm.max():.2f} mm  ← 最も平面から離れた点")
    sep()
    print(f"  インライア     : {n_inlier:,} 点  ({100*n_inlier/len(pts):.1f}%)")
    print(f"  アウトライア   : {len(pts)-n_inlier:,} 点  "
          f"({100*(len(pts)-n_inlier)/len(pts):.1f}%)  "
          f"(>{RANSAC_INLIER_THRESHOLD_MM:.1f}mm)")

    # ── 3. 保存済み Method 1 の計算式を検証 ─────────────────
    if stored_used_m1 and stored_wpx is not None:
        nx, ny = float(stored_normal[0]), float(stored_normal[1])
        recomputed_scale = stored_zmed_m * np.sqrt((nx/fx)**2 + (ny/fy)**2)
        recomputed_mm    = stored_wpx * recomputed_scale * 1000.0

        print(f"\n【保存済み Method 1 の計算式を検証】")
        print(f"  ─ 保存されていたパラメータ ─")
        print(f"  axis   (長手)  : {stored_axis}  (画像上の背表紙方向)")
        print(f"  normal (幅)    : {stored_normal}  (画像上の幅方向)")
        print(f"  width_px       : {stored_wpx:.2f} px")
        print(f"  z_median       : {stored_zmed_m*1000:.1f} mm  ({stored_zmed_m:.4f} m)")
        print(f"  fx             : {fx:.2f} px")
        print(f"  ─ 計算式 ─")
        print(f"  scale(mm/px) = z_med × sqrt((nx/fx)² + (ny/fy)²)")
        print(f"               = {stored_zmed_m*1000:.1f} mm × "
              f"sqrt(({nx:.2f}/{fx:.0f})² + ({ny:.2f}/{fy:.0f})²)")
        print(f"               = {recomputed_scale*1000:.6f} mm/px")
        print(f"  幅 = {stored_wpx:.2f} px × {recomputed_scale*1000:.6f} mm/px")
        print(f"     = {recomputed_mm:.2f} mm")
        if abs(recomputed_mm - stored_mm) < 0.01:
            print(f"  ✔ 保存済み {stored_mm:.2f} mm と一致")
        else:
            print(f"  ✘ 保存済み {stored_mm:.2f} mm と差あり: {recomputed_mm - stored_mm:+.3f} mm")

    # ── 4. 今読めるマスクで Method 1 を再推定 ───────────────
    # pca_result に normal があればそれを使い、なければ 2D PCA で推定
    use_normal = stored_normal if stored_used_m1 else None
    w_all, info_all = method1_from_mask(depth_raw, fx, fy, use_normal)

    # RANSAC インライアのみ
    H2, W2 = depth_raw.shape
    uu2, vv2 = np.meshgrid(np.arange(W2), np.arange(H2))
    valid_idx   = np.where(depth_raw.ravel() > 0)[0]
    outlier_idx = valid_idx[~inlier_mask]
    inlier_dep  = depth_raw.ravel().copy()
    inlier_dep[outlier_idx] = 0
    inlier_dep  = inlier_dep.reshape(H2, W2)
    w_inlier, info_inlier = method1_from_mask(inlier_dep, fx, fy, use_normal)

    note = ("※ 使用マスク: クリップ前の depth_prefilter マスク。\n"
            "   保存時は apply_final_t_width_clip 後の細いマスクを使用のため値がずれる場合がある。")
    print(f"\n【Method 1 再推定: 全点】")
    print(f"  {note}")
    print(f"  軸の取得方法   : {info_all['axis_source']}")
    print(f"  normal (幅)    : {[round(x,4) for x in info_all['normal_img']]}")
    print(f"  有効ピクセル数 : {info_all['n_valid_px']:,} px")
    print(f"  width_px       : {info_all['width_px']:.2f} px")
    print(f"    [{info_all['t_lo_px']:.1f}, {info_all['t_hi_px']:.1f}] px の 2%~98% 幅")
    print(f"  z_median       : {info_all['z_median_mm']:.1f} mm")
    print(f"  scale          : {info_all['scale_mm_per_px']:.6f} mm/px")
    print(f"  書籍幅         : {w_all:.2f} mm")

    print(f"\n【Method 1 再推定: RANSAC インライアのみ (>{RANSAC_INLIER_THRESHOLD_MM:.1f}mm 除外)】")
    print(f"  有効ピクセル数 : {info_inlier['n_valid_px']:,} px")
    print(f"  width_px       : {info_inlier['width_px']:.2f} px")
    print(f"  z_median       : {info_inlier['z_median_mm']:.1f} mm")
    print(f"  書籍幅         : {w_inlier:.2f} mm")

    # ── 5. 比較サマリ ─────────────────────────────────────────
    sep("=")
    print("  ★ 結果まとめ")
    sep("=")
    if stored_mm is not None:
        print(f"  保存済み (pca_result.json)       : {stored_mm:.2f} mm  [{stored_method}]")
        if stored_used_m1:
            print(f"    → width_px={stored_wpx:.1f}  z_med={stored_zmed_m*1000:.1f}mm  "
                  f"normal={stored_normal}")
    print(f"  再推定 Method 1 (全点)           : {w_all:.2f} mm")
    print(f"  再推定 Method 1 (RANSACインライア): {w_inlier:.2f} mm")
    if stored_mm is not None:
        print(f"  差 (全点 - 保存済み)             : {w_all - stored_mm:+.2f} mm")
        print(f"  差 (インライア - 保存済み)       : {w_inlier - stored_mm:+.2f} mm")
    print(f"  深度ばらつき (RANSAC最大距離)    : {dist_mm.max():.2f} mm")
    sep("=")


if __name__ == "__main__":
    main()
