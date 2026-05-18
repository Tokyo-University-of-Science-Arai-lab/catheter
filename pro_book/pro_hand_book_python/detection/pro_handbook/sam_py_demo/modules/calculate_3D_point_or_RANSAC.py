# modules/retrieval_mask_yaw.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pyrealsense2 as rs

def mask_depth_to_points(
    mask01: np.ndarray,          # (H,W) 0/1
    depth_u16: np.ndarray,        # (H,W) Z16
    intr: rs.intrinsics,          # dprof.get_intrinsics()
    depth_scale: float,           # profile.get_device().first_depth_sensor().get_depth_scale()
    stride: int = 2,
    max_points: int = 60000,
    z_min_m: float = 0.10,
    z_max_m: float = 4.0,
    seed: int = 0,
) -> np.ndarray:
    """
    マスク領域の画素を rs2_deproject_pixel_to_point で 3D化して (N,3)[m] を返す。
    """
    if mask01.shape != depth_u16.shape:
        raise ValueError(f"mask {mask01.shape} != depth {depth_u16.shape}")

    rng = np.random.default_rng(seed)

    m = (np.asarray(mask01) > 0)
    if stride > 1:
        s = np.zeros_like(m, dtype=bool)
        s[::stride, ::stride] = True
        m &= s

    ys, xs = np.where(m)
    if ys.size == 0:
        return np.empty((0, 3), np.float32)

    z_m = depth_u16[ys, xs].astype(np.float32) * float(depth_scale)
    ok = (z_m > z_min_m) & (z_m < z_max_m)
    if not np.any(ok):
        return np.empty((0, 3), np.float32)

    xs = xs[ok].astype(np.int32)
    ys = ys[ok].astype(np.int32)
    z_m = z_m[ok]

    if z_m.size > max_points:
        idx = rng.choice(z_m.size, size=max_points, replace=False)
        xs, ys, z_m = xs[idx], ys[idx], z_m[idx]

    pts = np.empty((z_m.size, 3), dtype=np.float32)
    for i, (u, v, zz) in enumerate(zip(xs, ys, z_m)):
        X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], float(zz))
        pts[i] = (X, Y, Z)

    return pts


def ransac_plane_normal(
    pts: np.ndarray,           # (N,3)
    max_iters: int = 600,
    inlier_thr_m: float = 0.01,
    min_inliers: int = 800,
    seed: int = 0,
) -> Tuple[Optional[np.ndarray], int]:
    """
    3点平面RANSACで法線(best_n)を返す（再フィットなし）
    戻り値: (best_n(3,), best_inlier_count) 失敗なら (None, 0)
    """
    n_pts = int(pts.shape[0])
    if n_pts < 3:
        return None, 0

    rng = np.random.default_rng(seed)
    thr = float(inlier_thr_m)

    best_n = None
    best_cnt = 0

    for _ in range(int(max_iters)):
        i1, i2, i3 = rng.choice(n_pts, size=3, replace=False)
        p1, p2, p3 = pts[i1], pts[i2], pts[i3]
        n = np.cross(p2 - p1, p3 - p1).astype(np.float32)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n /= nn
        d = -float(np.dot(n, p1))

        dist = np.abs(pts @ n + d)
        cnt = int(np.count_nonzero(dist < thr))
        if cnt > best_cnt:
            best_cnt = cnt
            best_n = n

    if best_n is None or best_cnt < int(min_inliers):
        return None, best_cnt

    # 法線向き揃え（z正）
    if best_n[2] < 0:
        best_n = -best_n
    print(f"[RANSAC] inliers= {best_cnt} / {pts.shape[0]}")
    return best_n, best_cnt


def normal_to_yaw(n: np.ndarray, degrees: bool = True) -> float:
    """
    yaw = atan2(nx, nz)  （カメラ座標: x右, y下, z前 を想定）
    """
    yaw = float(np.arctan2(float(n[0]), float(n[2])))
    return float(np.degrees(yaw)) if degrees else yaw


def calculate_yaw(
    mask01: np.ndarray,
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    stride: int = 2,
    max_points: int = 60000,
    z_min_m: float = 0.10,
    z_max_m: float = 4.0,
    max_iters: int = 600,
    inlier_thr_m: float = 0.01,
    min_inliers: int = 800,
    degrees: bool = True,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    取り出し書籍マスク → 3D → RANSAC → yaw をまとめて返す
    """
    pts = mask_depth_to_points(
        mask01=mask01,
        depth_u16=depth_u16,
        intr=intr,
        depth_scale=depth_scale,
        stride=stride,
        max_points=max_points,
        z_min_m=z_min_m,
        z_max_m=z_max_m,
        seed=seed,
    )
    if pts.shape[0] < 3:
        return {"ok": False, "reason": "no_points", "yaw": None, "inliers": 0, "points": pts}

    n, cnt = ransac_plane_normal(
        pts, max_iters=max_iters, inlier_thr_m=inlier_thr_m,
        min_inliers=min_inliers, seed=seed
    )
    if n is None:
        return {"ok": False, "reason": "ransac_failed", "yaw": None, "inliers": cnt, "points": pts}

    yaw = normal_to_yaw(n, degrees=degrees)
    return {"ok": True, "yaw": yaw, "normal": n, "inliers": cnt, "points": pts}
