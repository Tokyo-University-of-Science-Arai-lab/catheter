from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# RealSense Depth filter
# ============================================================

def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    RealSense Viewerに近いDepthフィルタ．
    RGBとDepthの対応を崩さないため，decimationは使わない．
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


# ============================================================
# Depth mask generation
# ============================================================

def estimate_depth_mode(
    depth_u16: np.ndarray,
    depth_scale: float,
    z_min_m: float = 0.2,
    z_max_m: float = 1.2,
    bins: int = 100,
) -> float:
    """
    画像内で最も多いDepth帯を代表的な書籍面Depthとして推定する．
    """
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    valid = (depth_m > z_min_m) & (depth_m < z_max_m)

    if not np.any(valid):
        raise RuntimeError("有効なDepthがありません．")

    vals = depth_m[valid]
    hist, edges = np.histogram(vals, bins=bins, range=(z_min_m, z_max_m))
    idx = int(np.argmax(hist))
    depth_ref = 0.5 * (edges[idx] + edges[idx + 1])

    return float(depth_ref)


def make_far_space_mask(
    depth_u16: np.ndarray,
    depth_scale: float,
    depth_ref_m: float,
    delta_depth_m: float = 0.03,
) -> np.ndarray:
    """
    代表Depthより奥にある領域を収納スペース候補として二値化する．
    """
    depth_m = depth_u16.astype(np.float32) * float(depth_scale)
    valid = depth_u16 > 0

    far = valid & (depth_m > float(depth_ref_m) + float(delta_depth_m))

    mask = far.astype(np.uint8)

    # 軽いノイズ除去
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    return mask


def remove_horizontal_structures(
    mask: np.ndarray,
    *,
    horizontal_kernel_w: int = 120,
    horizontal_kernel_h: int = 7,
) -> np.ndarray:
    """
    横長構造を除去する．
    収納候補が削れすぎる場合は horizontal_kernel_w を大きくする．
    """
    mask_u8 = (mask > 0).astype(np.uint8)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (int(horizontal_kernel_w), int(horizontal_kernel_h)),
    )

    horizontal = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, horizontal_kernel)
    cleaned = cv2.subtract(mask_u8, horizontal)

    # 小ノイズ除去
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    # 縦方向を少しだけ補完
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    return cleaned


# ============================================================
# Candidate component selection
# ============================================================

def select_space_component(
    candidate_mask: np.ndarray,
    *,
    min_area_px: int = 300,
    horizontal_ratio_thr: float = 2.0,
    max_width_px: int = 220,
    max_area_px: int = 60000,
) -> Optional[dict[str, Any]]:
    """
    Depthから得た候補二値画像に対して連結成分解析を行い，
    横長領域・大きすぎる領域を除外した上で収納スペース候補を1つ選ぶ．

    horizontal_ratio_thr:
        w / h > horizontal_ratio_thr の成分を横長として除外する．

    max_width_px:
        bbox幅がこれより大きい成分を除外する．

    max_area_px:
        面積が大きすぎる成分を除外する．
    """
    mask_u8 = (candidate_mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8,
        connectivity=8,
    )

    candidates: list[dict[str, Any]] = []

    print(f"[DEBUG] connected components: {num_labels - 1}")

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]

        aspect_w_h = float(w / max(h, 1))
        aspect_h_w = float(h / max(w, 1))

        component_mask = (labels == label).astype(np.uint8)

        # 行ごとの幅を確認する
        row_widths = []
        ys = np.where(component_mask > 0)[0]
        for yy in np.unique(ys):
            xs = np.where(component_mask[yy] > 0)[0]
            if xs.size > 0:
                row_widths.append(xs.max() - xs.min() + 1)

        if len(row_widths) > 0:
            median_row_width = float(np.median(row_widths))
            max_row_width = float(np.max(row_widths))
        else:
            median_row_width = 0.0
            max_row_width = 0.0

        print(
            f"[DEBUG] label={label}, bbox=({x},{y},{w},{h}), "
            f"area={area}, w/h={aspect_w_h:.2f}, "
            f"median_row_width={median_row_width:.1f}, "
            f"max_row_width={max_row_width:.1f}"
        )

        if area < int(min_area_px):
            print("  -> reject: area too small")
            continue

        if area > int(max_area_px):
            print("  -> reject: area too large")
            continue

        if h <= 0 or w <= 0:
            print("  -> reject: invalid bbox")
            continue

        if aspect_w_h > float(horizontal_ratio_thr):
            print("  -> reject: horizontal")
            continue

        if w > int(max_width_px):
            print("  -> reject: bbox too wide")
            continue

        if median_row_width > float(max_width_px):
            print("  -> reject: median row too wide")
            continue

        if max_row_width > float(max_width_px) * 1.8:
            print("  -> reject: row too wide")
            continue

        # 面積最大ではなく，細長さ・高さを重視する
        score = 0.0
        score += 5.0 * float(h)
        score += 300.0 * float(aspect_h_w)
        score -= 2.0 * float(w)
        score += 0.01 * float(area)

        candidates.append({
            "label": int(label),
            "bbox": (int(x), int(y), int(w), int(h)),
            "center": (float(cx), float(cy)),
            "area": int(area),
            "width": int(w),
            "height": int(h),
            "aspect_w_h": aspect_w_h,
            "aspect_h_w": aspect_h_w,
            "median_row_width": median_row_width,
            "max_row_width": max_row_width,
            "score": float(score),
            "mask": component_mask,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda d: d["score"], reverse=True)
    print("[DEBUG] selected component:", {
        "bbox": candidates[0]["bbox"],
        "area": candidates[0]["area"],
        "score": candidates[0]["score"],
        "aspect_w_h": candidates[0]["aspect_w_h"],
        "median_row_width": candidates[0]["median_row_width"],
        "max_row_width": candidates[0]["max_row_width"],
    })

    return candidates[0]


# ============================================================
# Guide line from left boundary
# ============================================================

def extract_left_boundary_points(component_mask: np.ndarray) -> np.ndarray:
    """
    収納スペース候補マスクの左側境界点を抽出する．
    各y行について，候補画素のうち最も左のxを取る．
    """
    mask = component_mask.astype(bool)
    pts: list[list[int]] = []

    ys = np.where(mask.any(axis=1))[0]

    for y in ys:
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue

        x_left = int(xs.min())
        pts.append([x_left, int(y)])

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)

def extract_side_boundary_points(
    component_mask: np.ndarray,
    side: str,
) -> np.ndarray:
    """
    収納スペース候補マスクの片側境界点を抽出する．

    side:
        "left"  -> 各y行で最も左のxを取る
        "right" -> 各y行で最も右のxを取る

    Returns
    -------
    pts : (N, 2) ndarray
        画像座標 [x, y]
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side}")

    mask = component_mask.astype(bool)
    pts = []

    ys = np.where(mask.any(axis=1))[0]

    for y in ys:
        xs = np.where(mask[y])[0]
        if xs.size == 0:
            continue

        if side == "left":
            x = int(xs.min())
        else:
            x = int(xs.max())

        pts.append([x, int(y)])

    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    return np.asarray(pts, dtype=np.float32)


def line_tilt_from_vertical_deg(
    p0: np.ndarray,
    p1: np.ndarray,
) -> float:
    """
    画像上の直線が垂直方向からどれだけ傾いているかを返す．

    戻り値：
        0 deg  -> 垂直
        大きい -> 斜め
    """
    p0 = np.asarray(p0, dtype=np.float32).reshape(2)
    p1 = np.asarray(p1, dtype=np.float32).reshape(2)

    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])

    angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    # [-180, 180] を [0, 180) に正規化
    if angle_deg < 0.0:
        angle_deg += 180.0

    # 垂直は90deg
    tilt = abs(angle_deg - 90.0)

    return float(tilt)


def extract_tilted_boundary_line(
    component_mask: np.ndarray,
    *,
    default_side: str = "left",
    min_points: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    収納スペース候補の左側境界・右側境界の両方を見て，
    より垂直から傾いている方をガイドラインとして選ぶ．

    三角形スペースでは，直立側ではなく斜辺側が選ばれやすい．

    Returns
    -------
    p0, p1 : ndarray
        選ばれた境界線の端点 [x, y]
    boundary_pts : ndarray
        選ばれた境界点列
    selected_side : str
        "left" or "right"
    info : dict
        デバッグ情報
    """
    left_pts = extract_side_boundary_points(component_mask, "left")
    right_pts = extract_side_boundary_points(component_mask, "right")

    candidates = []

    if left_pts.shape[0] >= min_points:
        left_p0, left_p1 = fit_line_pca_2d(left_pts)
        left_tilt = line_tilt_from_vertical_deg(left_p0, left_p1)
        candidates.append({
            "side": "left",
            "pts": left_pts,
            "p0": left_p0,
            "p1": left_p1,
            "tilt_from_vertical_deg": left_tilt,
        })

    if right_pts.shape[0] >= min_points:
        right_p0, right_p1 = fit_line_pca_2d(right_pts)
        right_tilt = line_tilt_from_vertical_deg(right_p0, right_p1)
        candidates.append({
            "side": "right",
            "pts": right_pts,
            "p0": right_p0,
            "p1": right_p1,
            "tilt_from_vertical_deg": right_tilt,
        })

    if len(candidates) == 0:
        raise RuntimeError("左右境界点が不足しています．")

    # 左右どちらも使える場合，垂直からの傾きが大きい方を選ぶ
    candidates.sort(key=lambda d: d["tilt_from_vertical_deg"], reverse=True)
    selected = candidates[0]

    # 完全に同程度なら default_side を優先
    if len(candidates) >= 2:
        diff = abs(
            candidates[0]["tilt_from_vertical_deg"]
            - candidates[1]["tilt_from_vertical_deg"]
        )
        if diff < 1.0:
            for c in candidates:
                if c["side"] == default_side:
                    selected = c
                    break

    info = {
        "left_points": int(left_pts.shape[0]),
        "right_points": int(right_pts.shape[0]),
        "left_tilt_from_vertical_deg": None,
        "right_tilt_from_vertical_deg": None,
    }

    for c in candidates:
        if c["side"] == "left":
            info["left_tilt_from_vertical_deg"] = c["tilt_from_vertical_deg"]
        elif c["side"] == "right":
            info["right_tilt_from_vertical_deg"] = c["tilt_from_vertical_deg"]

    return (
        selected["p0"].astype(np.float32),
        selected["p1"].astype(np.float32),
        selected["pts"].astype(np.float32),
        selected["side"],
        info,
    )


def fit_line_pca_2d(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D点列にPCAで直線フィットする．
    戻り値:
        p0, p1 : 直線端点 [x, y]
    """
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)

    if pts.shape[0] < 2:
        raise RuntimeError("直線フィットに必要な点が不足しています．")

    mean = pts.mean(axis=0)
    X = pts - mean

    _, _, vh = np.linalg.svd(X, full_matrices=False)
    axis = vh[0].astype(np.float32)

    # 画像下方向へ向くように揃える
    if axis[1] < 0:
        axis = -axis

    proj = X @ axis
    p0 = mean + axis * float(proj.min())
    p1 = mean + axis * float(proj.max())

    # p0を上側，p1を下側にそろえる
    if p0[1] > p1[1]:
        p0, p1 = p1, p0

    return p0.astype(np.float32), p1.astype(np.float32)


def enumerate_line_pixels(
    p0: tuple[int, int],
    p1: tuple[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    """
    画像座標上の2点を結ぶ線分上のピクセル座標を列挙する．
    """
    x0, y0 = p0
    x1, y1 = p1

    n = int(np.hypot(x1 - x0, y1 - y0)) + 1
    n = max(n, 1)

    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)

    xs_i = np.clip(np.round(xs).astype(np.int32), 0, width - 1)
    ys_i = np.clip(np.round(ys).astype(np.int32), 0, height - 1)

    coords = np.stack([xs_i, ys_i], axis=1)
    coords = np.unique(coords, axis=0)

    return coords


# ============================================================
# 2D -> 3D deprojection
# ============================================================

def deproject_line_to_3d(
    line_coords: np.ndarray,
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    画像上の線分座標列をDepthからカメラ座標系3D点列に変換する．

    Returns
    -------
    pts_cam : (N, 3)
        カメラ座標系3D点列 [m]
    valid_pixels : (N, 2)
        pts_camに対応する画像座標 [x, y]
    """
    pts = []
    valid_pixels = []

    H, W = depth_u16.shape[:2]

    for x, y in line_coords:
        x = int(x)
        y = int(y)

        if y < 0 or y >= H or x < 0 or x >= W:
            continue

        d = int(depth_u16[y, x])
        if d == 0:
            continue

        z = float(d) * float(depth_scale)
        if z < z_min_m or z > z_max_m:
            continue

        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            intr,
            [float(x), float(y)],
            z,
        )

        pts.append([X, Y, Z])
        valid_pixels.append([x, y])

    if len(pts) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
        )

    return (
        np.asarray(pts, dtype=np.float32),
        np.asarray(valid_pixels, dtype=np.int32),
    )


def select_first_target_from_line_points_with_pixel(
    pts_line_cam: np.ndarray,
    valid_pixels: np.ndarray,
    target_dist_m: float = 0.08,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    3D線分点列からリーチング用第一目標点を選び，
    その点に対応する画像座標も返す．

    RealSense座標系ではYが画像下向きなので，
    Y最大点を下側とし，そこから target_dist_m 離れた点を選ぶ．
    """
    pts = np.asarray(pts_line_cam, dtype=np.float32).reshape(-1, 3)
    pixels = np.asarray(valid_pixels, dtype=np.int32).reshape(-1, 2)

    if pts.shape[0] == 0:
        raise RuntimeError("3D線分点群が空です．")

    if pts.shape[0] != pixels.shape[0]:
        raise RuntimeError(
            f"pts_line_cam と valid_pixels の数が一致しません: "
            f"{pts.shape[0]} vs {pixels.shape[0]}"
        )

    idx_bottom = int(np.argmax(pts[:, 1]))
    p_bottom = pts[idx_bottom]

    dists = np.linalg.norm(pts - p_bottom.reshape(1, 3), axis=1)
    idx_first = int(np.argmin(np.abs(dists - float(target_dist_m))))

    first_target_cam = pts[idx_first].astype(np.float32)
    first_target_px = (int(pixels[idx_first, 0]), int(pixels[idx_first, 1]))

    return first_target_cam, first_target_px


# ============================================================
# Book surface plane estimation
# ============================================================

def collect_book_surface_points_around_space(
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    bbox: tuple[int, int, int, int],
    depth_ref_m: float,
    *,
    depth_tol_m: float = 0.05,
    side_margin_px: int = 140,
    gap_px: int = 5,
    stride: int = 2,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    選択された収納スペースbboxの左右から，書籍面らしいDepth点を集める．

    条件：
      - bbox左右のROIを見る
      - depth_ref_m に近いDepthだけを採用する
      - 各点の画像座標も返す
    """
    x, y, w, h = [int(v) for v in bbox]
    H, W = depth_u16.shape[:2]

    y0 = max(0, y)
    y1 = min(H, y + h)

    left_x0 = max(0, x - int(side_margin_px))
    left_x1 = max(0, x - int(gap_px))

    right_x0 = min(W, x + w + int(gap_px))
    right_x1 = min(W, x + w + int(side_margin_px))

    rois = [
        (left_x0, left_x1, y0, y1),
        (right_x0, right_x1, y0, y1),
    ]

    pts = []
    pixels = []

    for rx0, rx1, ry0, ry1 in rois:
        if rx1 <= rx0 or ry1 <= ry0:
            continue

        for v in range(ry0, ry1, max(1, int(stride))):
            for u in range(rx0, rx1, max(1, int(stride))):
                d = int(depth_u16[v, u])
                if d == 0:
                    continue

                z = float(d) * float(depth_scale)

                if z < z_min_m or z > z_max_m:
                    continue

                if abs(z - float(depth_ref_m)) > float(depth_tol_m):
                    continue

                X, Y, Z = rs.rs2_deproject_pixel_to_point(
                    intr,
                    [float(u), float(v)],
                    z,
                )

                pts.append([X, Y, Z])
                pixels.append([u, v])

    if len(pts) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 2), dtype=np.int32),
        )

    return (
        np.asarray(pts, dtype=np.float32),
        np.asarray(pixels, dtype=np.int32),
    )


def fit_plane_ransac(
    points: np.ndarray,
    *,
    distance_threshold_m: float = 0.01,
    max_iter: int = 300,
    min_inliers: int = 30,
    random_seed: int = 0,
) -> tuple[Optional[np.ndarray], np.ndarray]:
    """
    3D点群にRANSACで平面を当てる．

    平面式：
        aX + bY + cZ + d = 0

    Returns
    -------
    plane : (4,) ndarray or None
        [a,b,c,d]
    inlier_mask : (N,) bool ndarray
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    n = pts.shape[0]

    if n < 3:
        return None, np.zeros((n,), dtype=bool)

    rng = np.random.default_rng(random_seed)

    best_plane = None
    best_inlier_mask = np.zeros((n,), dtype=bool)
    best_count = 0

    for _ in range(int(max_iter)):
        ids = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = pts[ids]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue

        normal = normal / norm
        d = -float(np.dot(normal, p1))

        plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

        distances = np.abs(pts @ plane[:3] + plane[3])
        inlier_mask = distances < float(distance_threshold_m)
        count = int(np.sum(inlier_mask))

        if count > best_count:
            best_count = count
            best_plane = plane
            best_inlier_mask = inlier_mask

    if best_plane is None or best_count < int(min_inliers):
        return None, np.zeros((n,), dtype=bool)

    # inlierで平面を最小二乗再推定
    inlier_pts = pts[best_inlier_mask]
    centroid = inlier_pts.mean(axis=0)
    X = inlier_pts - centroid

    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)

    d = -float(np.dot(normal, centroid))
    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)

    return plane, best_inlier_mask


def correct_point_z_by_plane(
    point_cam: np.ndarray,
    plane: np.ndarray,
    *,
    surface_clearance_m: float = 0.0,
) -> np.ndarray:
    """
    目標点のX,Yを固定し，Zだけを平面上の値に補正する．

    surface_clearance_m:
        書籍面から少し手前にしたい場合に使う．
        RealSense座標系でZがカメラ前方向の場合，
        正の値を指定すると Z を小さくする．
    """
    p = np.asarray(point_cam, dtype=np.float32).reshape(3)
    a, b, c, d = [float(v) for v in np.asarray(plane).reshape(4)]

    if abs(c) < 1e-8:
        raise RuntimeError("平面のcが小さすぎるため，Z補正できません．")

    z_plane = -(a * float(p[0]) + b * float(p[1]) + d) / c

    corrected = p.copy()
    corrected[2] = float(z_plane) - float(surface_clearance_m)

    return corrected.astype(np.float32)


def fallback_correct_point_z_by_median_surface(
    point_cam: np.ndarray,
    surface_points: np.ndarray,
    *,
    surface_clearance_m: float = 0.0,
) -> np.ndarray:
    """
    RANSACが失敗した場合のフォールバック．
    書籍面候補点群のZ中央値で目標点Zを補正する．
    """
    pts = np.asarray(surface_points, dtype=np.float32).reshape(-1, 3)
    p = np.asarray(point_cam, dtype=np.float32).reshape(3)

    if pts.shape[0] == 0:
        return p.astype(np.float32)

    corrected = p.copy()
    corrected[2] = float(np.median(pts[:, 2])) - float(surface_clearance_m)

    return corrected.astype(np.float32)


# ============================================================
# Debug visualization
# ============================================================

def save_binary_target_overlay(
    binary_mask: np.ndarray,
    *,
    line_p0: tuple[int, int] | None = None,
    line_p1: tuple[int, int] | None = None,
    target_px: tuple[int, int] | None = None,
    save_path: Path,
) -> None:
    """
    二値化画像にガイドラインと目標点を描画して保存する．

    白：二値化領域
    赤線：ガイドライン
    赤丸：目標点
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    vis = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)

    if line_p0 is not None and line_p1 is not None:
        cv2.line(
            vis,
            (int(line_p0[0]), int(line_p0[1])),
            (int(line_p1[0]), int(line_p1[1])),
            (0, 0, 255),
            3,
        )

    if target_px is not None:
        cv2.circle(
            vis,
            (int(target_px[0]), int(target_px[1])),
            8,
            (0, 0, 255),
            -1,
        )
        cv2.circle(
            vis,
            (int(target_px[0]), int(target_px[1])),
            13,
            (0, 0, 255),
            2,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] binary target overlay: {save_path}")


def save_debug_overlay(
    rgb_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    selected_mask: np.ndarray,
    line_p0: tuple[int, int],
    line_p1: tuple[int, int],
    target_px: tuple[int, int] | None,
    save_path: Path,
) -> None:
    """
    RGB画像上に候補領域，選択領域，ガイドライン，目標点を重ねて保存する．
    """
    vis = rgb_bgr.copy()

    cand = candidate_mask.astype(bool)
    vis[cand] = (
        0.5 * vis[cand] + 0.5 * np.array([0, 255, 255], dtype=np.float32)
    ).astype(np.uint8)

    sel = selected_mask.astype(bool)
    vis[sel] = (
        0.4 * vis[sel] + 0.6 * np.array([0, 0, 255], dtype=np.float32)
    ).astype(np.uint8)

    cv2.line(vis, line_p0, line_p1, (255, 0, 255), 4)
    cv2.circle(vis, line_p0, 5, (0, 255, 255), -1)
    cv2.circle(vis, line_p1, 5, (255, 255, 255), -1)

    if target_px is not None:
        cv2.circle(vis, target_px, 8, (0, 255, 0), -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] RGB debug overlay: {save_path}")


def save_surface_points_overlay(
    rgb_bgr: np.ndarray,
    surface_pixels: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    save_path: Path,
) -> None:
    """
    書籍面平面推定に使った画素をRGB上に可視化する．
    青：候補点
    緑：RANSAC inlier
    """
    vis = rgb_bgr.copy()

    pixels = np.asarray(surface_pixels, dtype=np.int32).reshape(-1, 2)

    for k, (u, v) in enumerate(pixels):
        if v < 0 or v >= vis.shape[0] or u < 0 or u >= vis.shape[1]:
            continue

        if inlier_mask is not None and k < len(inlier_mask) and bool(inlier_mask[k]):
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        cv2.circle(vis, (int(u), int(v)), 1, color, -1)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)
    print(f"[SAVE] surface points overlay: {save_path}")

def show_image(
    title: str,
    img: np.ndarray,
    wait: bool = True,
    delay_ms: int = 1,
) -> None:
    """
    OpenCVで画像を表示する．

    wait=True:
        キー入力があるまで待つ
    wait=False:
        delay_msだけ表示して次へ進む
    """
    if img is None:
        print(f"[WARN] show_image: {title} is None")
        return

    cv2.imshow(title, img)

    if wait:
        print(f"[INFO] showing {title}. Press any key on image window.")
        cv2.waitKey(0)
    else:
        cv2.waitKey(delay_ms)


def show_image_from_path(
    title: str,
    path: str | Path,
    wait: bool = True,
    delay_ms: int = 1,
) -> None:
    """
    保存済み画像を読み込んで表示する．
    """
    path = Path(path)

    if not path.exists():
        print(f"[WARN] image not found: {path}")
        return

    img = cv2.imread(str(path))

    if img is None:
        print(f"[WARN] failed to read image: {path}")
        return

    show_image(title, img, wait=wait, delay_ms=delay_ms)

def trim_bottom_excess_from_space_mask(
    component_mask: np.ndarray,
    *,
    max_row_width_px: int = 160,
    row_width_ratio_thr: float = 2.0,
    bottom_search_ratio: float = 0.45,
    min_keep_height_px: int = 60,
) -> np.ndarray:
    """
    選択された収納スペース候補から，下側に連結した余分な領域を削除する．

    前提：
      - 収納スペースは三角形または四角形になりやすい
      - 下側の余分領域は，行方向に幅が大きくなりやすい
      - 画像全体の下端ではなく，選択成分の下側だけを見る

    Parameters
    ----------
    component_mask:
        選択された収納スペース候補の二値画像．0/1 or 0/255．
    max_row_width_px:
        1行あたりの白画素幅がこれを超えると，下側余分領域とみなす候補．
    row_width_ratio_thr:
        通常幅の何倍を超えたら異常幅とみなすか．
    bottom_search_ratio:
        成分の下側何割を探索対象にするか．
    min_keep_height_px:
        削りすぎ防止．最低限この高さは残す．

    Returns
    -------
    cleaned_mask:
        下側余分領域を削った二値マスク．
    """
    mask = (component_mask > 0).astype(np.uint8)
    H, W = mask.shape[:2]

    ys_all, xs_all = np.where(mask > 0)
    if ys_all.size == 0:
        return mask

    y_min = int(ys_all.min())
    y_max = int(ys_all.max())
    comp_h = y_max - y_min + 1

    if comp_h < min_keep_height_px:
        return mask

    row_infos = []

    for y in range(y_min, y_max + 1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size == 0:
            continue

        x_left = int(xs.min())
        x_right = int(xs.max())
        width = x_right - x_left + 1
        cx = 0.5 * (x_left + x_right)

        row_infos.append((y, x_left, x_right, width, cx))

    if len(row_infos) < 5:
        return mask

    widths = np.array([r[3] for r in row_infos], dtype=np.float32)

    # 幅の基準値．極端な広がりに引っ張られないよう中央値を使う．
    ref_width = float(np.median(widths))
    width_thr = min(
        float(max_row_width_px),
        max(float(max_row_width_px) * 0.5, ref_width * float(row_width_ratio_thr)),
    )

    # 成分の下側だけを見る
    search_start_y = int(y_min + comp_h * (1.0 - bottom_search_ratio))

    # 下から上に見て，異常に広い行が続く部分を削る
    cut_y = None

    for y, x_left, x_right, width, cx in reversed(row_infos):
        if y < search_start_y:
            break

        if width > width_thr:
            cut_y = y
        else:
            # 下から見て，正常幅に戻ったらそこより下を削る
            if cut_y is not None:
                break

    cleaned = mask.copy()

    if cut_y is not None:
        # 削りすぎ防止
        min_allowed_cut_y = y_min + min_keep_height_px
        cut_y = max(cut_y, min_allowed_cut_y)

        # cut_y 以降を削る
        cleaned[cut_y:, :] = 0

        print(
            f"[INFO] trim_bottom_excess: y_min={y_min}, y_max={y_max}, "
            f"ref_width={ref_width:.1f}, width_thr={width_thr:.1f}, cut_y={cut_y}"
        )
    else:
        print(
            f"[INFO] trim_bottom_excess: no trim, "
            f"ref_width={ref_width:.1f}, width_thr={width_thr:.1f}"
        )

    # 削った後に最大連結成分だけ残す
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return cleaned

    best_label = None
    best_area = 0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    if best_label is None:
        return cleaned

    cleaned = (labels == best_label).astype(np.uint8)

    return cleaned

def select_target_pixel_on_line_by_y(
    p0: tuple[int, int],
    p1: tuple[int, int],
    *,
    target_y_px: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    """
    画像上のガイドライン p0-p1 に対して，
    指定した y 座標 target_y_px 上の点を求める．

    目的：
      目標点の高さを3D距離ではなく画像上のy座標で固定する．
    """
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])

    target_y_px = int(np.clip(target_y_px, 0, height - 1))

    # ガイドラインがほぼ水平の場合は，中央xを使う
    if abs(y1 - y0) < 1:
        x = int(round(0.5 * (x0 + x1)))
        return (
            int(np.clip(x, 0, width - 1)),
            target_y_px,
        )

    # 線形補間で target_y_px に対応するxを求める
    t = (float(target_y_px) - float(y0)) / float(y1 - y0)
    t = float(np.clip(t, 0.0, 1.0))

    x = float(x0) + t * float(x1 - x0)

    return (
        int(np.clip(round(x), 0, width - 1)),
        target_y_px,
    )


def deproject_single_pixel_to_3d(
    pixel_xy: tuple[int, int],
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    search_radius_px: int = 5,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> np.ndarray:
    """
    指定画素をDepthから3D化する．
    指定画素のDepthが無効な場合，周辺 search_radius_px 内の有効Depth中央値を使う．
    """
    H, W = depth_u16.shape[:2]
    u, v = int(pixel_xy[0]), int(pixel_xy[1])

    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    d = int(depth_u16[v, u])

    if d == 0:
        x0 = max(0, u - search_radius_px)
        x1 = min(W, u + search_radius_px + 1)
        y0 = max(0, v - search_radius_px)
        y1 = min(H, v + search_radius_px + 1)

        patch = depth_u16[y0:y1, x0:x1]
        valid = patch[patch > 0]

        if valid.size == 0:
            raise RuntimeError(
                f"target pixel depth is invalid and no valid depth nearby: pixel={pixel_xy}"
            )

        d = int(np.median(valid))

    z = float(d) * float(depth_scale)

    if z < z_min_m or z > z_max_m:
        raise RuntimeError(
            f"target depth out of range: pixel={pixel_xy}, z={z:.4f}"
        )

    X, Y, Z = rs.rs2_deproject_pixel_to_point(
        intr,
        [float(u), float(v)],
        z,
    )

    return np.asarray([X, Y, Z], dtype=np.float32)

def intersect_pixel_ray_with_plane(
    pixel_xy: tuple[int, int],
    intr: rs.intrinsics,
    plane: np.ndarray,
) -> np.ndarray:
    """
    画像上の1画素から出るカメラレイと，RANSAC平面の交点を求める．

    plane:
        aX + bY + cZ + d = 0
    """
    u, v = int(pixel_xy[0]), int(pixel_xy[1])
    a, b, c, d = [float(x) for x in np.asarray(plane).reshape(4)]

    # Z=1.0[m] の点をdeprojectすると，その画素方向のレイ上の点が得られる
    ray = np.asarray(
        rs.rs2_deproject_pixel_to_point(
            intr,
            [float(u), float(v)],
            1.0,
        ),
        dtype=np.float32,
    )

    normal = np.asarray([a, b, c], dtype=np.float32)
    denom = float(np.dot(normal, ray))

    if abs(denom) < 1e-8:
        raise RuntimeError("画素レイと平面がほぼ平行なため，交点を計算できません．")

    scale = -d / denom

    if scale <= 0:
        raise RuntimeError(
            f"平面交点がカメラ後方になりました．pixel={pixel_xy}, scale={scale:.6f}"
        )

    point = ray * float(scale)
    return point.astype(np.float32)


def deproject_pixel_with_neighborhood(
    pixel_xy: tuple[int, int],
    depth_u16: np.ndarray,
    intr: rs.intrinsics,
    depth_scale: float,
    *,
    search_radius_px: int = 5,
    z_min_m: float = 0.1,
    z_max_m: float = 1.5,
) -> np.ndarray:
    """
    指定画素をDepthから3D化する．
    その画素のDepthが無効な場合は，周辺の有効Depth中央値を使う．
    """
    H, W = depth_u16.shape[:2]
    u, v = int(pixel_xy[0]), int(pixel_xy[1])

    u = int(np.clip(u, 0, W - 1))
    v = int(np.clip(v, 0, H - 1))

    d = int(depth_u16[v, u])

    if d == 0:
        x0 = max(0, u - search_radius_px)
        x1 = min(W, u + search_radius_px + 1)
        y0 = max(0, v - search_radius_px)
        y1 = min(H, v + search_radius_px + 1)

        patch = depth_u16[y0:y1, x0:x1]
        valid = patch[patch > 0]

        if valid.size == 0:
            raise RuntimeError(
                f"Depthが無効で，周辺にも有効Depthがありません．pixel={pixel_xy}"
            )

        d = int(np.median(valid))

    z = float(d) * float(depth_scale)

    if z < z_min_m or z > z_max_m:
        raise RuntimeError(
            f"Depthが範囲外です．pixel={pixel_xy}, z={z:.4f}"
        )

    X, Y, Z = rs.rs2_deproject_pixel_to_point(
        intr,
        [float(u), float(v)],
        z,
    )

    return np.asarray([X, Y, Z], dtype=np.float32)


def compute_guide_edge_length_3d(
    p0_int: tuple[int, int],
    p1_int: tuple[int, int],
    intr: rs.intrinsics,
    depth_u16: np.ndarray,
    depth_scale: float,
    *,
    plane: Optional[np.ndarray] = None,
    search_radius_px: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, str]:
    """
    ガイドラインとして選択された斜辺 p0_int -> p1_int の3D長さを計算する．

    RANSAC平面がある場合：
        画素レイと平面の交点を使って3D化する．

    RANSAC平面がない場合：
        端点周辺のDepthを使って3D化する．

    Returns
    -------
    edge_length_m:
        斜辺の3D長さ [m]
    p0_cam:
        p0_intに対応する3D点 [m]
    p1_cam:
        p1_intに対応する3D点 [m]
    source:
        "ransac_plane" or "depth_endpoint"
    """
    if plane is not None:
        try:
            p0_cam = intersect_pixel_ray_with_plane(p0_int, intr, plane)
            p1_cam = intersect_pixel_ray_with_plane(p1_int, intr, plane)

            edge_length_m = float(np.linalg.norm(p1_cam - p0_cam))
            return edge_length_m, p0_cam, p1_cam, "ransac_plane"

        except Exception as e:
            print(f"[WARN] guide edge length by RANSAC plane failed: {e}")
            print("[WARN] fallback to endpoint depth.")

    p0_cam = deproject_pixel_with_neighborhood(
        p0_int,
        depth_u16,
        intr,
        depth_scale,
        search_radius_px=search_radius_px,
    )

    p1_cam = deproject_pixel_with_neighborhood(
        p1_int,
        depth_u16,
        intr,
        depth_scale,
        search_radius_px=search_radius_px,
    )

    edge_length_m = float(np.linalg.norm(p1_cam - p0_cam))
    return edge_length_m, p0_cam, p1_cam, "depth_endpoint"

def intersect_pixel_ray_with_plane(
    pixel_xy: tuple[int, int],
    intr: rs.intrinsics,
    plane: np.ndarray,
    *,
    surface_clearance_m: float = 0.0,
) -> np.ndarray:
    """
    画像上の1画素から出るカメラレイと，RANSAC平面の交点を求める．

    plane:
        aX + bY + cZ + d = 0

    surface_clearance_m:
        書籍面より少し手前にしたい場合のオフセット[m]．
        カメラ座標系Z方向に対して手前へずらす簡易補正として使う．
    """
    u, v = int(pixel_xy[0]), int(pixel_xy[1])
    a, b, c, d = [float(x) for x in np.asarray(plane).reshape(4)]

    # Z=1.0 の仮Depthでdeprojectすると，
    # その画素方向のカメラレイ上の点が得られる
    ray = np.asarray(
        rs.rs2_deproject_pixel_to_point(
            intr,
            [float(u), float(v)],
            1.0,
        ),
        dtype=np.float32,
    )

    normal = np.asarray([a, b, c], dtype=np.float32)
    denom = float(np.dot(normal, ray))

    if abs(denom) < 1e-8:
        raise RuntimeError("画素レイと平面がほぼ平行なため，交点を計算できません．")

    scale = -d / denom

    if scale <= 0:
        raise RuntimeError(
            f"平面交点がカメラ後方になりました．pixel={pixel_xy}, scale={scale:.6f}"
        )

    point = ray * float(scale)

    # 必要なら書籍面から少し手前へずらす
    # RealSense座標系では一般にZが前方向なので，手前はZを小さくする
    point[2] -= float(surface_clearance_m)

    return point.astype(np.float32)


def project_cam_point_to_pixel(
    point_cam: np.ndarray,
    intr: rs.intrinsics,
) -> tuple[int, int]:
    """
    カメラ座標系3D点を画像座標に投影する．
    GUI表示と送信目標点の対応確認に使う．
    """
    p = np.asarray(point_cam, dtype=np.float32).reshape(3)

    u, v = rs.rs2_project_point_to_pixel(
        intr,
        [float(p[0]), float(p[1]), float(p[2])],
    )

    return int(round(u)), int(round(v))

# ============================================================
# Main function
# ============================================================

def run_capture_and_pca_depth_space(
    out_dir: str | Path = "captures_depth_space",
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    delta_depth_m: float = 0.03,
    min_area_px: int = 300,
    horizontal_ratio_thr: float = 2.0,
    max_space_width_px: int = 220,
    max_space_area_px: int = 60000,
    trim_max_row_width_px: int = 150,
    trim_row_width_ratio_thr: float = 1.8,
    trim_bottom_search_ratio: float = 0.50,
    trim_min_keep_height_px: int = 60,
    target_dist_m: float = 0.08,
    rotate_180: bool = False,
    # 横長領域除去
    horizontal_kernel_w: int = 300,
    horizontal_kernel_h: int = 30,
    # 書籍面点群収集
    surface_depth_tol_m: float = 0.05,
    surface_side_margin_px: int = 140,
    surface_gap_px: int = 5,
    surface_stride: int = 2,
    # RANSAC
    use_plane_correction: bool = True,
    plane_ransac_threshold_m: float = 0.01,
    plane_ransac_max_iter: int = 300,
    plane_min_inliers: int = 30,
    surface_clearance_m: float = 0.0,
    target_y_from_space_top_px: int = 240,
):
    """
    Depth画像のみから収納スペースを推定し，
    書籍面RANSAC平面で目標点Zを補正して返す．

    Returns
    -------
    angle_rad : float
        収納スペース左側境界の画像上傾き [rad]
    first_target_cam : (3,) ndarray
        カメラ座標系の第一目標点 [m]
    final_target_cam : (3,) ndarray
        カメラ座標系の最終目標点 [m]
    res : dict
        デバッグ情報
    """
    out_dir = Path(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    shot_dir = out_dir / ts
    shot_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. RealSense capture
    # ========================================================
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipe.start(cfg)
        align = rs.align(rs.stream.color)

        for _ in range(30):
            pipe.wait_for_frames()

        frames = pipe.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            raise RuntimeError("RealSense frame取得に失敗しました．")

        depth_frame = depth_filter_like_viewer(depth_frame)

        rgb_bgr = np.asanyarray(color_frame.get_data())
        depth_u16 = np.asanyarray(depth_frame.get_data())

        color_prof = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intr = color_prof.get_intrinsics()
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    finally:
        try:
            pipe.stop()
        except Exception:
            pass

    if rotate_180:
        rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_180)
        depth_u16 = cv2.rotate(depth_u16, cv2.ROTATE_180)
        print("[WARN] rotate_180=True の場合，3D復元にはintrinsics補正が必要です．")
        print("[WARN] 2D確認は可能ですが，3D座標は注意してください．")

    H, W = depth_u16.shape[:2]

    cv2.imwrite(str(shot_dir / "rgb.png"), rgb_bgr)
    np.save(shot_dir / "depth_u16.npy", depth_u16)

    # ========================================================
    # 2. Candidate mask
    # ========================================================
    depth_ref_m = estimate_depth_mode(
        depth_u16,
        depth_scale,
        z_min_m=0.2,
        z_max_m=1.2,
        bins=100,
    )
    print(f"[INFO] depth_ref_m = {depth_ref_m:.4f} m")

    candidate_mask_raw = make_far_space_mask(
        depth_u16,
        depth_scale,
        depth_ref_m,
        delta_depth_m=delta_depth_m,
    )

    cv2.imwrite(str(shot_dir / "candidate_far_mask_raw.png"), candidate_mask_raw * 255)

    candidate_mask = remove_horizontal_structures(
        candidate_mask_raw,
        horizontal_kernel_w=horizontal_kernel_w,
        horizontal_kernel_h=horizontal_kernel_h,
    )

    cv2.imwrite(str(shot_dir / "candidate_far_mask.png"), candidate_mask * 255)

    # ========================================================
    # 3. Select storage space component
    # ========================================================
    selected = select_space_component(
        candidate_mask,
        min_area_px=min_area_px,
        horizontal_ratio_thr=horizontal_ratio_thr,
        max_width_px=max_space_width_px,
        max_area_px=max_space_area_px,
    )

    if selected is None:
        save_binary_target_overlay(
            candidate_mask,
            save_path=shot_dir / "candidate_mask_no_target.png",
        )
        raise RuntimeError("収納スペース候補が見つかりませんでした．")

    selected_mask_raw = selected["mask"]
    bbox_raw = selected["bbox"]

    print(
        f"[INFO] selected raw bbox={bbox_raw}, "
        f"area={selected['area']}, "
        f"aspect_w_h={selected['aspect_w_h']:.3f}"
    )

    cv2.imwrite(str(shot_dir / "selected_space_mask_raw.png"), selected_mask_raw * 255)

    # ========================================================
    # 3.5. 選択領域の下側余分領域を削る
    # ========================================================
    selected_mask = trim_bottom_excess_from_space_mask(
        selected_mask_raw,
        max_row_width_px=trim_max_row_width_px,
        row_width_ratio_thr=trim_row_width_ratio_thr,
        bottom_search_ratio=trim_bottom_search_ratio,
        min_keep_height_px=trim_min_keep_height_px,
    )

    cv2.imwrite(str(shot_dir / "selected_space_mask_cleaned.png"), selected_mask * 255)

    # cleaned後のbboxを再計算する
    ys_clean, xs_clean = np.where(selected_mask > 0)

    if ys_clean.size == 0:
        raise RuntimeError("下側余分領域削除後，収納スペース候補が空になりました．")

    x0 = int(xs_clean.min())
    x1 = int(xs_clean.max())
    y0 = int(ys_clean.min())
    y1 = int(ys_clean.max())

    bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

    print(f"[INFO] cleaned selected bbox={bbox}")

    # selected の情報も，後続のres用に更新しておく
    selected["bbox_raw"] = bbox_raw
    selected["bbox"] = bbox
    selected["mask_raw"] = selected_mask_raw
    selected["mask"] = selected_mask
    selected["area_raw"] = int(np.sum(selected_mask_raw > 0))
    selected["area"] = int(np.sum(selected_mask > 0))
    selected["center"] = (
        float(np.mean(xs_clean)),
        float(np.mean(ys_clean)),
    )
    selected["width"] = int(bbox[2])
    selected["height"] = int(bbox[3])
    selected["aspect_w_h"] = float(bbox[2] / max(bbox[3], 1))

    cv2.imwrite(str(shot_dir / "selected_space_mask.png"), selected_mask * 255)

    # ========================================================
    # 4. Tilted boundary guide line
    # ========================================================
    p0, p1, guide_boundary_pts, guide_side, guide_info = extract_tilted_boundary_line(
        selected_mask,
        default_side="left",
    )

    p0_int = (int(round(float(p0[0]))), int(round(float(p0[1]))))
    p1_int = (int(round(float(p1[0]))), int(round(float(p1[1]))))

    print(f"[INFO] guide_side = {guide_side}")
    print(f"[INFO] left_tilt_from_vertical_deg = {guide_info.get('left_tilt_from_vertical_deg')}")
    print(f"[INFO] right_tilt_from_vertical_deg = {guide_info.get('right_tilt_from_vertical_deg')}")

    dx = float(p1_int[0] - p0_int[0])
    dy = float(p1_int[1] - p0_int[1])
    angle_rad = float(np.arctan2(dy, dx))
    angle_deg = float(np.degrees(angle_rad))

    print(f"[INFO] left boundary angle = {angle_deg:.2f} deg")

    # ========================================================
    # 5. Raw target from guide line
    # ========================================================
    line_coords = enumerate_line_pixels(
        p0_int,
        p1_int,
        width=W,
        height=H,
    )

    pts_line_cam, valid_line_pixels = deproject_line_to_3d(
        line_coords,
        depth_u16,
        intr,
        depth_scale,
        z_min_m=0.1,
        z_max_m=1.5,
    )

    if pts_line_cam.shape[0] == 0:
        save_binary_target_overlay(
            selected_mask,
            line_p0=p0_int,
            line_p1=p1_int,
            save_path=shot_dir / "selected_mask_line_no_depth.png",
        )
        raise RuntimeError("ガイドライン上で有効なDepthが得られませんでした．")

    # ========================================================
    # 5.5. Target point by fixed image height
    # ========================================================
    # selected_mask の上端から target_y_from_space_top_px だけ下を目標高さにする
    ys_selected, xs_selected = np.where(selected_mask > 0)

    if ys_selected.size == 0:
        raise RuntimeError("selected_mask is empty when selecting target point.")

    space_top_y = int(ys_selected.min())
    space_bottom_y = int(ys_selected.max())

    target_y_px = space_top_y + int(target_y_from_space_top_px)
    target_y_px = int(np.clip(target_y_px, space_top_y, space_bottom_y))

    first_target_px = select_target_pixel_on_line_by_y(
        p0_int,
        p1_int,
        target_y_px=target_y_px,
        width=W,
        height=H,
    )

    first_target_cam_raw = deproject_single_pixel_to_3d(
        first_target_px,
        depth_u16,
        intr,
        depth_scale,
        search_radius_px=5,
        z_min_m=0.1,
        z_max_m=1.5,
    )

    print(f"[INFO] space_top_y={space_top_y}, space_bottom_y={space_bottom_y}")
    print(f"[INFO] target_y_from_space_top_px={target_y_from_space_top_px}")
    print(f"[INFO] first_target_px={first_target_px}")
    print("[INFO] first_target_cam_raw =", first_target_cam_raw)

    # ========================================================
    # 6. Book surface plane estimation and Z correction
    # ========================================================
    surface_points, surface_pixels = collect_book_surface_points_around_space(
        depth_u16,
        intr,
        depth_scale,
        bbox,
        depth_ref_m,
        depth_tol_m=surface_depth_tol_m,
        side_margin_px=surface_side_margin_px,
        gap_px=surface_gap_px,
        stride=surface_stride,
        z_min_m=0.1,
        z_max_m=1.5,
    )

    print(f"[INFO] surface_points = {surface_points.shape[0]}")

    plane = None
    inlier_mask = np.zeros((surface_points.shape[0],), dtype=bool)
    plane_correction_used = False

    first_target_cam = first_target_cam_raw.copy()

    # GUI表示用：
    # 初期値は「画像上で選んだ点」．
    # RANSAC補正後は「実際に送信する3D点を再投影した点」に更新する．
    first_target_px_selected = first_target_px
    first_target_px_projected = first_target_px

    if use_plane_correction and surface_points.shape[0] >= 3:
        plane, inlier_mask = fit_plane_ransac(
            surface_points,
            distance_threshold_m=plane_ransac_threshold_m,
            max_iter=plane_ransac_max_iter,
            min_inliers=plane_min_inliers,
            random_seed=0,
        )

        if plane is not None:
            try:
                # 重要：
                # first_target_px から出るカメラレイとRANSAC平面の交点を
                # そのまま送信用の3D目標点にする．
                # これにより，GUI上の点と送信する3D点が対応する．
                first_target_cam = intersect_pixel_ray_with_plane(
                    first_target_px,
                    intr,
                    plane,
                    surface_clearance_m=surface_clearance_m,
                )
                plane_correction_used = True

                print("[INFO] RANSAC plane =", plane)
                print("[INFO] first_target_cam by ray-plane intersection =", first_target_cam)

            except Exception as e:
                print(f"[WARN] plane correction failed: {e}")
                first_target_cam = fallback_correct_point_z_by_median_surface(
                    first_target_cam_raw,
                    surface_points,
                    surface_clearance_m=surface_clearance_m,
                )
                print("[WARN] fallback median Z correction used.")

        else:
            first_target_cam = fallback_correct_point_z_by_median_surface(
                first_target_cam_raw,
                surface_points,
                surface_clearance_m=surface_clearance_m,
            )
            print("[WARN] RANSAC plane not found. fallback median Z correction used.")

    elif use_plane_correction:
        print("[WARN] surface points are insufficient. raw target is used.")

    # 実際に送信する first_target_cam を画像に再投影する．
    # 今後のGUI表示は必ずこちらを使う．
    try:
        first_target_px_projected = project_cam_point_to_pixel(first_target_cam, intr)
    except Exception as e:
        print(f"[WARN] failed to project first_target_cam: {e}")
        first_target_px_projected = first_target_px_selected

    print("[INFO] first_target_px selected  =", first_target_px_selected)
    print("[INFO] first_target_px projected =", first_target_px_projected)

    final_target_cam = first_target_cam.copy()

    # ========================================================
    # 6.5. Guide edge length estimation
    # ========================================================
    guide_edge_length_m = None
    guide_edge_length_mm = None
    guide_edge_p0_cam = None
    guide_edge_p1_cam = None
    guide_edge_length_source = None

    try:
        guide_edge_length_m, guide_edge_p0_cam, guide_edge_p1_cam, guide_edge_length_source = compute_guide_edge_length_3d(
            p0_int,
            p1_int,
            intr,
            depth_u16,
            depth_scale,
            plane=plane if plane_correction_used else None,
            search_radius_px=5,
        )

        guide_edge_length_mm = float(guide_edge_length_m * 1000.0)

        print(
            f"[INFO] guide_edge_length = {guide_edge_length_m:.4f} m "
            f"({guide_edge_length_mm:.1f} mm), "
            f"source={guide_edge_length_source}"
        )
        print("[INFO] guide_edge_p0_cam =", guide_edge_p0_cam)
        print("[INFO] guide_edge_p1_cam =", guide_edge_p1_cam)

    except Exception as e:
        print(f"[WARN] guide edge length estimation failed: {e}")

    # ========================================================
    # 7. Debug images
    # ========================================================
    selected_mask_target_overlay_path = shot_dir / "selected_mask_target_overlay.png"
    candidate_mask_target_overlay_path = shot_dir / "candidate_mask_target_overlay.png"
    overlay_path = shot_dir / "depth_space_overlay.png"
    surface_overlay_path = shot_dir / "surface_points_overlay.png"

    save_binary_target_overlay(
        selected_mask,
        line_p0=p0_int,
        line_p1=p1_int,
        target_px=first_target_px_projected,
        save_path=selected_mask_target_overlay_path,
    )

    save_binary_target_overlay(
        candidate_mask,
        line_p0=p0_int,
        line_p1=p1_int,
        target_px=first_target_px_projected,
        save_path=candidate_mask_target_overlay_path,
    )

    save_debug_overlay(
        rgb_bgr,
        candidate_mask,
        selected_mask,
        p0_int,
        p1_int,
        first_target_px_projected,
        overlay_path,
    )

    save_surface_points_overlay(
        rgb_bgr,
        surface_pixels,
        inlier_mask if surface_points.shape[0] == inlier_mask.shape[0] else None,
        surface_overlay_path,
    )

    show_image_from_path(
    "selected_mask_target_overlay",
    selected_mask_target_overlay_path,
    wait=True,
    )

    show_image_from_path(
        "candidate_mask_target_overlay",
        candidate_mask_target_overlay_path,
        wait=True,
    )

    show_image_from_path(
        "depth_space_overlay",
        overlay_path,
        wait=True,
    )

    show_image_from_path(
        "surface_points_overlay",
        surface_overlay_path,
        wait=True,
    )

    cv2.destroyAllWindows()

    # ========================================================
    # 8. res
    # ========================================================
    res = {
        "line_p0": p0_int,
        "line_p1": p1_int,
        "sub_line_p0": p0_int,
        "sub_line_p1": p1_int,
        "right_is_tilted": False,
        "is_right_half": False,
        "pair_indices": None,
        "centers": None,

        "guide_side": guide_side,
        "guide_info": guide_info,
        "left_tilt_from_vertical_deg": guide_info.get("left_tilt_from_vertical_deg"),
        "right_tilt_from_vertical_deg": guide_info.get("right_tilt_from_vertical_deg"),

        "space_bbox": bbox,
        "space_center_px": selected["center"],
        "space_area": selected["area"],
        "space_aspect_w_h": selected["aspect_w_h"],

        "first_target_px": first_target_px_projected,
        "first_target_px_selected": first_target_px_selected,
        "first_target_px_projected": first_target_px_projected,
        "first_target_cam_raw": first_target_cam_raw,
        "first_target_cam": first_target_cam,
        "final_target_cam": final_target_cam,

        "depth_ref_m": depth_ref_m,
        "delta_depth_m": delta_depth_m,
        "target_dist_m": target_dist_m,

        "use_plane_correction": use_plane_correction,
        "plane_correction_used": plane_correction_used,
        "plane": plane.tolist() if plane is not None else None,
        "surface_points_count": int(surface_points.shape[0]),
        "surface_inliers_count": int(np.sum(inlier_mask)) if inlier_mask is not None else 0,
        "surface_clearance_m": surface_clearance_m,

        "overlay_path": str(overlay_path),
        "selected_mask_target_overlay_path": str(selected_mask_target_overlay_path),
        "candidate_mask_target_overlay_path": str(candidate_mask_target_overlay_path),
        "surface_points_overlay_path": str(surface_overlay_path),
        "shot_dir": str(shot_dir),
        
        "guide_edge_length_m": guide_edge_length_m,
        "guide_edge_length_mm": guide_edge_length_mm,
        "guide_edge_length_source": guide_edge_length_source,
        "guide_edge_p0_cam": guide_edge_p0_cam.tolist() if guide_edge_p0_cam is not None else None,
        "guide_edge_p1_cam": guide_edge_p1_cam.tolist() if guide_edge_p1_cam is not None else None,
    }

    print("[INFO] first_target_px       =", first_target_px)
    print("[INFO] first_target_cam_raw  =", first_target_cam_raw)
    print("[INFO] first_target_cam      =", first_target_cam)
    print("[INFO] final_target_cam      =", final_target_cam)
    print("[INFO] plane_correction_used =", plane_correction_used)
    print("[INFO] files saved under     =", shot_dir)

    return angle_rad, first_target_cam, res