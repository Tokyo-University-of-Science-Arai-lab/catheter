# modules/axis_centerline.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import cv2

__all__ = ["axis_x_on_centerline"]

def _pca_major_axis(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    points_xy: (N,2) with columns (x,y). Return (center(cx,cy), unit major axis u=(ux,uy)).
    """
    mean = points_xy.mean(axis=0)
    centered = points_xy - mean
    # 2x2 covariance (unbiasedでなくてOK)
    cov = centered.T @ centered / max(len(points_xy) - 1, 1)
    # eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)  # ascending
    u = vecs[:, 1]  # eigenvector for largest eigenvalue
    # 正規化
    n = np.linalg.norm(u)
    if n < 1e-12:
        u = np.array([1.0, 0.0], dtype=np.float32)
    else:
        u = (u / n).astype(np.float32)
    return mean.astype(np.float32), u

def axis_x_on_centerline(
    mask: np.ndarray,
    y_top: int = 120,
    y_bottom: int = 400,
    img_hw: Optional[Tuple[int, int]] = None,  # (H,W)
) -> Tuple[Optional[int], Optional[int]]:
    """
    マスクの『中心軸』(主成分の長軸) を画像座標で取り直し，
    その直線と y=y_top / y=y_bottom の交点の x を返す。
    返り値は (top, bottom)。必要に応じて img_hw で [0, W-1] にクリップ。

    戻り値:
      top:    y=y_top における交点 x（int）/ 求まらなければ None
      bottom: y=y_bottom における交点 x（int）/ 求まらなければ None
    """
    # ピクセル座標列を抽出
    mm = (mask > 0).astype(np.uint8)
    ys, xs = np.where(mm > 0)
    if len(xs) == 0:
        return None, None
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)  # (N,2) = (x,y)

    center, u = _pca_major_axis(pts)   # center=(cx,cy), u=(ux,uy)
    cx, cy = float(center[0]), float(center[1])
    ux, uy = float(u[0]), float(u[1])

    def x_at_y(y_target: float) -> Optional[float]:
        # 直線: p(t) = c + t*u。  y: cy + t*uy = y_target -> t = (y_target - cy)/uy
        if abs(uy) < 1e-8:
            return None
        t = (y_target - cy) / uy
        x = cx + t * ux
        return x

    top_x = x_at_y(float(y_top))
    bottom_x = x_at_y(float(y_bottom))

    # 画面幅にクリップ（任意）
    if img_hw is not None:
        H, W = img_hw
        def clip_or_none(x):
            if x is None:
                return None
            return int(np.clip(round(x), 0, W - 1))
        return clip_or_none(top_x), clip_or_none(bottom_x)

    def round_or_none(x):
        return None if x is None else int(round(x))
    return round_or_none(top_x), round_or_none(bottom_x)
