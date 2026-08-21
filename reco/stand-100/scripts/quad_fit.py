#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
マスクを4角形（矩形/台形）に整形する2方式の実装。

方式A: min_area_rect
    cv2.minAreaRect + boxPoints。真の長方形を仮定した最小外接矩形。
    正面から見た形状には強いが、側面が見えてパースがついた台形状の
    シルエットには、実際の輪郭とズレた矩形になりやすい。

方式B: approx_poly_quad
    convexHull -> approxPolyDP のepsilonを二分探索し、
    頂点数がちょうど4になる近似多角形を探す。
    台形などの非平行四辺形にも追従できるが、epsilonが4点にきれいに
    収束しない形状（凹みが強い/ノイズが多い等）では失敗しうる
    （その場合は None を返す＝呼び出し側でフォールバックする設計）。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

__all__ = [
    "fit_quad_min_area_rect",
    "fit_quad_approx_poly",
    "quad_to_mask",
    "largest_contour",
]


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """0/1 または 0/255 の2値マスクから最大の外側輪郭を返す。無ければ None。"""
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_quad_min_area_rect(mask: np.ndarray) -> Optional[np.ndarray]:
    """マスク -> 最小外接矩形の4頂点 (4,2) float32。マスクが空なら None。"""
    cnt = largest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < 1e-6:
        return None
    rect = cv2.minAreaRect(cnt)  # (cx,cy),(w,h),angle
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def fit_quad_approx_poly(
    mask: np.ndarray,
    n_iter: int = 60,
) -> Optional[np.ndarray]:
    """
    マスク -> convexHullをapproxPolyDPで4点に近似した4頂点 (4,2) float32。

    epsilon（弧長に対する割合）を二分探索し、頂点数がちょうど4になる
    最小のepsilonを探す。凸包の頂点数はepsilonに対して単調非増加という
    前提で二分探索している。ちょうど4点に収束しなかった場合は None。
    """
    cnt = largest_contour(mask)
    if cnt is None or cv2.contourArea(cnt) < 1e-6:
        return None

    hull = cv2.convexHull(cnt)
    if len(hull) <= 4:
        # 凸包自体が4点以下 -> そのまま返す（3点なら退化として None）
        if len(hull) == 4:
            return hull.reshape(4, 2).astype(np.float32)
        return None

    peri = cv2.arcLength(hull, True)
    if peri < 1e-6:
        return None

    lo, hi = 0.0, 1.0
    found = None
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(hull, mid * peri, True)
        n = len(approx)
        if n > 4:
            lo = mid
        elif n < 4:
            hi = mid
        else:
            found = approx
            # さらに小さいepsilon側にも4点解があるかもしれないが、
            # ここでは最初に見つかった4点解で確定させる
            break

    if found is None:
        return None
    return found.reshape(4, 2).astype(np.float32)


def quad_to_mask(quad: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """4頂点 (4,2) から、同じ形状の0/1マスクを塗りつぶしで作る。"""
    h, w = shape_hw
    out = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(out, [pts], 1)
    return out
