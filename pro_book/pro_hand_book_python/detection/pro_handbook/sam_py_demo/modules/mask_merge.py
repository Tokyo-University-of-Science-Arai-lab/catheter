# modules/mask_merge.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Optional
import numpy as np
import cv2

# 互換のため axis_filter の _axis_feats_from_mask を使う
from modules.axis_angle_filter import _axis_feats_from_mask

__all__ = ["merge_coaxial_rect_masks"]

def _interval_along(u: np.ndarray, box: np.ndarray) -> tuple[float, float]:
    """回転矩形の4頂点を主軸 u に射影した最小/最大値"""
    ts = box @ u
    return float(ts.min()), float(ts.max())

def merge_coaxial_rect_masks(
    masks: List[np.ndarray],
    axis_angle_tol_deg: float = 8.0,   # 軸の平行許容（度）
    offset_width_factor: float = 0.1,  # 軸からの直交距離許容=平均幅×係数
    gap_len_factor: float = 0.25,      # 軸方向の隙間許容=平均長さ×係数
    rectify: bool = False              # Trueなら回転長方形で橋渡し（形を整える）
) -> List[np.ndarray]:
    """
    軸がほぼ同じ＆同一直線上に並ぶ“本の背”みたいな長方形マスク群を統合。

    Parameters
    ----------
    masks : List[np.ndarray]   # bool/uint8 どちらでもOK（True/1が前景）
    axis_angle_tol_deg : float # 主軸のなす角の許容（平行性）
    offset_width_factor : float# 軸からの直交距離の許容（平均幅×係数）
    gap_len_factor : float     # 主軸方向のすき間の許容（平均長×係数）
    rectify : bool             # Trueで橋渡しポリゴンで形を整える

    Returns
    -------
    List[np.ndarray] : 統合後のマスク
    """
    if not masks:
        return []

    feats = []
    for m in masks:
        f = _axis_feats_from_mask(m)
        if f is not None:
            # 入力マスクは bool 化しておく
            f["mask"] = (m.astype(bool, copy=False))
            feats.append(f)
    if not feats:
        return []

    used = [False] * len(feats)
    merged = []
    cos_tol = float(np.cos(np.deg2rad(axis_angle_tol_deg)))

    for i, fi in enumerate(feats):
        if used[i]:
            continue
        ui = fi["u"]; ci = fi["center"]
        group_idx = [i]; used[i] = True
        g_tmin, g_tmax = _interval_along(ui, fi["box"])
        g_widths = [fi["W"]]

        for j, fj in enumerate(feats):
            if used[j]:
                continue
            # 1) 主軸の平行性
            if abs(float(np.dot(ui, fj["u"]))) < cos_tol:
                continue
            # 2) 軸からの直交距離（線分と点の距離）
            v = fj["center"] - ci
            off = abs(ui[0]*v[1] - ui[1]*v[0])
            off_thr = offset_width_factor * 0.5 * (fi["W"] + fj["W"])
            if off > off_thr:
                continue
            # 3) 主軸方向ギャップ
            tj_min, tj_max = _interval_along(ui, fj["box"])
            if tj_max < g_tmin:
                gap = g_tmin - tj_max
            elif tj_min > g_tmax:
                gap = tj_min - g_tmax
            else:
                gap = 0.0
            gap_thr = gap_len_factor * 0.5 * (fi["L"] + fj["L"])
            if gap > gap_thr:
                continue

            used[j] = True
            group_idx.append(j)
            g_tmin = min(g_tmin, tj_min); g_tmax = max(g_tmax, tj_max)
            g_widths.append(fj["W"])

        H, W = masks[0].shape
        if not rectify:
            # 単純 OR
            uni = np.zeros((H, W), np.uint8)
            for k in group_idx:
                uni |= feats[k]["mask"].astype(np.uint8)
            merged.append(uni > 0)
        else:
            # 橋渡し：回転長方形を再構成して塗りつぶす
            width = float(np.median(g_widths))
            c_mean = np.mean([feats[k]["center"] for k in group_idx], axis=0)
            L = float(g_tmax - g_tmin)
            n = np.array([-ui[1], ui[0]], dtype=np.float32)   # 主軸に直交
            t_center = float(np.dot(c_mean, ui))
            t0 = t_center - L/2.0; t1 = t_center + L/2.0
            p0 = ui * t0 + n * (-width/2.0)
            p1 = ui * t1 + n * (-width/2.0)
            p2 = ui * t1 + n * ( width/2.0)
            p3 = ui * t0 + n * ( width/2.0)
            poly = np.stack([p0, p1, p2, p3], axis=0).astype(np.float32)
            canvas = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(canvas, poly.astype(np.int32), 1)
            merged.append(canvas > 0)

    return merged
