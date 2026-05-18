# modules/axis_filter.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import cv2
import math

__all__ = [
    "AxisInfo",
    "get_axis_info",
    "_axis_feats_from_mask",
    "filter_by_angle_window",          # 互換用（raw角ベース）
    "filter_keep_vertical_by_fold",    # 推奨：fold角ベース
]

@dataclass
class AxisInfo:
    mask: np.ndarray          # (H,W) bool
    center: np.ndarray        # (2,) float32
    u: np.ndarray             # (2,) float32  長辺方向の単位ベクトル
    L: float                  # 長辺長さ（minAreaRectの長辺）
    W: float                  # 短辺長さ（minAreaRectの短辺）
    angle_deg_raw: float      # 水平基準の角度 ∈ [0,180)  0=水平, 90=垂直
    angle_deg_fold: float     # 折り畳み角 ∈ [0,90]      0=水平, 90=垂直
    endpoints: np.ndarray     # (2,2) float32 中心±L/2*u
    box: np.ndarray           # (4,2) float32 回転外接矩形4点

def _rect_longside_angles(rect) -> Tuple[float, float]:
    """
    minAreaRectの“幅高入替＆角度跳び”を吸収して、長辺基準の角度に直す。
      rect = ((cx,cy),(w,h),a), a∈[-90,0)
    Returns:
      ang_raw  ∈ [0,180)  0=水平, 90=垂直
      ang_fold ∈ [0,90]   0=水平, 90=垂直（180°同値を折り畳み）
    """
    (_, _), (w, h), a = rect
    a_long = a if (w >= h) else (a + 90.0)      # 長辺の向きに統一
    ang_raw  = (a_long + 180.0) % 180.0
    ang_fold = ang_raw if ang_raw <= 90.0 else (180.0 - ang_raw)
    return float(ang_raw), float(ang_fold)

def get_axis_info(m: np.ndarray) -> Optional[AxisInfo]:
    """
    2値マスクから主軸（回転外接矩形の長辺）情報を返す。輪郭が無ければ None。
    - 角度は minAreaRect の“長辺基準”に正規化し、fold角も併記。
    - 端点は L/2 だけ中心から u に沿って前後に取る。
    """
    m_bool = m.astype(bool, copy=False)
    if not m_bool.any():
        return None
    u8 = (m_bool.astype(np.uint8) * 255)

    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(cnt)                      # ((cx,cy),(w,h),angle[-90,0))
    box  = cv2.boxPoints(rect).astype(np.float32)    # (4,2)

    # 主軸ベクトル u は box の長い辺から取得（boxの点順は一定でないため辺長で選ぶ）
    v0 = box[1] - box[0]
    v1 = box[2] - box[1]
    if np.linalg.norm(v0) >= np.linalg.norm(v1):
        major_v = v0
    else:
        major_v = v1

    norm = float(np.linalg.norm(major_v))
    if norm < 1e-6:
        return None
    u = (major_v / (norm + 1e-12)).astype(np.float32)

    # 中心・長短辺長（minAreaRectの w,h を採用）
    (cx, cy), (w, h), _ = rect
    c = np.array([cx, cy], dtype=np.float32)
    L = float(max(w, h))
    W = float(min(w, h))
    if L < 1e-6:
        return None

    # 端点
    halfL = 0.5 * L
    p0 = c - halfL * u
    p1 = c + halfL * u
    endpoints = np.stack([p0, p1], axis=0).astype(np.float32)

    # 角度（長辺基準に正規化）と折り畳み角
    ang_raw, ang_fold = _rect_longside_angles(rect)

    return AxisInfo(
        mask=m_bool, center=c, u=u, L=L, W=W,
        angle_deg_raw=ang_raw, angle_deg_fold=ang_fold,
        endpoints=endpoints, box=box
    )

def _axis_feats_from_mask(m: np.ndarray) -> Optional[Dict[str, object]]:
    """
    互換ラッパー: 既存コードが期待する dict を返す（キー名は従来通り）。
      keys = ["mask","center","u","L","W","box","tmin","tmax"]
    """
    info = get_axis_info(m)
    if info is None:
        return None
    ts = info.box @ info.u
    return {
        "mask":   info.mask,
        "center": info.center,
        "u":      info.u,
        "L":      info.L,
        "W":      info.W,
        "box":    info.box,
        "tmin":   float(ts.min()),
        "tmax":   float(ts.max()),
    }

# ---- フィルタ群 ----

def filter_by_angle_window(
    masks: List[np.ndarray],
    keep_min_deg: float = 45.0,
    keep_max_deg: float = 135.0,
) -> Tuple[List[np.ndarray], List[AxisInfo]]:
    """
    【互換用】主軸角（raw）angle∈[0,180) が (keep_min_deg, keep_max_deg) に入るものだけ残す。
    既定: 45°< angle <135° ＝“縦寄り”を残す（水平0°/180°は除外）。
    ※ 推奨は fold角ベースの filter_keep_vertical_by_fold を使用。
    Returns: (kept_masks, kept_infos)
    """
    kept_m, kept_i = [], []
    for m in masks:
        info = get_axis_info(m)
        if info is None:
            continue
        ang = info.angle_deg_raw
        if (keep_min_deg < ang) and (ang < keep_max_deg):
            kept_m.append(info.mask)
            kept_i.append(info)
    return kept_m, kept_i

def filter_keep_vertical_by_fold(
    masks: List[np.ndarray],
    min_fold_deg: float = 70.0,   # 80は厳しめなので既定は70°
    min_aspect: float = 1.3,      # 正方形～曖昧な形を除外 (L/W >= 1.3)
    min_len_px: float = 20.0,     # 極小ノイズ除去
) -> Tuple[List[np.ndarray], List[AxisInfo]]:
    """
    推奨：折り畳み角（0=水平,90=垂直）で“縦だけ残す”。minAreaRectの角度跳びの影響を受けにくい。
    - min_fold_deg を上げるほど“より垂直のみ”に厳しくなる（例: 65〜75の範囲で調整）。
    - min_aspect で向きが定まらないブロブを除外。
    """
    kept_m, kept_i = [], []
    for m in masks:
        info = get_axis_info(m)
        if info is None:
            continue
        if (info.angle_deg_fold >= min_fold_deg and
            info.L / max(info.W, 1e-6) >= min_aspect and
            info.L >= min_len_px):
            kept_m.append(info.mask)
            kept_i.append(info)
    return kept_m, kept_i
