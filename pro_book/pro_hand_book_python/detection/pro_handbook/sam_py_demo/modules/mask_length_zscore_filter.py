#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List
import numpy as np
import cv2

__all__ = ["major_axis_length", "prune_by_major_axis_zscore"]

def _to_u8_mask(m: np.ndarray) -> np.ndarray:
    if m.dtype == np.bool_:
        return (m.astype(np.uint8) * 255)
    if m.dtype != np.uint8:
        return m.astype(np.uint8)
    return m

def major_axis_length(m: np.ndarray) -> float:
    """マスク(任意dtype)の回転外接矩形(minAreaRect)の長辺長を返す(px)。"""
    u8 = _to_u8_mask(m)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(cnt)  # w,h は辺長
    return float(max(w, h))

def prune_by_major_axis_zscore(
    masks: List[np.ndarray],
    z_lower: float = -2.0,     # z ≤ z_lower を除去
    min_len_px: float = 0.0,   # 絶対長下限（σ≈0時の保険）
) -> List[np.ndarray]:
    """
    各マスクの主軸長Lを集計し z-score を計算。
    z ≤ z_lower のマスクを削除。さらに L < min_len_px も削除。
    """
    if not masks:
        return []

    lengths = np.array([major_axis_length(m) for m in masks], dtype=np.float32)
    mu = float(lengths.mean()) if lengths.size else 0.0
    sigma = float(lengths.std(ddof=0)) if lengths.size else 0.0

    kept = []
    if sigma < 1e-6:
        # ばらつき無し：z-score無意味 → 絶対長のみで判定
        for m, L in zip(masks, lengths):
            if L >= min_len_px:
                kept.append(m)
        return kept

    z = (lengths - mu) / (sigma + 1e-12)
    for m, zi, L in zip(masks, z, lengths):
        if (zi > z_lower) and (L >= min_len_px):
            kept.append(m)
    return kept
