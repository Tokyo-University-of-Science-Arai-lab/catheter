# modules/sam_style_dedup.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

__all__ = [
    "stability_score_from_lowres",
    "mask_bbox_xyxy",
    "bbox_iou_xyxy",
    "mask_iou_downsample",
    "pick_best_lowres_and_score",
    "candidate_from_lowres",
    "dedup_sam_style",
]

# -------------------------------
# 基本ユーティリティ
# -------------------------------
def _is_prob(arr: np.ndarray) -> bool:
    """配列が確率[0,1]っぽいかを判定。"""
    xmin, xmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    return (xmin >= -1e-6) and (xmax <= 1.0 + 1e-6)

def _logit(p: float) -> float:
    """数値安定な logit。"""
    eps = 1e-6
    p = min(max(p, eps), 1.0 - eps)
    return np.log(p / (1.0 - p))

def mask_bbox_xyxy(m: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """bool/0-255 マスク → (x0,y0,x1,y1)。空なら None。"""
    if m.dtype != np.bool_:
        m = (m.astype(np.uint8) > 0)
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0, y0, x1 + 1, y1 + 1)

def bbox_iou_xyxy(a: Optional[Tuple[int,int,int,int]], b: Optional[Tuple[int,int,int,int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return 0.0 if ua <= 0 else inter / ua

def mask_iou_downsample(a: np.ndarray, b: np.ndarray, side: int = 64) -> float:
    """ピクセルIoUを低解像度で近似（高速化）。"""
    au = cv2.resize((a > 0).astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST) > 0
    bu = cv2.resize((b > 0).astype(np.uint8), (side, side), interpolation=cv2.INTER_NEAREST) > 0
    inter = np.logical_and(au, bu).sum()
    union = np.logical_or(au, bu).sum()
    return 0.0 if union == 0 else inter / union

# -------------------------------
# 安定性スコア（sigmoid不要版）
# -------------------------------
def stability_score_from_lowres(
    lowres_256: np.ndarray,
    delta: float = 0.05,
    is_logits: Optional[bool] = None,
) -> float:
    """
    SAM AMGの“しきい値ゆらぎ耐性”に相当。
    - logits入力なら、確率0.5±deltaをlogitに変換した閾値で直接二値化
    - 確率入力なら、そのまま 0.5±delta で二値化
    """
    if is_logits is None:
        is_logits = not _is_prob(lowres_256)

    if is_logits:
        t_lo = _logit(0.5 - delta)   # 例: δ=0.05 → -0.20067...
        t_hi = _logit(0.5 + delta)   # 例: δ=0.05 → +0.20067...
        m_lo = (lowres_256 > t_lo)
        m_hi = (lowres_256 > t_hi)
    else:
        p = lowres_256
        m_lo = (p > (0.5 - delta))
        m_hi = (p > (0.5 + delta))

    inter = np.logical_and(m_lo, m_hi).sum()
    union = np.logical_or(m_lo, m_hi).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# -------------------------------
# SAM出力 → 「最良候補のlow-res + スコア」
# -------------------------------
def pick_best_lowres_and_score(
    outs: Dict[str, np.ndarray],
    delta: float = 0.05
) -> Tuple[np.ndarray, float, bool]:
    """
    decoder出力dict(outs)から、K候補のうち IoU予測最大のlow-resを選び、
    stability scoreと掛け合わせた複合スコアを返す（本家AMG風）。
    Returns:
        lowres_256: (256,256) logits/prob
        score:      float  (= iou_pred_best * stability)
        is_logits:  bool   (low_res_masksならTrue, masksならFalseの想定)
    """
    # IoU予測最大のindex
    if "iou_predictions" in outs:
        ious = outs["iou_predictions"][0]  # (K,)
        best = int(np.argmax(ious))
        iou_pred_best = float(ious[best])
    else:
        best = 0
        iou_pred_best = 1.0

    # low_res_masks があれば logits、なければ masks(確率) とみなす
    if "low_res_masks" in outs:
        lowres_256 = outs["low_res_masks"][0, best]
        is_logits = True
    else:
        lowres_256 = outs["masks"][0, best]  # prob in [0,1]
        is_logits = False

    stab = stability_score_from_lowres(lowres_256, delta=delta, is_logits=is_logits)
    score = iou_pred_best * stab
    return lowres_256, float(score), bool(is_logits)

def candidate_from_lowres(
    lowres_256: np.ndarray,
    score: float,
    postprocess_fn,
    orig_hw: Tuple[int,int],
    new_hw: Tuple[int,int],
) -> Dict:
    """
    low-res(256x256)→原寸2値マスクを作り、面積/矩形/スコアをまとめたcandidate辞書に。
    Args:
        postprocess_fn: postprocess_like_sam(lowres, orig_hw, new_hw, is_logits=...) 互換関数
    """
    m_u8 = postprocess_fn(lowres_256, orig_hw, new_hw)  # 0/255 を想定
    m = (m_u8 > 0)
    return {
        "mask":  m,
        "area":  int(m.sum()),
        "bbox":  mask_bbox_xyxy(m),
        "score": float(score),
    }

# -------------------------------
# “SAM風”の重複除去（スコア順 + しきい値）
# -------------------------------
def dedup_sam_style(
    candidates: List[Dict],
    iou_thr: float = 0.90,
    bbox_gate: float = 0.50,
    max_keep: int = 100,
    downsample_side: int = 64,
) -> List[Dict]:
    """
    本家AMGに近い重複除去：
      1) score降順で走査
      2) 既存採択とbbox IoU < bbox_gate なら精査スキップ（速い）
      3) 近いものだけピクセルIoU（ダウンサンプル近似） > iou_thr なら捨てる
    """
    cands = sorted(candidates, key=lambda d: d["score"], reverse=True)
    selected: List[Dict] = []
    for c in cands:
        m, bb = c.get("mask"), c.get("bbox")
        if m is None or bb is None:
            continue
        if c.get("area", 0) <= 0:
            continue

        duplicate = False
        for s in selected:
            if bbox_iou_xyxy(bb, s["bbox"]) < bbox_gate:
                continue
            if mask_iou_downsample(m, s["mask"], side=downsample_side) > iou_thr:
                duplicate = True
                break

        if not duplicate:
            selected.append(c)
            if len(selected) >= max_keep:
                break
    return selected
