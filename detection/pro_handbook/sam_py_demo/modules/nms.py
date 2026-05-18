#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMSユーティリティ（SAM/AMG想定）

- ボックスNMS（グリッド/クロップ内・クロップ間）
- マスクIoUベースのデデュープ（"ほぼ同一"だけ潰す）
- 品質フィルタ（pred_iou / stability）

cands は以下のキーを含む dict のリストを想定：
  {
    "mask": np.ndarray(bool or 0/1/0-255)  # (H, W)
    "bbox": (x, y, w, h)                   # 省略可（なければ計算）
    "score": float                         # 任意（なければ pred_iou を使用）
    "pred_iou": float                      # 任意
    "stability": float                     # 任意
    "area": int                            # 任意（なければ mask.sum()）
    "crop_id": int                         # AMG のクロップ識別子（任意）
    "crop_level": int                      # 0: 全体 / 大きい値ほどズームイン（任意）
  }

典型的な使い方：

    from modules.nms import (
        quality_filter, nms_per_crop_then_cross, dedup_by_mask_iou
    )

    cands = quality_filter(cands, pred_iou_thr=0.88, stability_thr=0.95)
    cands = nms_per_crop_then_cross(
        cands,
        per_crop_iou=0.7,
        cross_crop_iou=0.7,
        score_key="pred_iou",           # or "score"
        prefer_zoom_in=True,
    )
    selected = dedup_by_mask_iou(
        cands,
        iou_thr=0.97,
        bbox_gate=0.60,
        downsample_side=192,
        score_key="pred_iou"
    )[:300]

注意：
- 本モジュールは速度重視の実装（cv2.resizeで固定サイズに縮小してIoU計算）
- 形状歪みに敏感な用途では downsample_side を大きくする or 原寸IoUに切替
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Iterable, Optional
from collections import defaultdict
import numpy as np
import cv2

__all__ = [
    "quality_filter",
    "nms_box_greedy",
    "nms_per_crop_then_cross",
    "dedup_by_mask_iou",
]

# ============================================================
# Helpers
# ============================================================

def _ensure_u8_mask(m: np.ndarray) -> np.ndarray:
    """bool/0-1/0-255 どれでも受け取り、uint8(0/255)に正規化。"""
    if m.dtype == np.bool_:
        return (m.astype(np.uint8) * 255)
    if m.dtype == np.uint8:
        if m.max() <= 1:
            return (m * 255).astype(np.uint8)
        return m
    return (m > 0).astype(np.uint8) * 255


def _mask_area(m: np.ndarray) -> int:
    return int((m > 0).sum())


def _bbox_from_mask(m: np.ndarray) -> Tuple[int, int, int, int]:
    mu8 = _ensure_u8_mask(m)
    x, y, w, h = cv2.boundingRect(mu8)
    return int(x), int(y), int(w), int(h)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = a
    x2, y2, w2, h2 = b
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter <= 0:
        return 0.0
    u = w1 * h1 + w2 * h2 - inter
    return inter / u if u > 0 else 0.0


def _mask_iou_fast(a: np.ndarray, b: np.ndarray, side: int = 192) -> float:
    """固定サイズ(side×side)に最近傍縮小してIoUを近似。"""
    au8 = _ensure_u8_mask(a)
    bu8 = _ensure_u8_mask(b)
    if au8.shape != (side, side):
        au8 = cv2.resize(au8, (side, side), interpolation=cv2.INTER_NEAREST)
    if bu8.shape != (side, side):
        bu8 = cv2.resize(bu8, (side, side), interpolation=cv2.INTER_NEAREST)
    a_bin = au8 > 0
    b_bin = bu8 > 0
    inter = np.logical_and(a_bin, b_bin).sum()
    u = a_bin.sum() + b_bin.sum() - inter
    return float(inter) / float(u) if u > 0 else 0.0


def _score_of(c: Dict, score_key: str) -> float:
    if score_key in c and c[score_key] is not None:
        return float(c[score_key])
    # フォールバック順
    for k in ("pred_iou", "score", "stability"):
        if k in c and c[k] is not None:
            return float(c[k])
    # 最後の手段：面積
    m = c.get("mask")
    return float(_mask_area(m)) if m is not None else 0.0


def _ensure_bbox_area(c: Dict) -> None:
    if c.get("bbox") is None:
        m = c.get("mask")
        if m is None:
            raise ValueError("candidate has neither bbox nor mask")
        c["bbox"] = _bbox_from_mask(m)
    if c.get("area") is None:
        m = c.get("mask")
        if m is not None:
            c["area"] = _mask_area(m)

# ============================================================
# Quality filter
# ============================================================

def quality_filter(
    cands: List[Dict],
    pred_iou_thr: Optional[float] = None,
    stability_thr: Optional[float] = None,
) -> List[Dict]:
    """一次ふるい（品質）
    - **面積フィルタやマスク有無チェックは行いません**（デコーダ側で実施済み前提）。
    - `pred_iou_thr` と `stability_thr` が指定された場合のみ、その下限でふるいます。
    """
    out = []
    for c in cands:
        if pred_iou_thr is not None:
            p = c.get("pred_iou")
            if p is None or p < pred_iou_thr:
                continue
        if stability_thr is not None:
            s = c.get("stability")
            if s is None or s < stability_thr:
                continue
        out.append(c)
    return out

# ============================================================
# Box NMS (greedy)
# ============================================================

def nms_box_greedy(
    cands: List[Dict],
    iou_thr: float = 0.7,
    score_key: str = "score",
) -> List[Dict]:
    """ボックスIoUによる貪欲NMS（AMGの基本）。
    スコア降順で走査し、既採用と bbox IoU >= iou_thr なら抑制。
    """
    if not cands:
        return []
    for c in cands:
        _ensure_bbox_area(c)
    order = sorted(range(len(cands)), key=lambda i: _score_of(cands[i], score_key), reverse=True)
    keep_idx: List[int] = []
    for i in order:
        bi = cands[i]["bbox"]
        ok = True
        for j in keep_idx:
            bj = cands[j]["bbox"]
            if _bbox_iou(bi, bj) >= iou_thr:
                ok = False
                break
        if ok:
            keep_idx.append(i)
    return [cands[i] for i in keep_idx]


def nms_per_crop_then_cross(
    cands: List[Dict],
    per_crop_iou: float = 0.7,
    cross_crop_iou: float = 0.7,
    score_key: str = "score",
    prefer_zoom_in: bool = True,
) -> List[Dict]:
    """AMG風：
    (1) クロップ内NMS → (2) クロップ間NMS。
    - prefer_zoom_in=True なら、クロップ間競合時は crop_level が大きい（ズームイン側）を優先。
      crop_level が無い個体は 0 とみなす。
    """
    if not cands:
        return []
    for c in cands:
        _ensure_bbox_area(c)

    # 1) per-crop NMS
    by_crop: Dict[int, List[Dict]] = defaultdict(list)
    for c in cands:
        by_crop[int(c.get("crop_id", 0))].append(c)

    kept: List[Dict] = []
    for _, group in by_crop.items():
        kept.extend(nms_box_greedy(group, iou_thr=per_crop_iou, score_key=score_key))

    if len(by_crop) <= 1:
        return kept

    # 2) cross-crop NMS with preference
    for c in kept:
        _ensure_bbox_area(c)
    def pref_key(c: Dict) -> Tuple[float, float]:
        # zoom優先→次にスコア
        zoom = float(c.get("crop_level", 0.0)) if prefer_zoom_in else 0.0
        return (zoom, _score_of(c, score_key))

    order = sorted(range(len(kept)), key=lambda i: pref_key(kept[i]), reverse=True)
    keep_idx: List[int] = []
    for i in order:
        bi = kept[i]["bbox"]
        ok = True
        for j in keep_idx:
            bj = kept[j]["bbox"]
            if _bbox_iou(bi, bj) >= cross_crop_iou:
                ok = False
                break
        if ok:
            keep_idx.append(i)
    return [kept[i] for i in keep_idx]

# ============================================================
# Mask-IoU de-dup (ほぼ同一のみを潰す)
# ============================================================

def dedup_by_mask_iou(
    cands: List[Dict],
    iou_thr: float = 0.97,
    bbox_gate: float = 0.60,
    downsample_side: int = 192,
    score_key: str = "score",
) -> List[Dict]:
    """"ほぼ同一"マスクをまとめて間引く（デデュープ）。
    - まず bbox IoU < bbox_gate のペアは詳細比較をスキップ（高速化）
    - 詳細比較は縮小マスクで IoU を近似
    - スコア降順の貪欲法
    """
    if not cands:
        return []
    for c in cands:
        _ensure_bbox_area(c)

    # 事前に縮小済みマスクを作る（再利用）
    downs: List[np.ndarray] = []
    for c in cands:
        mu8 = _ensure_u8_mask(c["mask"])
        d = cv2.resize(mu8, (downsample_side, downsample_side), interpolation=cv2.INTER_NEAREST)
        downs.append(d)

    order = sorted(range(len(cands)), key=lambda i: _score_of(cands[i], score_key), reverse=True)
    keep_idx: List[int] = []
    for i in order:
        bi = cands[i]["bbox"]
        ok = True
        for j in keep_idx:
            bj = cands[j]["bbox"]
            if _bbox_iou(bi, bj) < bbox_gate:
                continue
            # bboxが近い時のみ詳細IoU
            if _mask_iou_fast(downs[i], downs[j], side=downsample_side) >= iou_thr:
                ok = False
                break
        if ok:
            keep_idx.append(i)
    return [cands[i] for i in keep_idx]

# ============================================================
# Spatially-pruned NMS / De-dup (近傍だけを見る高速版)
# ============================================================

def _cells_covered(b: Tuple[int,int,int,int], cell: int) -> Tuple[range, range]:
    x, y, w, h = b
    if cell <= 0:
        raise ValueError("cell must be > 0")
    i0 = int(np.floor(x / cell))
    j0 = int(np.floor(y / cell))
    i1 = int(np.floor((x + w) / cell))
    j1 = int(np.floor((y + h) / cell))
    return range(i0, i1 + 1), range(j0, j1 + 1)


def _estimate_cell_size(bboxes: List[Tuple[int,int,int,int]]) -> int:
    if not bboxes:
        return 64
    ws = np.array([max(1, b[2]) for b in bboxes])
    hs = np.array([max(1, b[3]) for b in bboxes])
    # 細長対策：短辺の中央値を基準にする
    base = int(np.median(np.minimum(ws, hs)))
    return max(16, min(512, base))


def nms_box_greedy_spatial(
    cands: List[Dict],
    iou_thr: float = 0.7,
    score_key: str = "score",
    cell_size: Optional[int] = None,
) -> List[Dict]:
    """ボックスNMSを**近傍だけ**で判定する高速版。
    - 一様グリッドで空間分割し、採用済みはグリッドに登録
    - 次候補は**自身の bbox が覆うセル**に登録された相手とだけ IoU を計算
    期待計算量は **O(N log N + N * k)**（k: 1セルあたりの平均保持件数）
    """
    if not cands:
        return []
    for c in cands:
        _ensure_bbox_area(c)
    bboxes = [c["bbox"] for c in cands]
    if cell_size is None:
        cell_size = _estimate_cell_size(bboxes)

    order = sorted(range(len(cands)), key=lambda i: _score_of(cands[i], score_key), reverse=True)

    grid: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    keep_idx: List[int] = []

    for i in order:
        bi = bboxes[i]
        ok = True
        ir, jr = _cells_covered(bi, cell_size)
        # 近傍候補集合
        neigh: List[int] = []
        for ii in ir:
            for jj in jr:
                neigh.extend(grid.get((ii, jj), []))
        # 重複除去（近傍に対してのみ）
        for j in neigh:
            if _bbox_iou(bi, bboxes[j]) >= iou_thr:
                ok = False
                break
        if ok:
            keep_idx.append(i)
            # 採用したら登録
            for ii in ir:
                for jj in jr:
                    grid[(ii, jj)].append(i)
    return [cands[i] for i in keep_idx]


def dedup_by_mask_iou_spatial(
    cands: List[Dict],
    iou_thr: float = 0.97,
    bbox_gate: float = 0.60,
    downsample_side: int = 192,
    score_key: str = "score",
    cell_size: Optional[int] = None,
) -> List[Dict]:
    """"ほぼ同一"だけを潰すデデュープの**近傍版**。
    - グリッドで採用済み近傍だけ比較
    - bboxゲートでさらに絞り、通過分のみ縮小マスクIoUを計算
    """
    if not cands:
        return []
    for c in cands:
        _ensure_bbox_area(c)

    bboxes = [c["bbox"] for c in cands]
    if cell_size is None:
        cell_size = _estimate_cell_size(bboxes)

    # 縮小マスクを事前作成
    downs: List[np.ndarray] = []
    for c in cands:
        mu8 = _ensure_u8_mask(c["mask"])
        d = cv2.resize(mu8, (downsample_side, downsample_side), interpolation=cv2.INTER_NEAREST)
        downs.append(d)

    order = sorted(range(len(cands)), key=lambda i: _score_of(cands[i], score_key), reverse=True)
    grid: Dict[Tuple[int,int], List[int]] = defaultdict(list)
    keep_idx: List[int] = []

    for i in order:
        bi = bboxes[i]
        ok = True
        ir, jr = _cells_covered(bi, cell_size)
        neigh: List[int] = []
        for ii in ir:
            for jj in jr:
                neigh.extend(grid.get((ii, jj), []))
        # 近傍だけ詳細チェック
        for j in neigh:
            bj = bboxes[j]
            if _bbox_iou(bi, bj) < bbox_gate:
                continue
            if _mask_iou_fast(downs[i], downs[j], side=downsample_side) >= iou_thr:
                ok = False
                break
        if ok:
            keep_idx.append(i)
            for ii in ir:
                for jj in jr:
                    grid[(ii, jj)].append(i)
    return [cands[i] for i in keep_idx]

# EOF
