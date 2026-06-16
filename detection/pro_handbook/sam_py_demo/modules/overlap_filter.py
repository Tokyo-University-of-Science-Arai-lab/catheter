# modules/mask_overlap_cut.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import List, Optional

import cv2  # 必須（minAreaRect を使う）

def _rectangularity_rotated(mask: np.ndarray) -> float:
    """
    回転最小外接矩形による矩形度 ∈ [0,1]
      矩形度 = 面積 / minAreaRectの面積
    入力は 0/1 または bool の2値配列を想定。
    """
    # bool化（0以外はTrue）
    m = mask.astype(bool, copy=False)
    area = int(m.sum())
    ys, xs = np.where(m)
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])  # (N,2) [x,y]
    (_, _), (w, h), _ = cv2.minAreaRect(pts)# 得られるのは (中心), (幅,高さ), 角度．中心座標は捨てる
    rect_area = float(w) * float(h)
    if rect_area <= 0.0:
        return 0.0
    return float(area) / rect_area


def keep_max_rectangularity_per_pixel_rotated(
    masks: List[np.ndarray],
    chunk_rows: Optional[int] = None,
) -> List[np.ndarray]:
    """
    あなたの3手順を厳密に実装（AABBは使わない）:
      1) 変更前マスク群から各マスクの矩形度（回転最小外接矩形）を一度だけ計算
      2) 各画素で「その画素を含むマスクの中で矩形度最大のみ1、他はその画素だけ0」
      3) 2 の結果でマスク群を更新して返す（uint8: 0/1）

    計算量:  O(N*H*W)  （Nはマスク数）
    メモリ:  既定は (N,H,W) のブールスタックを使用。chunk_rows 指定で等価に省メモリ。
    """
    
    if not masks:
        return []

    H, W = masks[0].shape
    for m in masks:
        if m.shape != (H, W):
            raise ValueError("All masks must have the same shape")
    # 形状チェック＆bool化
    bool_masks = [m.astype(bool, copy=False) for m in masks]
    N = len(bool_masks)

    # 変更前マスクで矩形度スコアを固定
    scores = np.asarray([_rectangularity_rotated(m) for m in bool_masks], dtype=np.float32)

    def _emit(out_bool_list: List[np.ndarray]) -> List[np.ndarray]:
        """内部 bool を、元のスタイルで返す変換"""
        return [m.astype(np.uint8) for m in out_bool_list]

    # ---- 一括 or チャンクで勝者決定 ----
    if chunk_rows is None:
        stack = np.stack(bool_masks, axis=0)                       # (N,H,W) bool
        scores3 = np.where(stack, scores[:, None, None], -np.inf)  # (N,H,W)
        winner = np.argmax(scores3, axis=0)                        # (H,W)
        any_on = stack.any(axis=0)                                 # (H,W)
        out_bool = [(stack[i] & any_on & (winner == i)) for i in range(N)]
        return _emit(out_bool)

    # 省メモリ：行方向チャンク
    step = int(chunk_rows)
    if step <= 0:
        raise ValueError("chunk_rows must be positive")
    out_bool = [np.zeros((H, W), dtype=bool) for _ in range(N)]
    for y0 in range(0, H, step):
        y1 = min(H, y0 + step)
        slab = np.stack([m[y0:y1, :] for m in bool_masks], axis=0)         # (N,h,W)
        scores3 = np.where(slab, scores[:, None, None], -np.inf)           # (N,h,W)
        winner = np.argmax(scores3, axis=0)                                 # (h,W)
        any_on = slab.any(axis=0)                                           # (h,W)
        for i in range(N):
            out_bool[i][y0:y1, :] = (slab[i] & any_on & (winner == i))
    return _emit(out_bool)

