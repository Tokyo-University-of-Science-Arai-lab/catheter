# modules/shape_metrics.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Iterable
import numpy as np
import cv2

def _ensure_u8_mask(m: np.ndarray) -> np.ndarray:#なんじゃこりゃ

    """bool/0-1/0-255 どれでも受け取り、輪郭抽出に使える uint8 (0/255) に正規化。"""
    if m.dtype == np.bool_:
        return (m.astype(np.uint8) * 255)
    if m.dtype == np.uint8:
        if m.max() <= 1:
            return (m * 255).astype(np.uint8)
        return m
    # その他 → 0/1 判定して 0/255 に
    mm = (m > 0).astype(np.uint8) * 255
    return mm

def mask_right_and_left(
    m: np.ndarray,
    prefer_rect_slice: bool = True,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    マスクの『左端点 (xL, y)』と『右端点 (xR, y)』を返す。
    ここで y は回転外接矩形の中心 y を丸めて画像内にクリップした水平ライン。

    Parameters
    ----------
    m : np.ndarray
        2値マスク（bool / 0-1 / 0-255 いずれも可）
    prefer_rect_slice : bool, default True
        True なら回転外接矩形 (minAreaRect) の4頂点から作る多角形と
        水平線 y との交点を使って (xL,xR) を求める（高速で安定）。
        False なら、最終的に行 y の「マスク実画素」から (xL,xR) を直接求める
        フォールバックを優先する。

    Returns
    -------
    ((xL, y), (xR, y)) : Tuple[Tuple[int,int], Tuple[int,int]]
        左端点・右端点の整数画素座標。
        マスクが空などで求まらない場合は ((0,0),(0,0)) を返す。
    """
    u8 = _ensure_u8_mask(m)
    H, W = u8.shape[:2]

    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return (0, 0), (0, 0)

    # 最大輪郭を使用
    cnt = max(cnts, key=cv2.contourArea)

    # 回転外接矩形と、その中心 y を使用
    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.float32)

    # 水平ライン y を矩形中心に設定
    y = int(np.clip(round(cy), 0, H - 1))

    def _intersections_with_horizontal_y(y_val: int) -> list[float]:
        xs: list[float] = []
        for i in range(4):
            x0, y0 = box[i]
            x1, y1 = box[(i + 1) % 4]
            ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
            if y_val < ymin - 1e-6 or y_val > ymax + 1e-6:
                continue
            dy = y1 - y0
            if abs(dy) < 1e-6:
                # 辺が水平：その辺区間を交点2つとして扱う
                xs.extend([x0, x1])
            else:
                t = (y_val - y0) / dy
                if -1e-6 <= t <= 1 + 1e-6:
                    xs.append(x0 + t * (x1 - x0))
        return xs

    # 1) 矩形と水平線の交点から (xL, xR) を優先的に取る
    xs = _intersections_with_horizontal_y(y)
    if prefer_rect_slice and len(xs) >= 2:
        x_left_f  = float(np.min(xs))
        x_right_f = float(np.max(xs))
        xL = int(np.clip(np.floor(x_left_f),  0, W - 1))
        xR = int(np.clip(np.ceil(x_right_f),   0, W - 1))
        return (xL, y), (xR, y)

    # 2) 行 y の実マスクから (xL, xR) を求める
    xs_row = np.where(u8[y] > 0)[0]
    if xs_row.size >= 1:
        xL = int(xs_row.min())
        xR = int(xs_row.max())
        return (xL, y), (xR, y)

    # 3) それでもダメなら全体の最小/最大 x を使用（y は中心のまま）
    ys_mask, xs_mask = np.where(u8 > 0)
    if xs_mask.size == 0:
        return (0, 0), (0, 0)
    xL = int(xs_mask.min())
    xR = int(xs_mask.max())
    return xL,xR


# modules/pixel_coordinates.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-


__all__ = [
    # 既存:
    # "mask_right_and_left",
    # 新規:
    "lr_on_row", "mask_lr_on_row",
]

def lr_on_row(mask: np.ndarray, y: int) -> Optional[Tuple[int, int]]:
    """
    上から y [px] の水平ラインで mask と交差する最長の連続区間の (xL, xR) を返す。
    交差しなければ None。
    """
    H, W = mask.shape
    if y < 0 or y >= H:
        return None
    row = mask[y]
    if row.dtype != np.bool_:
        row = row > 0

    xs = np.flatnonzero(row)
    if xs.size == 0:
        return None
    if xs.size == 1:
        x0 = int(xs[0])
        return x0, x0

    # 連続区間に分割して最長を採用
    splits = np.where(np.diff(xs) > 1)[0] + 1
    segments = np.split(xs, splits) if splits.size else [xs]
    seg = max(segments, key=lambda s: (s[-1] - s[0] + 1))
    return int(seg[0]), int(seg[-1])


def mask_lr_on_row(
    mask: np.ndarray,
    y_from_top_px: int,
    *,
    search_radius_px: int = 80,
    prefer_up_first: bool = True,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    指定 y（上から y_from_top_px [px]）で左右端 ((xL,y),(xR,y)) を返す。
    その行で交差しなければ ±search_radius_px で近い行を探索し、見つからなければ None。
    """
    H, W = mask.shape
    y0 = int(np.clip(y_from_top_px, 0, H - 1))

    lr = lr_on_row(mask, y0)
    if lr is not None:
        xL, xR = lr
        return (int(xL), int(y0)), (int(xR), int(y0))

    rng = range(1, int(search_radius_px) + 1)
    for d in rng:
        # 上下のどちらを先に探すか選べる
        order = (("up", y0 - d), ("down", y0 + d)) if prefer_up_first else (("down", y0 + d), ("up", y0 - d))
        for _, y in order:
            if 0 <= y < H:
                lr = lr_on_row(mask, y)
                if lr is not None:
                    xL, xR = lr
                    return (int(xL), int(y)), (int(xR), int(y))

    return None

