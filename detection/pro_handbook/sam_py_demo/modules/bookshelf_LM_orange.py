# modules/bookshelf_LM_orange.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any


# ===== geometry helpers for L/M and orange mask (bookshelf bottom) ==========

def _min_area_rect_box(mask: np.ndarray) -> np.ndarray:
    """
    単一マスクから最小外接矩形を求めて4頂点(4,2)を返す (x,y, float32)。
    """
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("mask has no foreground pixels")
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def _short_side_midline(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    最小外接矩形の短辺の中点を結ぶ直線の2点を返す。
    """
    pts = box.reshape(4, 2)
    edges = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        length2 = float(np.sum((p1 - p0) ** 2))
        edges.append((i, p0, p1, length2))

    lengths = [e[3] for e in edges]
    min_len = min(lengths)
    eps = 1e-3 * max(min_len, 1.0)
    short_edges = [e for e in edges if e[3] <= min_len + eps]
    if len(short_edges) != 2:
        short_edges = sorted(edges, key=lambda e: e[3])[:2]

    mids = [(e[1] + e[2]) / 2.0 for e in short_edges]
    return mids[0], mids[1]


def _line_dir(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("degenerate line")
    return v / n


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    2方向ベクトルのなす角(度)。0〜90deg程度を想定して絶対値内積。
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_t = float(np.dot(v1, v2) / (n1 * n2))
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_t))))


def _long_edges(box: np.ndarray):
    """
    最小外接矩形の長辺2本を (p0,p1,length2) で返す。
    """
    pts = box.reshape(4, 2)
    edges = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        length2 = float(np.sum((p1 - p0) ** 2))
        edges.append((p0, p1, length2))
    lengths = [e[2] for e in edges]
    max_len = max(lengths)
    eps = 1e-3 * max(max_len, 1.0)
    long_es = [e for e in edges if e[2] >= max_len - eps]
    if len(long_es) != 2:
        long_es = sorted(edges, key=lambda e: e[2], reverse=True)[:2]
    return long_es


def _edge_x_for_left_book(box: np.ndarray) -> float:
    """
    左側の書籍に対して，「右側の長辺」の x 位置(中点)を返す。
    """
    es = _long_edges(box)
    xs = [(e[0][0] + e[1][0]) / 2.0 for e in es]
    return float(xs[int(np.argmax(xs))])


def _edge_x_for_right_book(box: np.ndarray) -> float:
    """
    右側の書籍に対して，「左側の長辺」の x 位置(中点)を返す。
    """
    es = _long_edges(box)
    xs = [(e[0][0] + e[1][0]) / 2.0 for e in es]
    return float(xs[int(np.argmin(xs))])


def _find_orange_mask(book_masks: List[np.ndarray]) -> np.ndarray:
    """
    書籍マスクの補集合から、下側領域のうち
    一番上に来るコンポーネントを「オレンジマスク」として返す。
    """
    H, W = book_masks[0].shape
    union = np.zeros((H, W), np.uint8)
    for m in book_masks:
        union |= (m > 0).astype(np.uint8)

    complement = (union == 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(complement)

    ys_book, xs_book = np.where(union > 0)
    if len(ys_book) == 0:
        raise ValueError("no book pixels found")
    y_books_max = int(ys_book.max())

    best_label = None
    best_min_y = None
    for lab in range(1, num_labels):
        ys, xs = np.where(labels == lab)
        if len(ys) == 0:
            continue
        cy = float(ys.mean())
        # 書籍より下にあるコンポーネントだけ見る
        if cy <= y_books_max:
            continue
        min_y = int(ys.min())  # 「境界の y が最も小さい」もの
        if best_min_y is None or min_y < best_min_y:
            best_min_y = min_y
            best_label = lab

    if best_label is None:
        raise RuntimeError("no bottom component for orange mask")

    orange = (labels == best_label).astype(np.uint8)
    return orange


def _intersect_vertical_with_mask(x: float, mask: np.ndarray) -> Tuple[float, float]:
    """
    縦線 x=const とマスクの交点のうち，一番上 (yが最小) の画素座標を返す。
    """
    H, W = mask.shape
    xi = int(round(x))
    xi = max(0, min(W - 1, xi))
    ys = np.where(mask[:, xi] > 0)[0]
    if len(ys) == 0:
        raise RuntimeError("no intersection between vertical line and mask")
    y = float(ys.min())
    return float(xi), y


def compute_LM_AB_and_overlay(
    book_masks: List[np.ndarray],
    img_bgr: np.ndarray,
    out_path: str,
) -> Dict[str, Any]:
    """
    図の L, M, A, B を求めて、2 本の縦線を黄色で描いた jpg を保存する。

      1. 書籍マスクの最小外接矩形の端点で x が小さい順に並べる
      2. 各矩形の短辺の中点を結ぶ直線を定義
      3. 隣り合う軸同士のなす角が最大の2冊(L,M)を探す
      4. L側は右辺，M側は左辺を通る縦線を黄色でオーバーレイ
      5. その2本の縦線とオレンジマスクの交点 A,B を求める
    """
    if len(book_masks) < 2:
        raise ValueError("need at least 2 book masks for L/M computation")

    # 1) 最小外接矩形 + x順ソート
    boxes = [_min_area_rect_box(m) for m in book_masks]
    min_xs = [float(b[:, 0].min()) for b in boxes]
    order = np.argsort(min_xs)
    boxes_sorted = [boxes[i] for i in order]

    # 2) 短辺の中点を結ぶ軸
    axes_dirs = []
    for box in boxes_sorted:
        p1, p2 = _short_side_midline(box)
        v = _line_dir(p1, p2)
        axes_dirs.append(v)

    # 3) 隣同士の軸のなす角が最大のペア
    max_angle = -1.0
    best_i = 0
    for i in range(len(axes_dirs) - 1):
        ang = _angle_between(axes_dirs[i], axes_dirs[i + 1])
        if ang > max_angle:
            max_angle = ang
            best_i = i

    idx_L_sorted = best_i
    idx_M_sorted = best_i + 1
    mask_index_L = int(order[idx_L_sorted])  # 元の book_masks インデックス
    mask_index_M = int(order[idx_M_sorted])

    box_L = boxes_sorted[idx_L_sorted]
    box_M = boxes_sorted[idx_M_sorted]

    # 4) L側=右辺, M側=左辺 を通る「縦線」(画像全体に延長)
    H, W = img_bgr.shape[:2]
    xL = _edge_x_for_left_book(box_L)
    xM = _edge_x_for_right_book(box_M)
    xL_i = int(round(xL))
    xM_i = int(round(xM))

    L0 = (xL_i, 0)
    L1 = (xL_i, H - 1)
    M0 = (xM_i, 0)
    M1 = (xM_i, H - 1)

    # オレンジマスク検出
    orange = _find_orange_mask(book_masks)

    # 5) 交点 A,B
    Ax, Ay = _intersect_vertical_with_mask(xL, orange)
    Bx, By = _intersect_vertical_with_mask(xM, orange)

    # オーバーレイ保存
    vis = img_bgr.copy()
    yellow = (0, 255, 255)  # BGR
    cv2.line(vis, L0, L1, yellow, thickness=2)
    cv2.line(vis, M0, M1, yellow, thickness=2)
    cv2.imwrite(out_path, vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    return {
        "mask_index_L": mask_index_L,
        "mask_index_M": mask_index_M,
        "angle_deg": float(max_angle),
        "A": (Ax, Ay),
        "B": (Bx, By),
        "line_L": (L0, L1),
        "line_M": (M0, M1),
    }
