#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any


# ================== 基本ジオメトリユーティリティ ==================

def min_area_rect_box(mask: np.ndarray) -> np.ndarray:
    """
    1つのバイナリマスクから最小外接矩形を求め，4頂点(4,2)を返す (float32, (x,y)).
    """
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("mask has no foreground pixels")
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)         # (cx,cy), (w,h), angle
    box = cv2.boxPoints(rect)           # (4,2)
    return box.astype(np.float32)


def short_side_midline(box: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    最小外接矩形の「短辺の中点同士を結ぶ直線」の 2 点 (p1, p2) を返す。
    box: (4,2) float32, (x,y)
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

    # 短辺 (長さが最小の2辺) を取得
    short_edges = [e for e in edges if e[3] <= min_len + eps]
    if len(short_edges) != 2:
        short_edges = sorted(edges, key=lambda e: e[3])[:2]

    mids = [(e[1] + e[2]) / 2.0 for e in short_edges]
    return mids[0], mids[1]


def line_direction(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-6:
        raise ValueError("degenerate line")
    return v / n


def angle_between_dirs(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    2方向ベクトルのなす角(度)を返す。0〜90deg想定なので内積の絶対値を使用。
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_t = float(np.dot(v1, v2) / (n1 * n2))
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_t))))


def long_edges(box: np.ndarray):
    """
    最小外接矩形の「長辺」2本を返す。
    return: List[(p0, p1, length2)]  長い順 2 本
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


def edge_x_for_left_book(box: np.ndarray) -> float:
    """
    左側の書籍に対して，「右側の長辺」の x 位置(中点)を返す。
    """
    es = long_edges(box)
    xs = [(e[0][0] + e[1][0]) / 2.0 for e in es]
    return float(xs[int(np.argmax(xs))])


def edge_x_for_right_book(box: np.ndarray) -> float:
    """
    右側の書籍に対して，「左側の長辺」の x 位置(中点)を返す。
    """
    es = long_edges(box)
    xs = [(e[0][0] + e[1][0]) / 2.0 for e in es]
    return float(xs[int(np.argmin(xs))])


# ================== オレンジマスク関連 ==================

def find_orange_mask(
    book_masks: List[np.ndarray],
    roi_ratio: float = 0.5,
) -> np.ndarray:
    """
    書籍マスクの union から補集合を作り、
    画像下側 ROI のみを残したものを「オレンジマスク」として返す。

    roi_ratio : 縦方向の何割から下を残すか（0.5 なら下半分）
    """
    if not book_masks:
        raise ValueError("book_masks is empty")

    # すべて同じサイズ前提
    H, W = book_masks[0].shape

    union = np.zeros((H, W), np.uint8)
    for m in book_masks:
        m = np.asarray(m)
        if m.shape != (H, W):
            raise ValueError("all book_masks must have same shape")
        union |= (m > 0).astype(np.uint8)

    complement = (union == 0).astype(np.uint8)

    cut_y = int(H * roi_ratio)
    cut_y = max(0, min(H, cut_y))

    orange = np.zeros_like(complement, dtype=np.uint8)
    orange[cut_y:, :] = complement[cut_y:, :]

    return orange


def debug_save_complement_and_orange(
    book_masks: List[np.ndarray],
    orange_mask: np.ndarray,
    out_jpg_path: str,
    A: Tuple[float, float] | None = None,
    B: Tuple[float, float] | None = None,
):
    """
    補集合と、その中で選ばれたオレンジマスクだけを可視化して保存する。
    A,B が渡されれば、その位置もマーカーで描画する。
    """
    H, W = book_masks[0].shape

    union = np.zeros((H, W), np.uint8)
    for m in book_masks:
        union |= (m > 0).astype(np.uint8)
    complement = (union == 0).astype(np.uint8)

    vis = np.zeros((H, W, 3), np.uint8)
    # 補集合を暗めグレー
    vis[complement > 0] = (0, 165, 255)
    # オレンジマスクをオレンジで上書き
    vis[orange_mask > 0] = (0, 165, 255)  # BGR: orange

    # --- A, B を描画（あれば） ---
    if A is not None:
        ax, ay = int(round(A[0])), int(round(A[1]))
        cv2.circle(vis, (ax, ay), 6, (0, 0, 255), thickness=-1)     # 赤丸
        cv2.putText(vis, "A", (ax + 5, ay - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if B is not None:
        bx, by = int(round(B[0])), int(round(B[1]))
        cv2.circle(vis, (bx, by), 6, (255, 0, 0), thickness=-1)     # 青丸
        cv2.putText(vis, "B", (bx + 5, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    p = Path(out_jpg_path)
    debug_path = p.with_name(p.stem + "_complement_orange.png")
    cv2.imwrite(str(debug_path), vis)
    print("[DEBUG] complement + orange mask saved to:", debug_path)



# ================== 線とオレンジマスクの交点 ==================

def intersect_vertical_with_mask(x: float, mask: np.ndarray) -> Tuple[float, float]:
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


def overlay_lines_and_save(
    rgb: np.ndarray,
    L0: Tuple[int, int], L1: Tuple[int, int],
    M0: Tuple[int, int], M1: Tuple[int, int],
    out_path: str,
):
    """
    2本の線分を黄色でオーバーレイして JPEG 保存。
    """
    img = rgb.copy()
    color_yellow = (0, 255, 255)  # BGR
    cv2.line(img, L0, L1, color_yellow, thickness=2)
    cv2.line(img, M0, M1, color_yellow, thickness=2)
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


# ================== メイン処理 ==================

def find_A_B_and_save(
    book_masks: List[np.ndarray],
    rgb: np.ndarray,
    out_jpg_path: str,
) -> Dict[str, Any]:
    """
    1) 書籍マスクを x 方向に並べる
    2) 隣り合う軸(短辺の中点を結ぶ線)のなす角が最大の2冊(L,M)を探す
    3) L側=右辺, M側=左辺 を通る縦線を画像全体に引く
    4) 書籍マスクの補集合 + 下側ROIからオレンジマスクを作る
    5) 2本の縦線とオレンジマスクの最上の交点 A,B を求める
    6) 線をオーバーレイして保存し，情報を dict で返す
    """
    N = len(book_masks)
    if N < 2:
        raise ValueError("need at least 2 book masks")

    H, W = rgb.shape[:2]

    # --- 1. 各書籍マスクの最小外接矩形を求める ---
    boxes = [min_area_rect_box(m) for m in book_masks]
    min_xs = [float(b[:, 0].min()) for b in boxes]
    order = np.argsort(min_xs)
    boxes_sorted = [boxes[i] for i in order]

    # --- 2. 短辺の中点を結ぶ軸を作る ---
    axes_dirs = []
    for box in boxes_sorted:
        p1, p2 = short_side_midline(box)
        v = line_direction(p1, p2)
        axes_dirs.append(v)

    # 隣同士で角度最大のペアを探す
    max_angle = -1.0
    best_i = 0
    for i in range(N - 1):
        ang = angle_between_dirs(axes_dirs[i], axes_dirs[i + 1])
        if ang > max_angle:
            max_angle = ang
            best_i = i

    idx_L_sorted = best_i
    idx_M_sorted = best_i + 1
    mask_index_L = int(order[idx_L_sorted])  # 元のインデックス
    mask_index_M = int(order[idx_M_sorted])

    box_L = boxes_sorted[idx_L_sorted]
    box_M = boxes_sorted[idx_M_sorted]

    # --- 3. L側=右辺, M側=左辺 を通る縦線 ---
    xL = edge_x_for_left_book(box_L)
    xM = edge_x_for_right_book(box_M)

    xL_i = int(round(xL))
    xM_i = int(round(xM))

    L0 = (xL_i, 0)
    L1 = (xL_i, H - 1)
    M0 = (xM_i, 0)
    M1 = (xM_i, H - 1)

    # --- 4. オレンジマスクを作成 ---
    orange_mask = find_orange_mask(book_masks, roi_ratio=0.5)

    # --- 5. 交点 A,B を計算 ---
    Ax, Ay = intersect_vertical_with_mask(xL, orange_mask)
    Bx, By = intersect_vertical_with_mask(xM, orange_mask)

    debug_save_complement_and_orange(
        book_masks,
        orange_mask,
        out_jpg_path,
        A=(Ax, Ay),
        B=(Bx, By),
    )
    # --- 6. オーバーレイして保存 ---
    overlay_lines_and_save(rgb, L0, L1, M0, M1, out_jpg_path)

    return {
        "mask_index_L": mask_index_L,
        "mask_index_M": mask_index_M,
        "angle_deg": float(max_angle),
        "A": (Ax, Ay),
        "B": (Bx, By),
        "line_L": (L0, L1),
        "line_M": (M0, M1),
    }
