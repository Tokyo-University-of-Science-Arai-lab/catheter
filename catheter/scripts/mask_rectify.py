#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM3マスクの矩形化（min_area_rect 方式のみ）。

側面が見えている箱のマスクは、面と面の境目（稜線）で確率が曖昧になり
輪郭が1px単位でギザギザに揺れる。これを最小外接矩形で置き換えて安定させる。

■ 方式Bを採用しなかった理由
  凸包を4点近似する方式（approx_poly_quad）も試したが、85インスタンスの実測で
      方式A min_area_rect : 平均IoU 0.9216（最小 0.8596）
      方式B approx_poly   : 平均IoU 0.8422（最小 0.5137）
  と明確に劣り、85件中66件でAが上回った。
  Bはラベルの角の飛び出しに引っ張られて隣の箱へはみ出す失敗が目立ったため不採用。
  検証コードは catheter/scripts/compare_quad_fit.py に残してある。

■ 方式Aの性質
  - 凸包を完全に内包するため、マスク画素が矩形からはみ出すことは無い。
    IoUが下がるのは「矩形が余分に広い」場合だけ。
  - 隣り合う辺が必ず直交するため、遠近で台形に写る箱では鋭角側に余白が出る。
    今回のデータは棚に正対しており実害は無かった（最小IoU 0.8596）。
  - 凸包の1頂点に引っ張られるので、突出したノイズには弱い。
    rectangularity が低いマスクは矩形化を見送る安全弁を用意している。

使い方:
    from catheter.scripts.mask_rectify import rectify_mask, rectify_masks

    mask01_rect = rectify_mask(mask01)          # 矩形化した0/1マスク
    masks_rect  = rectify_masks(masks)          # まとめて
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

__all__ = [
    "min_area_rect_box",
    "rectangularity",
    "rectify_mask",
    "rectify_masks",
]

# 元マスクの面積 / 最小外接矩形の面積。
# これを下回るマスクは「矩形とかけ離れている」＝ノイズや分割失敗の疑いがあるので
# 矩形化せず元のまま返す（無理に矩形化すると隣の箱まで飲み込むため）。
MIN_RECTANGULARITY = 0.55

# 前景がこれ未満のマスクは対象外
MIN_AREA_PX = 50


def min_area_rect_box(mask01: np.ndarray) -> np.ndarray | None:
    """0/1マスク -> 最小外接矩形の4頂点 (4,2) float32。前景が無ければ None。"""
    m = (np.asarray(mask01) > 0).astype(np.uint8)
    if int(m.sum()) < MIN_AREA_PX:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 0:
        return None

    return cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)


def rectangularity(mask01: np.ndarray, box: np.ndarray | None = None) -> float:
    """元マスク面積 / 最小外接矩形面積。1.0に近いほど矩形らしい。"""
    m = (np.asarray(mask01) > 0).astype(np.uint8)
    area = int(m.sum())
    if area < MIN_AREA_PX:
        return 0.0

    if box is None:
        box = min_area_rect_box(m)
    if box is None:
        return 0.0

    # 4頂点から矩形面積を出す（隣接2辺の長さの積）
    w = float(np.linalg.norm(box[0] - box[1]))
    h = float(np.linalg.norm(box[1] - box[2]))
    rect_area = w * h
    if rect_area < 1e-6:
        return 0.0
    return float(area / rect_area)


def rectify_mask(
    mask01: np.ndarray,
    *,
    min_rectangularity: float = MIN_RECTANGULARITY,
    return_info: bool = False,
):
    """
    マスクを最小外接矩形で塗り直した0/1マスクを返す。

    矩形化を見送る条件（元マスクをそのまま返す）:
      - 前景が小さすぎる / 輪郭が取れない
      - rectangularity が min_rectangularity 未満
        （元の形が矩形とかけ離れている＝矩形化すると余計なものを飲み込む）

    return_info=True なら (mask, info) を返す。
    """
    m = (np.asarray(mask01) > 0).astype(np.uint8)
    info = {
        "applied": False,
        "reason": "",
        "rectangularity": 0.0,
        "box": None,
        "area_before": int(m.sum()),
        "area_after": int(m.sum()),
    }

    box = min_area_rect_box(m)
    if box is None:
        info["reason"] = "輪郭が取れない、または面積が小さすぎる"
        return (m, info) if return_info else m

    r = rectangularity(m, box)
    info["rectangularity"] = round(r, 4)
    info["box"] = box.tolist()

    if r < min_rectangularity:
        info["reason"] = f"rectangularity {r:.3f} < {min_rectangularity} のため矩形化を見送り"
        return (m, info) if return_info else m

    # np.zeros_like だと元配列が非連続のときOpenCVが受け付けないため明示的に確保する
    out = np.zeros(m.shape, dtype=np.uint8)
    cv2.fillPoly(out, [np.round(box).astype(np.int32).reshape(-1, 1, 2)], 1)

    info["applied"] = True
    info["reason"] = "ok"
    info["area_after"] = int(out.sum())
    return (out, info) if return_info else out


def rectify_masks(
    masks: Sequence[np.ndarray],
    *,
    min_rectangularity: float = MIN_RECTANGULARITY,
) -> list[np.ndarray]:
    """複数マスクをまとめて矩形化する。"""
    return [rectify_mask(m, min_rectangularity=min_rectangularity) for m in masks]
