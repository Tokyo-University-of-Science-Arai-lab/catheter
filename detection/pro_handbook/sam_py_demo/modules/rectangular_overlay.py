# modules/rectangular_overlay.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
from PIL import Image


def min_area_rect_box(mask: np.ndarray) -> Optional[np.ndarray]:
    """
    バイナリマスク (H,W) から最小外接矩形の4頂点 (4,2) float32 を返す。
    - マスクに前景が無ければ None を返す。
    """
    # 0/1 に正規化
    m = (mask > 0).astype(np.uint8)

    # 外接輪郭を取得
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 一番大きい輪郭から最小外接矩形
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)    # (cx, cy), (w, h), angle
    box = cv2.boxPoints(rect)      # (4, 2) float32

    return box.astype(np.float32)


def save_rectangular_overlay(
    img: Image.Image,
    masks: List[np.ndarray],
    out_dir: Path,
    prefix: str,
) -> None:
    """
    与えられたマスクそれぞれについて min_area_rect_box で最小外接矩形を求め、
    元画像に矩形をオーバーレイした PNG を1枚保存する。

    出力ファイル名: out_dir / f"{prefix}.png"
    """
    if not masks:
        return

    # PIL.Image (RGB) -> OpenCV BGR
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for mask in masks:
        if mask is None:
            continue

        box = min_area_rect_box(mask)
        if box is None:
            continue

        # (4,2) -> OpenCV 用 (4,1,2) int32
        pts = np.round(box).astype(np.int32).reshape(-1, 1, 2)

        # 矩形を描画（色や太さは必要なら調整してOK）
        cv2.polylines(
            bgr,
            [pts],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    out_path = Path(out_dir) / f"{prefix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)
