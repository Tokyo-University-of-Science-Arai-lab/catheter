# modules/overlay_viz.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Iterable
import numpy as np
import cv2
from PIL import Image

def _ensure_u8_mask(m: np.ndarray) -> np.ndarray:
    """bool/uint8 どちらでも受け取り、uint8(binary) に正規化"""
    if m.dtype == bool:
        return (m.astype(np.uint8) * 255)
    if m.dtype != np.uint8:
        # 0/255 に寄せる（0/1 でも可）
        return (m > 0).astype(np.uint8) * 255
    return m

def major_axis_endpoints(m: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]] | Tuple[None, None]:
    """
    マスクの主軸（長辺方向）の両端点を返す。
    - minAreaRect で回転外接矩形を取得
    - 長辺の向きから単位ベクトル u を作成
    - 中心 ± (L/2) * u を端点にする
    戻り値: ((x0,y0),(x1,y1)) もしくは (None, None)（検出失敗）
    """
    u8 = _ensure_u8_mask(m)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    cnt = max(cnts, key=cv2.contourArea)

    (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)  # ang ∈ [-90,0)
    if w >= h:
        theta = np.deg2rad(ang); L = float(w)
    else:
        theta = np.deg2rad(ang + 90.0); L = float(h)

    if L < 1e-6:
        return None, None

    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    p0 = (int(round(cx - 0.5 * L * u[0])), int(round(cy - 0.5 * L * u[1])))
    p1 = (int(round(cx + 0.5 * L * u[0])), int(round(cy + 0.5 * L * u[1])))
    return p0, p1

def draw_axes_on_overlay(
    overlay_pil: Image.Image,
    masks: Iterable[np.ndarray],
    color=(0, 255, 255),   # BGR なので注意（OpenCV仕様）
    thickness: int = 3
) -> Image.Image:
    """
    画像（PIL, RGB）上に、各マスクの“主軸”をラインで描画して返す。
    - color は BGR（例: シアンっぽい黄= (0,255,255)）
    - thickness はライン太さ
    """
    img = np.array(overlay_pil)              # RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for m in masks:
        p0, p1 = major_axis_endpoints(m)
        if p0 is None:
            continue
        cv2.line(img, p0, p1, color, thickness, lineType=cv2.LINE_AA)
        cv2.circle(img, p0, thickness + 1, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, p1, thickness + 1, color, -1, lineType=cv2.LINE_AA)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)
