#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
マスク平滑化モジュール
cv2.approxPolyDP を使ってギザギザを滑らかにする
"""

import numpy as np
import cv2
from typing import List

__all__ = ["smooth_masks_with_dp"]

def smooth_masks_with_dp(masks: List[np.ndarray], epsilon_factor: float = 0.004) -> List[np.ndarray]:
    """
    マスク群を平滑化する

    Parameters
    ----------
    masks : List[np.ndarray]
        0/1 または 0/255 の2値マスクのリスト
    epsilon_factor : float
        輪郭近似の強さ（0.005〜0.02程度で調整）

    Returns
    -------
    List[np.ndarray]
        平滑化されたマスクのリスト
    """
    smoothed = []
    for m in masks:
        # uint8 に正規化
        mask_u8 = (m.astype(np.uint8) * 255) if m.max() <= 1 else m.astype(np.uint8)

        # 輪郭抽出
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        new_mask = np.zeros_like(mask_u8)

        for cnt in contours:
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(new_mask, [approx], -1, 255, -1)

        smoothed.append(new_mask > 0)

    return smoothed
