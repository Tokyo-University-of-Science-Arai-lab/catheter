# modules/mask_remove_islands.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from typing import List

__all__ = ["remove_islands_from_masks"]

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """
    マスクから最大の連結成分だけ残す (他は削除)。
    mask: 0/1 または 0/255 の2値画像
    return: 最大成分のみ残したマスク (dtype=bool)
    """
    m = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    if num_labels <= 1:
        return (m > 0)

    # stats[:,4] が面積（ピクセル数）
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label)

def remove_islands_from_masks(masks: List[np.ndarray]) -> List[np.ndarray]:
    """
    各マスクから飛び地を削除し、最大成分だけ残す。
    """
    cleaned = []
    for m in masks:
        cleaned.append(_largest_component(m))
    return cleaned
