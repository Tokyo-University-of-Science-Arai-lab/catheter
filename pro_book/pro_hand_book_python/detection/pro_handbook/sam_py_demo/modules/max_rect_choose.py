import cv2
import numpy as np  

def mask_rectangularity(mask01: np.ndarray) -> float:
    """rectangularity = area(mask) / area(minAreaRect)"""
    m = (mask01 > 0).astype(np.uint8)
    area = int(m.sum())
    if area < 50:
        return 0.0

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (w, h) = rect[1]
    rect_area = float(w * h)
    if rect_area < 1e-6:
        return 0.0
    return float(area / rect_area)


def select_rectmax_mask(masks) -> tuple[int, np.ndarray, float]:
    """(1-based idx, mask01, rectangularity)"""
    best_i = -1
    best_r = -1.0
    best_mask01 = None
    for i, m in enumerate(masks):
        mask01 = (np.asarray(m) > 0).astype(np.uint8)
        r = mask_rectangularity(mask01)
        if r > best_r:
            best_r = r
            best_i = i
            best_mask01 = mask01
    return best_i + 1, best_mask01, best_r


