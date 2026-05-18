import numpy as np
import cv2
from typing import List, Tuple

def _min_area_rect_points(mask: np.ndarray):

    m = np.array(mask)

    if m.ndim < 2:
        return None

    if m.dtype == bool:
        m = m.astype(np.uint8)

    if m.ndim == 3:
        m = m[:, :, 0]

    if m.ndim != 2:
        return None

    if m.size == 0:
        return None

    m = (m > 0).astype(np.uint8)

    if np.sum(m) == 0:
        return None

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    return box.astype(np.float32)

def _order_box_points(pts: np.ndarray) -> np.ndarray:
    """
    minAreaRectの4頂点を (top-left, top-right, bottom-right, bottom-left)
    の順に並べ替える。
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)  # y - x
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def run_step2_sort_by_bottom_left_x(
    masks: List[np.ndarray],
) -> List[np.ndarray]:
    """
    STEP 2:
      - 各マスクの最小外接矩形の左下点の x 座標を計算
      - x が小さい順にマスク配列を並べ替える
    """
    if len(masks) <= 1:
        return masks

    sortable: List[Tuple[float, np.ndarray]] = []
    fallback: List[np.ndarray] = []

    for m in masks:
        try:
            box = _order_box_points(_min_area_rect_points(m))
            bl_x = float(box[3, 0])  # bottom-left x
            sortable.append((bl_x, m))
        except Exception:
            fallback.append(m)

    sortable.sort(key=lambda t: t[0])
    sorted_masks = [m for _, m in sortable]
    # 失敗したマスクはとりあえず末尾に追加
    sorted_masks.extend(fallback)
    return sorted_masks
 