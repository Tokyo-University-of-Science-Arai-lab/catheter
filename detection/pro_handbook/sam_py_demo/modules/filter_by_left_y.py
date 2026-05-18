import numpy as np
import cv2
from typing import List

def _min_area_rect_points(mask: np.ndarray) -> np.ndarray:
    """
    1つのバイナリマスクから最小外接矩形を求め，4頂点(4,2)を返す (float32, (x,y)).
    """
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("mask has no foreground pixels")
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)         # (cx,cy), (w,h), angle
    box = cv2.boxPoints(rect)           # (4,2)
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

def run_step1_filter_by_bottom_left_y(
    masks: List[np.ndarray],
    z_thr: float = 1.8,
) -> List[np.ndarray]:
    """
    STEP 1:
      - 各マスクの最小外接矩形を計算
      - 左下点の y 座標の z-score を求め，
        z >= z_thr を満たす“下側の外れ値”マスクを削除する
    """
    if len(masks) <= 2:
        return masks

    bl_y_list = []
    for m in masks:
        try:
            box = _order_box_points(_min_area_rect_points(m))
            bl_y_list.append(float(box[3, 1]))  # bottom-left y
        except Exception:
            # 形が崩れていそうなマスクは z-score 判定から除外（あとで全残し扱い）
            bl_y_list.append(None)

    ys = np.array([y for y in bl_y_list if y is not None], dtype=np.float32)
    if ys.size == 0:
        return masks

    mu = float(ys.mean())
    sigma = float(ys.std())
    if sigma < 1e-6:
        # ほぼ同じ位置なら何も削らない
        return masks

    kept_masks: List[np.ndarray] = []
    for m, y in zip(masks, bl_y_list):
        if y is None:
            kept_masks.append(m)
        else:
            z = (y - mu) / sigma
            if z < z_thr:
                kept_masks.append(m)
    return kept_masks
