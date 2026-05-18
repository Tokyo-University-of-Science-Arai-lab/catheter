from .sort_by_bottom_left_x import _min_area_rect_points, _order_box_points #毎回計算するのやばない？
from typing import List, Dict, Any
import numpy as np
def run_step3_compute_rects_and_centers(
    masks: List[np.ndarray],
) -> Dict[str, Any]:

    rects = []
    centers = []

    for m in masks:
        box_raw = _min_area_rect_points(m)

        if box_raw is None:
            continue   # 🔥 壊れたmaskはスキップ

        box = _order_box_points(box_raw)

        rects.append(box)
        centers.append(box.mean(axis=0))

    if len(centers) > 0:
        centers_arr = np.stack(centers, axis=0)
    else:
        centers_arr = np.zeros((0, 2), dtype=np.float32)

    return {
        "masks": masks,
        "rects": rects,
        "centers": centers_arr,
    }