import numpy as np
from typing import Dict, Any

def run_step5_compute_midpoint_and_side(
    ctx: Dict[str, Any],
    image_width: int,
) -> Dict[str, Any]:
    """
    STEP 5:
      - STEP4 で得たペアについて，両矩形中心の中点を計算
      - その中点が画像の左右どちらの半分に位置するかを判定
    """
    pair = ctx.get("pair_indices")
    centers = ctx.get("centers")  # (N,2)
    if pair is None or centers is None or len(centers) == 0:
        ctx["midpoint"] = None
        ctx["is_right_half"] = None
        return ctx

    i, j = pair
    c_i = centers[i]
    c_j = centers[j]
    mid = 0.5 * (c_i + c_j)
    mid_x = float(mid[0])

    half_x = float(image_width) * 0.5
    is_right_half = mid_x >= half_x

    ctx["midpoint"] = mid
    ctx["is_right_half"] = is_right_half #右半分なら True, 左半分なら False
    return ctx
