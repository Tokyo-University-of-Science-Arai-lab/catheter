import numpy as np
from typing import Dict, Any, List

def _rect_tilt_deg(box: np.ndarray) -> float:
    """
    最小外接矩形の「長辺の傾き」を 0〜180[deg] で返す。
    box は (tl, tr, br, bl) の 4点 (4,2) 配列を想定。
    """
    b = np.asarray(box, dtype=np.float32).reshape(4, 2)
    edges = [
        b[1] - b[0],  # top
        b[2] - b[1],  # right
        b[3] - b[2],  # bottom
        b[0] - b[3],  # left
    ]
    lens = [np.linalg.norm(e) for e in edges]
    v = edges[int(np.argmax(lens))]          # 最長の辺 = 長辺
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return 90.0                          # ほぼ点 → 垂直扱い
    v = v / n
    ang = float(np.degrees(np.arctan2(v[1], v[0])))  # [-180,180)
    if ang < 0.0:
        ang += 180.0                         # [0,180)
    return ang

def run_step6_select_base_point(
    ctx: Dict[str, Any],
) -> Dict[str, Any]:
    """
    STEP 6:
      - ペアのマスクのうち，
        ・中点が右半分にある場合：画像左の方のマスクの右上点を基準点
        ・中点が左半分にある場合：画像右の方のマスクの左上点を基準点
      を選択し，あわせて「中心軸（長辺方向）を使う側」の矩形インデックスも保存する
    """
    pair = ctx.get("pair_indices")
    centers = ctx.get("centers")
    rects: List[np.ndarray] = ctx.get("rects", [])
    is_right_half = ctx.get("is_right_half")

    if pair is None or centers is None or len(rects) == 0 or is_right_half is None:
        ctx["base_point"] = None
        ctx["axis_index"] = None
        ctx["left_index"] = None
        ctx["right_index"] = None
        return ctx

    i, j = pair
    c_i = centers[i]
    c_j = centers[j]

    # 左右のインデックス
    if c_i[0] <= c_j[0]:
        left_idx, right_idx = i, j
    else:
        left_idx, right_idx = j, i

    if is_right_half:
        # 中点が右半分：左側マスクの右上点を基準点，右側マスクの中心軸を用いる
        base_rect = rects[left_idx]
        base_pt = base_rect[1]  # top-right
        axis_index = right_idx
    else:
        # 中点が左半分：右側マスクの左上点を基準点，左側マスクの中心軸を用いる
        base_rect = rects[right_idx]
        base_pt = base_rect[0]  # top-left
        axis_index = left_idx

    # ---- ここから追加：どちらの本が「より傾いているか」を判定 ----
    rect_left  = rects[left_idx]
    rect_right = rects[right_idx]

    theta_left  = _rect_tilt_deg(rect_left)   # 0〜180deg
    theta_right = _rect_tilt_deg(rect_right)  # 0〜180deg

    # 「垂直 (90deg) からどれだけ離れているか」を傾き量とみなす
    tilt_left  = abs(theta_left  - 90.0)
    tilt_right = abs(theta_right - 90.0)

    ctx["left_is_tilted"]  = tilt_left  > tilt_right
    ctx["right_is_tilted"] = tilt_right > tilt_left
    # （完全に同じときは両方 False → run_step7 側のフォールバックが効く）

    ctx["base_point"] = base_pt
    ctx["axis_index"] = axis_index
    ctx["left_index"] = left_idx
    ctx["right_index"] = right_idx
    return ctx
