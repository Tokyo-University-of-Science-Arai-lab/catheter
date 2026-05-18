from typing import List, Dict, Any, Tuple
import numpy as np

from .sort_by_bottom_left_x import run_step2_sort_by_bottom_left_x
from .compute_center import run_step3_compute_rects_and_centers
from .select_max_gap_pair import run_step4_select_max_gap_pair
from .midpoint_and_side import run_step5_compute_midpoint_and_side
from .select_base_point import run_step6_select_base_point
from .compute_guide_line import run_step7_compute_guide_line


def run_steps_2_to_7(
    masks: List[np.ndarray],
    image_width: int,
    line_len: float = 500.0,
    max_allowed_gap_px: float = 1000.0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    masks を入力として STEP2〜7 を順番に実行するユーティリティ関数。

    Parameters
    ----------
    masks : List[np.ndarray]
        バイナリマスク (H, W) のリスト。STEP1（y方向zスコア）後のものを渡す想定。
    image_width : int
        元画像の横幅（STEP5の「右半分 or 左半分」判定に使用）。
    line_len : float, optional
        STEP7 で引くガイド線の長さ [px]。デフォルト 607px。

    Returns
    -------
    masks_sorted : List[np.ndarray]
        STEP2 により左下x昇順に並べ替えられたマスクのリスト。
    ctx : Dict[str, Any]
        STEP3〜7 で共通して使うコンテキスト。
        主なキー:
          - "rects"       : 各マスクの最小外接矩形 (tl,tr,br,bl)
          - "centers"     : 各矩形の中心座標 (N,2)
          - "pair_indices": 最大ギャップのペア (i, j)
          - "midpoint"    : そのペアの中心同士の中点
          - "is_right_half": 中点が画像右半分なら True
          - "base_point"  : ガイド線の基準点
          - "line_p0"     : ガイド線の始点 (x,y)
          - "line_p1"     : ガイド線の終点 (x,y)
    """
    # STEP 2: 左下x座標が小さい順にマスクを並べ替える
    masks_sorted = run_step2_sort_by_bottom_left_x(masks)

    # STEP 3: 各マスクの最小外接矩形 & 中心を計算
    ctx = run_step3_compute_rects_and_centers(masks_sorted)

    # STEP 4: 隣り合う矩形のマスク間隔を計算し，最大のペアを選択
    ctx = run_step4_select_max_gap_pair(ctx, max_allowed_gap_px=max_allowed_gap_px)

    # STEP 5: ペア矩形中心同士の中点を計算し，画像の右半分/左半分を判定
    ctx = run_step5_compute_midpoint_and_side(ctx, image_width=image_width)

    # STEP 6: 基準点 (base_point) と，中心軸を使う側の矩形インデックスを決定
    ctx = run_step6_select_base_point(ctx)

    # STEP 7: 基準点から中心軸方向に line_len のガイド線を計算
    ctx = run_step7_compute_guide_line(ctx, line_len=line_len)

    return masks_sorted, ctx
