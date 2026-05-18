# modules/select_max_gap_pair.py
import numpy as np
from typing import Dict, Any, List, Tuple


def _min_horizontal_gap(mask_left: np.ndarray, mask_right: np.ndarray) -> float:
    """
    2つのバイナリマスク(同じH,W)について，
    各行ごとに「左マスクの最右画素」と「右マスクの最左画素」の間の
    0 の個数（= ピクセル距離）を計算し，その最小値を返す。

    例: 0000111111000001111111000000
          ^----^    ^----^
          左本        右本
        → 間の 0 が 5 なので gap = 5

    どの行でも両方のマスクが出現しない場合は +inf を返す。
    """
    L = mask_left.astype(bool)
    R = mask_right.astype(bool)
    H, W = L.shape

    min_gap = np.inf

    for y in range(H):
        xsL = np.where(L[y])[0]
        xsR = np.where(R[y])[0]
        if xsL.size == 0 or xsR.size == 0:
            continue

        xL_max = xsL.max()
        xR_min = xsR.min()

        # 左の右端より右の位置に右マスクがなければ、その行では評価しない
        if xR_min <= xL_max:
            # 重なっている or 右マスクが左側に食い込んでる行は、ここでは距離 0 とみなすか無視か悩ましいが
            # 「最短距離」を考えるなら 0 とみなしてOK
            gap = 0
        else:
            # 間に挟まっている 0 の数 = xR_min - xL_max - 1
            gap = xR_min - xL_max - 1

        if gap < min_gap:
            min_gap = gap

    return float(min_gap)

def run_step4_select_max_gap_pair(
    ctx: Dict[str, Any],
    max_allowed_gap_px: float = 20.0,
) -> Dict[str, Any]:
    """
    STEP 4（改訂版）:
      - 左下xでソート済みの rects / masks を前提とする
      - 隣り合うペア(i, i+1)ごとに
          1) 左の右下x > 右の左下x ならペア候補から除外
          2) 行ごとの横方向最短距離(min_horizontal_gap) が max_allowed_gap_px 以上なら除外
      - 上の条件を満たすペアの中から，
        「右下(左) - 左下(右) のユークリッド距離」が大きい順に見て
        最初に条件を満たしたペアを採用する。
    """
    rects: List[np.ndarray] = ctx.get("rects", []) #最小外接矩形のリスト
    masks: List[np.ndarray] = ctx.get("masks", []) #バイナリマスクのリスト

    n = len(rects)
    if n < 2 or len(masks) < 2:                     # ペアが作れない
        ctx["pair_indices"] = None
        ctx["max_gap_distance"] = 0.0
        ctx["min_horizontal_gap"] = np.inf
        return ctx

    # --- まずは「右下–左下距離」を使って候補ペアを列挙する ---
    candidates: List[Tuple[float, int, int]] = []  # (distance, i, j)

    for i in range(n - 1):
        left_rect = rects[i]                       # 左の矩形
        right_rect = rects[i + 1]                  # 右の矩形         

        rb_left = left_rect[2]   # bottom-right of left
        lb_right = right_rect[3] # bottom-left of right

        # 条件①: 左の右下x > 右の左下x ならペアにしない．そっか0がx座標，1がy座標か
        if rb_left[0] > lb_right[0]:
            continue

        #d = float(np.linalg.norm(rb_left - lb_right)) #ユークリッド距離はあかん
        d = float(lb_right[0] - rb_left[0])
        candidates.append((d, i, i + 1))            #リストを追加していくのか

    # 距離が大きい順にソート（1番目が「一番離れているペア」）
    candidates.sort(key=lambda t: t[0], reverse=True)#並び替えするのか  reverse=True で降順

    best_pair = None
    best_d = 0.0
    best_min_gap = np.inf

    for d, i, j in candidates:
        mL = masks[i]
        mR = masks[j]

        # 条件②: 行ごとの「横方向最短距離」を計算
        g = _min_horizontal_gap(mL, mR)
        print(f"Pair ({i},{j}): distance={d:.1f}, min_horizontal_gap={g:.1f}")
        # g が inf の場合 = どの行でも2つが同時に現れない → 物理的にかなり離れている
        if not np.isfinite(g):
            print("  -> min_horizontal_gap is infinite, skip")
            continue

        # しきい値以上離れていたら「遠すぎる」とみなしてスキップ
        if g >= max_allowed_gap_px:
            print(f"  -> min_horizontal_gap {g:.1f} >= {max_allowed_gap_px}, skip")
            continue

        # このペアを採用してループ終了（＝1番目がダメなら2番目…という仕様）
        best_pair = (i, j)
        best_d = d
        best_min_gap = g
        break

    ctx["pair_indices"] = best_pair
    ctx["max_gap_distance"] = float(best_d)
    ctx["min_horizontal_gap"] = float(best_min_gap)

    return ctx
