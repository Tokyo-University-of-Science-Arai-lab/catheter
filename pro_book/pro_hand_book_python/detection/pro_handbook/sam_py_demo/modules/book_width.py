from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


def estimate_book_width(
    pts: np.ndarray,                 # (N,3) in camera frame [m]
    mean: Any,
    pc1: Any,
    pc2: Any,
    slice_half_thickness_m: float = 0.0015,  # 少し厚めにして安定化
    a_step_m: float = 0.002,                 # 2mm step
    a_start_m: float = 0.0,                  # start at 0 (after shifting)
    a_end_margin_m: float = 0.002,           # 最大端の少し手前まで
    min_points_in_slice: int = 20,           # 少なすぎるスライスを除外
    lower_percentile: float = 2.0,           # 下位1%
    upper_percentile: float = 98.0,          # 上位99%
) -> Dict[str, Any]:
    """
    書籍点群をPCA空間に射影し、pc1方向に薄いスライスを切りながら、
    各スライス内の pc2 方向の広がりから書籍幅を推定する。

    改善点:
      1. 幅は3次元距離ではなく pc2 成分差で計算
      2. min/max ではなく percentile で外れ値に頑健化
      3. 最終値は平均ではなく中央値を使用
    """
    pts = np.asarray(pts, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(3,)
    pc1 = np.asarray(pc1, dtype=np.float32).reshape(3,)
    pc2 = np.asarray(pc2, dtype=np.float32).reshape(3,)

    if pts.ndim != 2 or pts.shape[1] != 3:
        return {
            "ok": False,
            "reason": f"invalid_pts_shape: {pts.shape}",
            "book_widths_m": [],
            "av_book_width_m": None,
        }

    if pts.shape[0] < 10:
        return {
            "ok": False,
            "reason": "too_few_points",
            "book_widths_m": [],
            "av_book_width_m": None,
        }

    X = pts - mean

    # PCA座標（pc1, pc2）に射影
    t1 = X @ pc1  # 長手方向
    t2 = X @ pc2  # 幅方向

    # pc1座標を 0 始まりにシフト
    t1_min = float(np.min(t1))
    s1 = t1 - t1_min

    s1_max = float(np.max(s1))
    a_end = max(0.0, s1_max - float(a_end_margin_m))
    if a_end <= float(a_start_m):
        return {
            "ok": False,
            "reason": "invalid_a_range",
            "pca": {"mean": mean.tolist(), "pc1": pc1.tolist(), "pc2": pc2.tolist()},
            "book_widths_m": [],
            "av_book_width_m": None,
        }

    book_widths: List[float] = []
    slice_debug: List[Dict[str, Any]] = []

    a = float(a_start_m)
    half = float(slice_half_thickness_m)
    step = float(a_step_m)

    while a <= a_end + 1e-12:
        idx = np.where(np.abs(s1 - a) <= half)[0]

        if idx.size >= int(min_points_in_slice):
            local_t2 = t2[idx]

            # 外れ点に強い幅推定
            t2_lo = float(np.percentile(local_t2, lower_percentile))
            t2_hi = float(np.percentile(local_t2, upper_percentile))
            w = float(t2_hi - t2_lo)  # pc2方向の幅そのもの

            # 念のため非負に丸める
            w = max(0.0, w)

            book_widths.append(w)
            slice_debug.append({
                "a_m": float(a),
                "num_points": int(idx.size),
                "t2_min": float(np.min(local_t2)),
                "t2_max": float(np.max(local_t2)),
                "t2_lo": t2_lo,
                "t2_hi": t2_hi,
                "width_m": w,
            })

        a += step

    if len(book_widths) == 0:
        return {
            "ok": False,
            "reason": "no_valid_slices",
            "book_widths_m": [],
            "av_book_width_m": None,
            "slice_debug": [],
        }

    widths_np = np.asarray(book_widths, dtype=np.float32)

    return {
        "ok": True,
        "av_book_width_m": float(np.median(widths_np)),   # 最終値は中央値
        "book_widths_m": book_widths,
        "mean_book_width_m": float(np.mean(widths_np)),   # 比較用に残す
        "slice_debug": slice_debug,
        "params": {
            "slice_half_thickness_m": float(slice_half_thickness_m),
            "a_step_m": float(a_step_m),
            "a_start_m": float(a_start_m),
            "a_end_margin_m": float(a_end_margin_m),
            "min_points_in_slice": int(min_points_in_slice),
            "lower_percentile": float(lower_percentile),
            "upper_percentile": float(upper_percentile),
        },
    }