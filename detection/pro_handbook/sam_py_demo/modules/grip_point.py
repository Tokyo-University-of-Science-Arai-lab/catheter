from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

def find_target_point(
    pts: np.ndarray,                  # (N,3) [m]
    y_offset_m: float = 0.1,         # 80mm
    y_band_half_m: float = 0.003,      # ±3mm
) -> Dict[str, Any]:
    """
    2)
    - y最大点A, y最小点C を見つけ、h = y_max - y_min を返す
    - y=y_min+80mm 近傍（±3mm）にある点のうち x が最小の点を target として返す

    返り値:
      {
        "ok": bool,
        "reason": str,
        "h_m": float | None,
        "y_min_m": float | None,
        "y_max_m": float | None,
        "target_m": np.ndarray(3,) | None,
        "num_candidates": int
      }
    """
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts shape must be (N,3), got {pts.shape}")
    if pts.shape[0] == 0:
        return {"ok": False, "reason": "no_points", "target_m": None, "h_m": None}

    ys = pts[:, 1]
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    y0 = y_min + float(y_offset_m)
    band = float(y_band_half_m)

    cand_idx = np.where(np.abs(ys - y0) <= band)[0]
    if cand_idx.size == 0:
        return {
            "ok": False,
            "target_m": None,
            "num_candidates": 0,
        }

    # x が最小の点
    xs = pts[cand_idx, 0]
    i = cand_idx[int(np.argmin(xs))]
    target = pts[i].copy()

    return {
        "ok": True,
        "target_m": target,
        "num_candidates": int(cand_idx.size),
    }