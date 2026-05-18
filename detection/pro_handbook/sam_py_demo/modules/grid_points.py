#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Sequence, Union
import numpy as np

__all__ = [
    "linspace_margined",
    "build_grid_xy",
]

def linspace_margined(lo: float, hi: float, n: int, margin: float = 5.0) -> np.ndarray:
    """
    [lo+margin, hi-margin] を n 等分で返す（float32）。
    n <= 1 のときは中央の1点を返す。
    """
    if n <= 1:
        return np.array([ (lo + hi) / 2.0 ], dtype=np.float32)
    return np.linspace(lo + margin, hi - margin, n, dtype=np.float32)

def build_grid_xy(width: int, height: int, side: Union[int, Sequence[int]], margin: float = 5.0):
    """
    グリッド点の x, y 座標配列を返す。
    side が int なら (side, side)、(side_x, side_y) ならそのまま適用。
    返り値: (xs: np.ndarray[float32], ys: np.ndarray[float32])
    """
    # 後方互換: side が単一整数なら正方グリッド
    if isinstance(side, (tuple, list, np.ndarray)) and len(side) == 2:
        side_x, side_y = int(side[0]), int(side[1])
    else:
        side_x = side_y = int(side)

    xs = linspace_margined(0.0, float(width),  side_x, margin)
    ys = linspace_margined(0.0, float(height), side_y, margin)
    return xs, ys
