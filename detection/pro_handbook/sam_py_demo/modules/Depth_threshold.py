import numpy as np
def depth_range_zscore_from_mask(
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    depth_scale: float,
    zscore_lim: float = 3.0,
    min_keep: int = 200,
) -> tuple[float, float]:
    """マスク前景depth[m]から zscore±lim のD_min/D_maxを返す（m）"""
    ys, xs = np.where(mask01 > 0)
    if ys.size < 10:
        return float("nan"), float("nan")

    d = depth_u16[ys, xs].astype(np.float32) * float(depth_scale)
    d = d[d > 0]
    if d.size < 10:
        return float("nan"), float("nan")

    mu = float(d.mean())
    sd = float(d.std())
    if sd < 1e-9:
        return float(d.min()), float(d.max())

    z = (d - mu) / sd
    keep = (z >= -zscore_lim) & (z <= zscore_lim)
    if int(keep.sum()) < min_keep:
        # 落ちすぎるならフォールバック
        return float(d.min()), float(d.max())

    d2 = d[keep]
    return float(d2.min()), float(d2.max())


def filter_points_by_depth_range(pts: np.ndarray, D_min: float, D_max: float) -> np.ndarray:
    """pts[:,2] を D_min..D_max でフィルタ"""
    if pts is None or pts.shape[0] == 0:
        return pts
    if not (np.isfinite(D_min) and np.isfinite(D_max)):
        return pts
    z = pts[:, 2]
    keep = (z >= D_min) & (z <= D_max)
    return pts[keep]