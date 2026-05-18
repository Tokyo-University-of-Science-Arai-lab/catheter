# modules/pointcloud_utils.py
import numpy as np
from pathlib import Path

def masked_depth_to_points(depth_u16: np.ndarray,
                           mask01: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float,
                           depth_scale: float,
                           rgb_bgr: np.ndarray | None = None):
    """
    depth_u16 : (H,W) uint16 (Z16 生値, RealSense)
    mask01    : (H,W) 0/1 or bool, True/1 の画素だけ点群化
    fx,fy,cx,cy : intrinsics（aligned_depth_to_color）
    depth_scale : 例 0.001 (m/カウント)
    rgb_bgr : (H,W,3) BGR (OpenCV保存/取得の想定)。PLYにはRGB順で格納する。

    return: (points[N,3], colors[N,3] or None)
            座標はカメラ座標系（+Z前方, +X右, +Y下）
    """
    assert depth_u16.ndim == 2, "depth_u16 must be (H,W)"
    assert mask01.shape == depth_u16.shape, "mask shape mismatch"
    H, W = depth_u16.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    Z = depth_u16.astype(np.float32) * float(depth_scale)  # [m]
    valid = (Z > 0) & (mask01.astype(bool))

    Zv = Z[valid]
    X = (uu[valid] - cx) / fx * Zv
    Y = (vv[valid] - cy) / fy * Zv
    pts = np.stack((X, Y, Zv), axis=1).astype(np.float32)

    cols = None
    if rgb_bgr is not None:
        # BGR -> RGB
        bgr = rgb_bgr.reshape(-1, 3)[valid.reshape(-1)]
        cols = bgr[:, ::-1].astype(np.uint8)

    return pts, cols


def save_ply_ascii(path: str | Path, points: np.ndarray,
                   colors: np.ndarray | None = None):
    """
    単純なASCII PLY保存。可視化ツール（CloudCompare, MeshLab, Open3Dなど）で読めます。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    N = int(points.shape[0])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is not None:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
