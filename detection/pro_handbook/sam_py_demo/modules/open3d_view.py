import numpy as np
from typing import Optional

def visualize_points_and_target_open3d(
    pts: np.ndarray,                 # (N,3) [m]
    target_m: Optional[np.ndarray],  # (3,) [m]
    point_size: float = 2.0,
    target_sphere_radius_m: float = 0.006,  # 見やすいよう6mm球
) -> None:
    """
    3) open3d で 点群(赤) と 目標点(黒) を表示する
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError("open3d is not installed. pip install open3d") from e

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts shape must be (N,3), got {pts.shape}")

    # 点群（赤）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    colors = np.zeros_like(pts, dtype=np.float64)
    colors[:, 0] = 1.0  # R
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geoms = [pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]

    # 目標点（黒球）
    if target_m is not None:
        t = np.asarray(target_m, dtype=np.float64).reshape(3)
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=float(target_sphere_radius_m))
        sph.translate(t)
        sph.paint_uniform_color([0.0, 0.0, 0.0])
        geoms.append(sph)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Points (red) & Target (black)")
    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)

    vis.run()
    vis.destroy_window()