from __future__ import annotations
import numpy as np
import cv2
import pyrealsense2 as rs


def rs_color_K_dist(profile: rs.pipeline_profile):
    """RealSense color intrinsics -> OpenCV K(3x3), dist(5,)"""
    intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics() # 内部パラメータ取得
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64).reshape(-1)[:5]
    if dist.size < 5:
        dist2 = np.zeros(5, dtype=np.float64)
        dist2[:dist.size] = dist
        dist = dist2 #dist: OpenCV用の歪み係数 (5,)
    return intr, K, dist


def aruco_marker_pose_target2cam(
    bgr: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    marker_len_m: float,
    *,
    dict_name: str = "DICT_4X4_50",
    target_id: int | None = None,
):

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) # グレースケール変換

    aruco = cv2.aruco   # arucoモジュール取得
    aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dict_name)) #どの種類のマーカーか指定

    # OpenCV新旧対応
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
        print("new aruco detect")
    else:
        params = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        return None

    ids_flat = ids.reshape(-1).tolist()
    if target_id is None:
        idx = 0
    else:
        if target_id not in ids_flat:
            return None
        idx = ids_flat.index(target_id)

    rot_vects, trans_vects, _ = aruco.estimatePoseSingleMarkers(corners, marker_len_m, K, dist)   # マーカーの姿勢推定
    rot_vec = rot_vects[idx].reshape(3).astype(np.float64)
    trans_vec = trans_vects[idx].reshape(3).astype(np.float64)  # [m]
    return int(ids_flat[idx]), rot_vec, trans_vec