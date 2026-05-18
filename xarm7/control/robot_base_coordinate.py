import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .robot_helper import Transform
from .xarm7 import XArm7

def _mm_to_m(x_mm: float) -> float:
    return float(x_mm) * 1e-3


def make_T_from_ypr_mm(x_mm, y_mm, z_mm, yaw, pitch, roll, *, ypr_order="ZYX") -> Transform:
    """平行移動(mm) + yaw/pitch/roll(rad) から Transform"""
    t = np.array([_mm_to_m(x_mm), _mm_to_m(y_mm), _mm_to_m(z_mm)], dtype=np.float64)
    R = Rotation.from_euler(ypr_order, [yaw, pitch, roll], degrees=False)
    return Transform(R, t)


def make_T_from_matrix4(T: np.ndarray) -> Transform:
    """4x4同次変換行列 -> Transform"""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be (4,4), got {T.shape}")
    Rm = T[:3, :3]
    t = T[:3, 3]
    R = Rotation.from_matrix(Rm)
    return Transform(R, t)


def load_T_tcp_camera_from_json(json_path: str | Path, *, key: str = "T_tcp_cam") -> Transform:
    """
    handeye_T_tcp_cam_*.json から TCP->Camera の同次変換を読む。
    JSONには key='T_tcp_cam' がある想定。 :contentReference[oaicite:1]{index=1}
    """
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if key not in data:
        raise KeyError(f"'{key}' not found in {p}. available keys={list(data.keys())}")
    return make_T_from_matrix4(np.array(data[key], dtype=np.float64))


class PoseChain:
    def __init__(self, *, handeye_json_path: str | Path):
        self.handeye_json_path = str(handeye_json_path)
        self.T_tcp_camera = load_T_tcp_camera_from_json(self.handeye_json_path, key="T_cam_tcp")


    def get_T_robot_tcp(self, arm: XArm7) -> Transform:
        """
        xArm から現在の TCP 姿勢を取得し、base(robot)->tcp の Transform にする。
        get_tcp_pose() は [x,y,z,roll,pitch,yaw] を返す想定。
        """
        x, y, z, roll, pitch, yaw = arm.get_tcp_pose(is_radian=True)

        t = np.array([_mm_to_m(x), _mm_to_m(y), _mm_to_m(z)], dtype=np.float64)

        # xArmの並びが roll,pitch,yaw なので一旦 xyz として回転を構成
        R = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=False)

        return Transform(R, t)


def _cam_mm_to_robot_mm_with_chain(chain: PoseChain, arm: XArm7, p_cam_mm: np.ndarray) -> np.ndarray:
    """
    p_cam_mm (camera座標, mm) -> p_robot_mm (robot/base座標, mm)
    """
    p_cam_mm = np.asarray(p_cam_mm, dtype=np.float64).reshape(3)
    p_cam_m = p_cam_mm * 1e-3

    T_robot_tcp = chain.get_T_robot_tcp(arm)

    # base->camera = base->tcp * tcp->camera
    T_robot_camera = T_robot_tcp * chain.T_tcp_camera

    p_robot_m = T_robot_camera.apply(p_cam_m)
    return p_robot_m * 1e3


def cam_mm_to_robot_mm(
    arm: XArm7,
    p_cam_mm: np.ndarray,
    *,
    handeye_json_path: str | Path = "/home/book/pro_book/pro_hand_book_python/xarm7/handeye_pairs/handeye_T_tcp_cam_20260216_165845_copy.json",
    dy_adj_mm: float = 0.0,
) -> np.ndarray:
    """
    JSONの hand-eye 結果（T_tcp_cam）を使って、
    camera(mm) -> robot(mm) に変換する関数版。
    """
    chain = PoseChain(handeye_json_path=handeye_json_path)
    p_robot_mm = _cam_mm_to_robot_mm_with_chain(chain, arm, p_cam_mm)

    p_robot_mm = p_robot_mm.copy()
    p_robot_mm[1] += dy_adj_mm
    return p_robot_mm
