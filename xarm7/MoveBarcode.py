from xarm.wrapper import XArmAPI
import numpy as np
import math
from pathlib import Path
import time
from xarm7.control.util import convert

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32
import threading


PKG_DIR = Path(__file__).resolve().parent

XARM_HOST = "192.168.1.208" # AC controller
#XARM_HOST = "192.168.1.197" # DC controller

# initial TCP_pose when 1st perception (pos[mm], rpy[deg])
INIT_TCP_POSE_A5 = [
    104.2,
    -156.2,
    558.8,
    -90.0,
    0.0,
    180.0,
] # TCP offset 修正前，使用不可

BARCODE_Q_A5 = [
    -76.5,
    -104.0,
    86.9,
    19.5,
    22.4,
    36.4,
    155.0
]  # deg, バーコード読み取り用
# INIT_Q_A5 = [
#     -22.9,
#     84.3,
#     -163.1,
#     56.3,
#     -154.6,
#     42.5,
#     -177.5
# ]#なんだっけ？？2号館上から3段目,上下機構400mm  
# INIT_Q_A5 = [
#     -217.5,
#     -20.0,
#     -240.6,
#     52.6,
#     -155.2,
#     -45.9,
#     -114.2
# ] # 2号館上から3段目,上下機構400mm  逆側（進行方向左側）書籍
# INIT_Q_A5 = [
#     -27.1,
#     96.0,
#     -189.7,
#     10.6,
#     -190.0,
#     73.7,
#     -150.4
# ]   #deg, 閉架書架 1H17


#INIT_Q_A5 = [
#    -81.1,
#    -80.7,
#    5.7,
#    9.1,
#    -81.7,
#    9.9,
#    177.3
#]   #deg, 閉架書架 1H13

#INIT_Q_A5 = [
#   -72.5,
#    -76.4,
#    2.6,
#    17.0,
#    -76.2,
#    18.7,
#    168.1
#]  # deg, 2号館

#INIT_Q_A5 = [
#    -68.6,
#    -71.7,
#    1.8,
#    26.1,
#    -69.8,
#    23.5,
#    159.8,
#]  # deg, 10度傾いているカメラ

#INIT_Q_A5 = [
#     -86.3,
#     -101.6,
#     65.6,
#     25.4,
#     16.1,
#     35.0,
#     141.1
#]  # deg, 開架B18（4,3)
INIT_Q_A5 = [
    90.0,
    0.0,
    163.89,
    43.55,
    -357.58,
    40.8,
    70.73
] # 直動軸縦置き撮影姿勢

PLACING_FOR_TEST_Q = [
    -81.1,
    56.2,
    13.9,
    72.3,
    -150.4,
    44.6,
    189.4
] # deg
# INIT_Q_A5_OFFSET = [
#     ,,,,,,]  # deg

DISTANCE_BOOKSHELF = np.radians(60.0)
INSERT_CONTAINER = -330

TCP_VEL_1 = 200
TCP_ACC_1 = 200

TCP_VEL_2 = 50
TCP_ACC_2 = 50

# J_VEL_0 = 1.0
# J_ACC_0 = 1.0

J_VEL_1 = 0.5
J_ACC_1 = 1.0

# defined position for retrieval
SECOND_POSE_DX = 177.5
INSERT_DX = 90.0
RETRIEVAL_DX = SECOND_POSE_DX + INSERT_DX
PASS_POLL_DY = 500.0
PASS_POLL_DX = -60.0
SECOND_POSE_OFFSETT = 72
Z_OFFSET = 0.0

# defined position for storage
# storage 全部で 420mm 前後
STORAGE_INSERT = 290
SPACER_INSERT_DX = 100
INSERT_BOOK_TIP_DX = 60
INSERT_BOOK_FULL_DX = STORAGE_INSERT - SPACER_INSERT_DX - INSERT_BOOK_TIP_DX
PLACE_BOOK_DZ = -5.0

def _wrap_to_pi(rad: float) -> float:
        """角度差を [-pi, pi] に正規化（回転の無駄な大回りを避ける）"""
        return float((rad + np.pi) % (2.0 * np.pi) - np.pi)


class XArm7:
    """
    UR向けのRTDEベース関数群を xArm7 + xarm_python_sdk 用に書き直し

    - UR movej  -> set_servo_angle (joint space)
    - UR movel  -> set_position   (cartesian space straight-line)
    """

    def __init__(self, host: str = XARM_HOST, is_radian: bool = True):
        """
        :param host: xArm コントロールボックスの IP
        :param is_radian: xArmAPI の角度単位。True なら rad, False なら deg
        """
        self.host = host
        self.arm = XArmAPI(host, is_radian=is_radian)

        #clean error and warn
        if self.arm.warn_code != 0:
            self.arm.clean_warn()
        if self.arm.error_code != 0:
            self.arm.clean_error()

        # self.arm.set_tcp_offset([-226.96, -19.125, 54.0, 0.0, 0.0, 0.0]) # manual setting
        self.arm.set_tcp_offset([-226.96, -35.0, 54.0, 0.0, 0.0, 0.0]) # adjusted for book hand
        # self.arm.set_tcp_load(1.924, [41.86, -9.61, 67.48]) # preset : by Ufactory Studio error here

        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0) # mode 0: position mode

        self.arm.set_state(state=0)

        self.is_radian = is_radian

    # ========= low-level helper =========

    def disconnect(self):
        self.arm.disconnect()

    def _get_tcp_pose(self, is_radian: bool = True):
        """xArm の現在TCP姿勢を [x,y,z,roll,pitch,yaw] で返す"""
        code, pose = self.arm.get_position(is_radian=is_radian)
        if code != 0:
            raise RuntimeError(f"get_position failed, code={code}")
        return pose[:6]
    
    def _get_joint_angle(self, is_radian: bool = True):
        code, angles = self.arm.get_servo_angle(is_radian=is_radian)  # angles: list（7軸なら7要素）
        if code != 0:
            raise RuntimeError(f"get_position failed, code={code}")
        return angles[:7]
        

    def _moveJ(self, joints, velocity, acceleration, asynchronous: bool = False):
        """URの moveJ 相当: joint space point-to-point"""
        wait = not asynchronous
        # joints は rad 単位で渡す想定
        return self.arm.set_servo_angle(
            angle=joints,
            speed=velocity,
            mvacc=acceleration,
            is_radian=True,
            wait=wait,
        )

    def _moveL(self, pose, velocity, acceleration, asynchronous: bool = False):
        """URの moveL 相当: TCP直線補間"""
        wait = not asynchronous
        x, y, z, roll, pitch, yaw = pose
        return self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=velocity,
            mvacc=acceleration,
            is_radian=True,   # rpy は rad
            wait=wait,
            relative=False,
        )

    # ========= public helper =========

    def get_tcp_pose(self, is_radian: bool = True):
        """現在のTCP姿勢を取得 (デバッグ用)"""
        return self._get_tcp_pose(is_radian=is_radian)
    
    def get_joint_angle(self, is_radian: bool = True):
        """現在の関節角を取得 (デバッグ用)"""
        return self._get_joint_angle(is_radian=is_radian)

    # ========= UR版 moveJ_* の移植 =========

    def moveJ_to_init_Q(self,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False):
        """UR版 moveJ_to_init_Q"""
        joints_deg = INIT_Q_A5
        joints_rad = convert.deg_to_rad(joints_deg)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)
    def moveJ_to_barcode(self,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False):
        joints_deg = BARCODE_Q_A5
        joints_rad = convert.deg_to_rad(joints_deg)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)
    # def moveJ_to_init_Q_offset(self,
    #                            velocity: float = J_VEL_1,
    #                            acceleration: float = J_ACC_1,
    #                            asynchronous: bool = False):
    #     """元のコメントアウトされていた offset版を一応移植"""
    #     joints_deg = INIT_Q_A5_OFFSET
    #     joints_rad = convert.deg_to_rad(joints_deg)
    #     return self._moveJ(joints_rad, velocity, acceleration, asynchronous)

    # ========= UR版 moveL_* の移植 =========

    def moveL_to_init_pose(self,
                           pose=None,
                           velocity: float = TCP_VEL_1,
                           acceleration: float = TCP_ACC_1,
                           asynchronous: bool = False):
        """
        初期TCP姿勢へ直線移動。
        UR版では pose=INIT_TCP_POSE_A5 をそのまま渡していたが、
        ここでは pos[mm], rpy[deg] -> pos[mm], rpy[rad] に変換して使う。
        """
        if pose is None:
            pos = INIT_TCP_POSE_A5[:3]
            rpy_deg = INIT_TCP_POSE_A5[3:]
            rpy_rad = convert.deg_to_rad(rpy_deg)
            pose = list(pos) + list(rpy_rad)
            #print("djoint=",pose)
        return self._moveL(pose, velocity, acceleration, asynchronous)

    def moveL_relative(self,
                   next_pose_relative: list,
                   velocity: float = TCP_VEL_1,
                   acceleration: float = TCP_ACC_1,
                   asynchronous: bool = False): # TODO 公式のrelative関数使用したほうが良いかと
        """
        現在TCP姿勢 + 差分(next_pose_relative) に moveL
        next_pose_relative = [dx, dy, dz, droll, dpitch, dyaw]
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = [a + b for a, b in zip(curr_pose, next_pose_relative)]
        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    def moveJ_relative(
                self,
                djoints: list[float],
                velocity: float = J_VEL_1,
                acceleration: float= J_ACC_1,
                asynchronous: bool = False,
                ):
        """現在関節角 + 差分(djoints) に moveJ（rad）"""
        print("djoint=",djoints)
        curr_angle = self._get_joint_angle(is_radian=True)
        target = [c + d for c, d in zip(curr_angle, djoints)]
        return self._moveJ(target, velocity, acceleration, asynchronous)

    @staticmethod
    def culc_pos_relative_from_dy(dy: float):
        """
        画像座標→カメラ座標の結果から算出した，ハンドの y 方向移動量。
        UR版 culc_pos_relative_from_dy の移植。
        """
        pose_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        return pose_relative

    def moveL_to_retrieval_pose(self,
                                dy: float,
                                d_roll: float,
                                velocity: float = TCP_VEL_1,
                                acceleration: float = TCP_ACC_1,
                                asynchronous: bool = False):
        """
        UR版 moveL_to_retrieval_pose の移植。
        - y方向に dy だけ平行移動
        - roll に d_roll (rad) だけ加算
        """
        curr_pose = self._get_tcp_pose(is_radian=True)

        # 位置成分の更新
        pos_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)
        next_pose[3] += d_roll

        return self._moveL(next_pose, velocity, acceleration, asynchronous)

    def moveL_rot_relative(self,
                       d_roll: float,
                       velocity: float = TCP_VEL_1,
                       acceleration: float = TCP_ACC_1,
                       asynchronous: bool = False):
        """
        roll だけを d_roll (rad) 変化させる moveL。
        UR版 moveL_rot_relative の移植。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[3] += d_roll
        return self._moveL(next_pose, velocity, acceleration, asynchronous)

    def moveL_to_2nd_pos(self,
                         velocity: float = TCP_VEL_1,
                         acceleration: float = TCP_ACC_1,
                         asynchronous: bool = False):
        """
        SECOND_POSE_DX だけ x 方向に寄る。
        UR版 moveL_to_2nd_pos の移植。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[0] += SECOND_POSE_DX
        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    def move_to_target_xyz_and_roll(
        arm: "XArm7",
        *,
        p_robot_mm: np.ndarray,   # 目標 base座標 [mm]
        d_roll_rad: float,        # roll加算 [rad]
        sleep_s: float = 0.02,
        ):
        p_robot_mm = np.asarray(p_robot_mm, dtype=np.float64).reshape(3)

        # 現在TCP [mm,mm,mm, roll,pitch,yaw]
        curr = arm.get_tcp_pose(is_radian=True)
        curr_xyz = np.array(curr[:3], dtype=np.float64)

        dxyz = p_robot_mm - curr_xyz  # [mm]
        print("dxyz =", dxyz)
        print("d_roll (deg) =", np.degrees(d_roll_rad))
        # dx,dy,dz を全部動かす + rollだけ回す
        arm.moveL_relative([float(dxyz[0]), float(dxyz[1]), float(dxyz[2]), float(d_roll_rad), 0.0, 0.0],
                            asynchronous=False)
        time.sleep(sleep_s)

    def moveL_to_2nd_pos_with_offset(self,
                                     velocity: float = TCP_VEL_1,
                                     acceleration: float = TCP_ACC_1,
                                     asynchronous: bool = False):
        """
        x 方向に SECOND_POSE_DX, z 方向に Z_OFFSET を加える版。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[0] += SECOND_POSE_DX
        next_pose[2] += Z_OFFSET
        return self._moveL(next_pose, velocity, acceleration, asynchronous)

    def moveL_to_insert(self,
                        velocity: float = TCP_VEL_1,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DX だけ x 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [-INSERT_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )

    # def moveL_post_insert(self,
    #                       velocity: float = TCP_VEL_1,
    #                       acceleration: float = TCP_ACC_1,
    #                       asynchronous: bool = False):
    #     """
    #     INSERT_DX だけ戻る。
    #     UR版 moveL_post_insert の移植。
    #     """
    #     next_pose_relative = [-INSERT_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     return self.moveL_relative(
    #         next_pose_relative,
    #         velocity=velocity,
    #         acceleration=acceleration,
    #         asynchronous=asynchronous,
    #     )

    def moveL_post_grasp(self,
                         velocity: float = TCP_VEL_1,
                         acceleration: float = TCP_ACC_1,
                         asynchronous: bool = False):
        """
        把持後に x を INIT_TCP_POSE_A5[0] に戻す。
        UR版 moveL_post_grasp の移植。
        """
        curr_pose = self._get_tcp_pose(is_radian=True)
        next_pose = list(curr_pose)
        next_pose[0] += RETRIEVAL_DX
        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    # ========= placing in container (元コードの末尾) =========

    # def moveJ_to_pre_place(self,
    #                        pre_place_q_deg,
    #                        velocity: float = J_VEL_1,
    #                        acceleration: float = J_ACC_1,
    #                        asynchronous: bool = False):
    #     """
    #     UR版 moveJ_to_pre_place の一般形。
    #     PRE_PLACE_Q_DEG を外から渡す形にしてある。
    #     """
    #     joints_rad = convert.deg_to_rad(pre_place_q_deg)
    #     return self._moveJ(joints_rad, velocity, acceleration, asynchronous)

    # def moveL_to_place_1(self,
    #                      placing_pose,
    #                      velocity: float = TCP_VEL_1,
    #                      acceleration: float = TCP_ACC_1,
    #                      asynchronous: bool = False):
    #     """
    #     UR版 moveL_to_place_1 の一般形。
    #     PLACING_1_POSE を外から渡す。
    #     placing_pose は [x,y,z,roll(rad),pitch(rad),yaw(rad)]。
    #     """
    #     return self._moveL(placing_pose, velocity, acceleration, asynchronous)

    # def moveL_to_post_place_1(self,
    #                           post_place_pose,
    #                           velocity: float = TCP_VEL_1,
    #                           acceleration: float = TCP_ACC_2,
    #                           asynchronous: bool = False):
    #     """
    #     UR版 moveL_to_post_place_1 の一般形。
    #     POST_PLACE_POSE を外から渡す。
    #     """
    #     return self._moveL(post_place_pose, velocity, acceleration, asynchronous)4

    def pass_poll(self,
                  target_pose: list,
                  velocity: float = TCP_VEL_2,
                  acceleration: float = TCP_ACC_1,
                  asynchronous: bool = False):
        """
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        curr_pose = self.get_tcp_pose(is_radian=True)
        next_pose_relative = [t - c for t, c in zip(target_pose, curr_pose)]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        ) 
    def distant_hand(self,
                            velocity: float = J_VEL_1,
                            acceleration: float = J_ACC_1,
                            asynchronous: bool = False):
            """
            Joint6だけ回転
            """
            djoint = [0.0, 0.0, 0.0, 0.0, 0.0, DISTANCE_BOOKSHELF, 0.0]
            return self.moveJ_relative(
                djoint,
                velocity=velocity,
                acceleration=acceleration,
                asynchronous=asynchronous,
            )
    def rotate_hand(self,
                            velocity: float = J_VEL_1,
                            acceleration: float = J_ACC_1,
                            asynchronous: bool = False):
            """
            Joint7だけ回転
            """
            djoint = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, DISTANCE_BOOKSHELF]
            return self.moveJ_relative(
                djoint,
                velocity=velocity,
                acceleration=acceleration,
                asynchronous=asynchronous,
            )
    #進行方向左側の書架のみ，もしかしたら両方使えるかも
    def switch_gripper_pose(self,
                        joints_deg: list,
                        velocity: float = J_VEL_1,
                        acceleration: float = J_ACC_1,
                        asynchronous: bool = False
                        ):
        curr_joint = self.get_joint_angle(is_radian=True)
        target_rad = convert.deg_to_rad(joints_deg)
        djoint = [t - c for t, c in zip(target_rad, curr_joint)]
        return self.moveJ_relative(djoint, velocity, acceleration, asynchronous)
    
    
    
    def insert_container(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [0.0, 0.0, INSERT_CONTAINER, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
            
    def distant_container(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [0.0, 0.0, -INSERT_CONTAINER, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
    def container_offset(self,
                        container_offset : float,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        INSERT_DZ だけ Z 方向に直線移動。
        UR版 moveL_to_insert の移植。
        """
        next_pose_relative = [container_offset, 0.0, 0.0, 0.0, 0.0, 0.0]
        return self.moveL_relative(
            next_pose_relative,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
    
   


# Storage -----------------------------------------------------------------------------------------------------------------------------

    def moveL_to_storage_pose(self,
                                dy: float,
                                d_roll: float = 0,
                                velocity: float = TCP_VEL_1,
                                acceleration: float = TCP_ACC_1,
                                asynchronous: bool = False):
        """
        UR版 moveL_to_retrieval_pose の移植。
        - y方向に dy だけ平行移動
        - roll に d_roll (rad) だけ加算
        """
        curr_pose = self._get_tcp_pose(is_radian=True)

        # 位置成分の更新
        pos_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)
        next_pose[3] += d_roll

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    def moveL_to_slide(self, dy: float,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        """
        y方向に dy だけ平行移動
        """
        curr_pose = self._get_tcp_pose(is_radian=True)

        # 位置成分の更新
        pos_relative = [0.0, dy, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    def moveL_to_insert_spacer(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):


        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [SPACER_INSERT_DX, 0.0, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    def moveL_to_insert_book_full(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    

    def moveL_to_place_book(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):
        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [0.0, 0.0, PLACE_BOOK_DZ, 0.0, 0.0, 0.0]
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        return self._moveL(next_pose, velocity, acceleration, asynchronous)


    def moveL_to_post_storage(self,
                        velocity: float = TCP_VEL_1,
                        acceleration: float = TCP_ACC_1,
                        asynchronous: bool = False):
        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [-STORAGE_INSERT, 0.0, -PLACE_BOOK_DZ, 0.0, 0.0, 0.0]
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    
    def move_to_target_rpy_only(
        arm: "XArm7",
        *,
        target_pose_mm_rad: np.ndarray | list,  # [x,y,z,roll,pitch,yaw] xyz[mm], rpy[rad]
        sleep_s: float = 0.02,
        wrap_angle: bool = True,
    ):
        """
        現在の xyz は維持したまま、roll/pitch/yaw だけ target に合わせる
        """
        target = np.asarray(target_pose_mm_rad, dtype=np.float64).reshape(6)

        curr = arm.get_tcp_pose(is_radian=True)  # [mm,mm,mm, roll,pitch,yaw] (想定)
        curr_rpy = np.array(curr[3:6], dtype=np.float64)
        target_rpy = target[3:6]

        drpy = target_rpy - curr_rpy
        if wrap_angle:
            drpy = np.array([_wrap_to_pi(v) for v in drpy], dtype=np.float64)

        print("drpy (deg) =", np.degrees(drpy))

        # xyzは0、rpyだけ動かす（moveL_relative は [dx,dy,dz,droll,dpitch,dyaw]）
        arm.moveL_relative(
            [0.0, 0.0, 0.0, float(drpy[0]), float(drpy[1]), float(drpy[2])],
            asynchronous=False,

        )
        time.sleep(sleep_s)


    def move_to_target_xyz_only(
        arm: "XArm7",
        *,
        target_pose_mm_rad: np.ndarray | list,  # [x,y,z,roll,pitch,yaw] xyz[mm], rpy[rad]
        sleep_s: float = 0.02,
    ):
        """
        現在の roll/pitch/yaw は維持したまま、xyz だけ target に合わせる
        """
        target = np.asarray(target_pose_mm_rad, dtype=np.float64).reshape(6)

        curr = arm.get_tcp_pose(is_radian=True)  # [mm,mm,mm, roll,pitch,yaw] (想定)
        curr_xyz = np.array(curr[:3], dtype=np.float64)
        target_xyz = target[:3]

        dxyz = target_xyz - curr_xyz  # [mm]
        print("dxyz =", dxyz)

        # rpyは0、xyzだけ動かす
        arm.moveL_relative(
            [float(dxyz[0]), float(dxyz[1]), float(dxyz[2]), 0.0, 0.0, 0.0],
            asynchronous=False,
        )
        time.sleep(sleep_s)

    
   
# for test --------------------------------------------------------------------------------------------------------
    def moveJ_to_place_for_test(self,
                           velocity: float = J_VEL_1,
                           acceleration: float = J_ACC_1,
                           asynchronous: bool = False):
        """
        テスト用の配置姿勢へ moveJ
        """
        joints_rad = convert.deg_to_rad(PLACING_FOR_TEST_Q)
        return self._moveJ(joints_rad, velocity, acceleration, asynchronous)
    
    def moveL_to_point_m(
        self,
        p_xyz_m,
        *,
        keep_rpy: bool = True,
        rpy_rad=None,                 # keep_rpy=False のとき [roll,pitch,yaw]
        approach_dz_m: float = 0.05,  # まず5cm上へ寄ってから降りる（安全用）
        velocity: float = TCP_VEL_2,
        acceleration: float = TCP_ACC_2,
        asynchronous: bool = False,
    ):
        """
        base座標系の点 p_xyz_m=[x,y,z] (m) へ moveL する（内部で mm に変換して set_position に渡す）。
        """
        # 位置: m -> mm
        x_mm, y_mm, z_mm = p_xyz_m[0], p_xyz_m[1], p_xyz_m[2]   

        # 姿勢: 現在の rpy を維持するか、指定する
        if keep_rpy:
            curr = self._get_tcp_pose(is_radian=True)  # [x,y,z,roll,pitch,yaw] :contentReference[oaicite:3]{index=3}
            roll, pitch, yaw = curr[3], curr[4], curr[5]
        else:
            if rpy_rad is None:
                raise ValueError("rpy_rad must be provided when keep_rpy=False")
            roll, pitch, yaw = map(float, rpy_rad)

        # まず上から寄る（z+dz）→ 目標zへ
        if approach_dz_m is not None and float(approach_dz_m) != 0.0:
            pre_pose = [x_mm, y_mm, z_mm + float(approach_dz_m) * 1000.0, roll, pitch, yaw]
            self._moveL(pre_pose, velocity, acceleration, asynchronous)  # :contentReference[oaicite:4]{index=4}

        pose = [x_mm, y_mm, z_mm, roll, pitch, yaw]
        return self._moveL(pose, velocity, acceleration, asynchronous)  # :contentReference[oaicite:5]{index=5}

    # ========= public generic moveJ =========
    def moveJ(self,
              joints_rad: list,
              velocity: float = J_VEL_1,
              acceleration: float = J_ACC_1,
              asynchronous: bool = False):
        """
        汎用 moveJ（関節角は rad 指定）
        """
        return self._moveJ(
            joints_rad,
            velocity=velocity,
            acceleration=acceleration,
            asynchronous=asynchronous,
        )
    
    def move_L_to_insert_book_tip(self,
                        velocity: float = TCP_VEL_2,
                        acceleration: float = TCP_ACC_2,
                        asynchronous: bool = False):


        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)

        return self._moveL(next_pose, velocity, acceleration, asynchronous)
    
    def move_L_to_insert_book_tip2(self,
                    velocity: float = TCP_VEL_2,
                    acceleration: float = TCP_ACC_2,
                    asynchronous: bool = False):


        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [0.0, -200.0, 0.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]

        # RPY の roll を加算 (curr_pose[3:] はすでに rpy(rad) 前提)

        return self._moveL(next_pose, velocity, acceleration, asynchronous)


    def move_L_to_TakePhoto(
        self,
        dis: float,   # ← 先に置く（必須）
        velocity: float = TCP_VEL_2,
        acceleration: float = TCP_ACC_2,
        asynchronous: bool = False,
    ):
        curr_pose = self._get_tcp_pose(is_radian=True)
        pos_relative = [float(dis) - 300.0, 0.0, -120.0, 0.0, 0.0, 0.0]
        print('pos_relative : ', pos_relative)
        next_pose = [a + b for a, b in zip(curr_pose, pos_relative)]
        return self._moveL(next_pose, velocity, acceleration, asynchronous)



class TakePhotoController(Node):
    def __init__(self, arm: "XArm7"):
        super().__init__("take_photo_controller")

        self.arm = arm
        self.base_mm = 800.0

        # wall_distance が "m で来る" 場合は True にする（とりあえず False で開始）
        self.wall_distance_is_meter = False

        self._moved_once = False
        self._lock = threading.Lock()

        self.sub = self.create_subscription(
            Float32,
            "/wall_distance",
            self._cb_wall_distance,
            qos_profile_sensor_data,   # センサ系トピック想定
        )

        self.get_logger().info("Waiting /wall_distance ...")

    def _cb_wall_distance(self, msg: Float32):
        # 1回だけ動かす
        with self._lock:
            if self._moved_once:
                return
            self._moved_once = True

        # ROS2で受け取った距離（m想定）
        wall_distance = float(msg.data)
        wall_distance_mm = wall_distance * 1000.0

        # dis = 800mm - wall_distance_mm
        dis = self.base_mm - wall_distance_mm  # mm

        # 安全のため移動量を制限（必要に応じて調整）
        dis = max(min(dis, 800.0), -800.0)

        self.get_logger().info(
            f"wall_distance={wall_distance:.3f} m ({wall_distance_mm:.1f} mm) "
            f"-> dis={dis:.1f} mm"
        )

        try:
            # 1) まず撮影位置へ
            self.get_logger().info("Move: TakePhoto")
            self.arm.move_L_to_TakePhoto(dis=dis, asynchronous=False)

            # 2) +10cm（例：右/左は座標系依存。今は y+100mm）
            self.get_logger().info("Move: tip1 (dy=+100mm)")
            self.arm.move_L_to_insert_book_tip(asynchronous=False)

            # 3) -20cm（y-200mm） → 結果として最初から -10cm 側へ行く
            self.get_logger().info("Move: tip2 (dy=-200mm)")
            self.arm.move_L_to_insert_book_tip2(asynchronous=False)

            pose = self.arm.get_tcp_pose(is_radian=True)
            self.get_logger().info(f"Done. Current TCP pose (rad) = {pose}")

        except Exception as e:
            self.get_logger().error(f"Arm motion failed: {e}")

        # 終了（全部動かしたら終了）
        try:
            self.arm.disconnect()
        except Exception:
            pass
        rclpy.shutdown()



def main():
    rclpy.init()

    arm = XArm7()
    node = TakePhotoController(arm)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        try:
            arm.disconnect()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

