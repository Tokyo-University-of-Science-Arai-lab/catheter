from xarm7.control.xarm7 import XArm7
from xarm.wrapper import XArmAPI
import numpy as np
import math
from pathlib import Path
import time
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook
from xarm7.control.xarm7 import XArm7
from xarm.wrapper import XArmAPI
import numpy as np
import math
from pathlib import Path
import time
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook
import os
import rclpy
from rclpy.node import Node

CONTAINER_0 = [
    90.0,
    6.1,
    138.1,
    23.6,
    -4.7,
    28.7,
    112.3
]#p1
CONTAINER_1 = [
    148.0,
    -10.4,
    180.1,
    17.6,
    39.4,
    15.2,
    200.9
]#p1

AFTER_PASS_POLL = [
   123.8,
   80.1,
   84.6,
   np.radians(180.0),
   np.radians(0.0),
   np.radians(90.0)
]#p1
BEFORE_PASS_POLL = [
   170.4,
   -390.1,
   19.0,
   np.radians(179.1),
   np.radians(-8.9),
   np.radians(88.4)
]#p2
CONTAINER_2 = [
   221.94,
   -36.54,
   208.59,
   57.76,
   147.98,
   -31.34,
   182.91
]

CONTAINER_3 = [
   221.48,
   -29.8,
   201.58,
   40.03,
   152.5,
   73.34,
   177.94
]#p3
CONTAINER_4 = [
   228.67,
   -50.73,
   194.86,
   49.33,
   149.35,
   81.61,
   174.48
]#p4
CONTAINER_5 = [
   205.5,
   -67.7,
   224.1,
   39.3,
   148.0,
   80.7,
   137.2
]# p5, オフセットが大きいときのみ使用
CONTAINER_6 = [
   90.0,
   0.0,
   163.89,
   43.55,
   2.42,
   40.8,
   130.7
]

BOOK_CAPTURE = -265 #-310.0
# BOOK_CAPTURE = -460 + x # x は本の奥行きの長さとする

def fatal_exit(arm, msg):
    print(f"\n🔥 FATAL: {msg}")
    try:
        arm.emergency_stop()
    except Exception:
        pass
    os._exit(1)

def Move_to_Container(offset : float, arm : XArm7):
   print("[Move_to_Container] offset:", offset)

   HandMotors = HandBook.init_dynamixels() # TODO class化
   #HandMotors = HandBook.init_dynamixels() # TODO class化
   arm.moveJ_to_capture_right()
   arm.switch_gripper_pose(CONTAINER_0,velocity=0.6, acceleration=1.5)
   arm.switch_gripper_pose(CONTAINER_1,velocity=0.6, acceleration=1.5)
   print("phase1 finished")
   arm.pass_poll(AFTER_PASS_POLL,velocity=100.0, acceleration=300.0)
   arm.switch_gripper_pose(CONTAINER_3,velocity=0.6, acceleration=1.5)
   arm.switch_gripper_pose(CONTAINER_5,velocity=0.5, acceleration=1.5)
   #arm.switch_gripper_pose(CONTAINER_4)
   print("phase4 finished")

   slide = offset + 160.0
   arm.container_offset(slide)
   arm.moveL_z_offset(BOOK_CAPTURE - offset * np.tan(0.2267))
   #上下機構 500mm から 1020mm
   HandBook.open_until_full(HandMotors, asynchronous = False) # ハンド全開
   time.sleep(1.0)

   arm.moveL_z_offset(-(BOOK_CAPTURE - offset * np.tan(13)))
   HandBook.grasp(HandMotors) # ハンド閉じる
   arm.container_offset(-offset,velocity=80.0, acceleration=200.0)

   arm.switch_gripper_pose(CONTAINER_3,velocity=0.6, acceleration=1.5)

   arm.switch_gripper_pose(CONTAINER_2,velocity=0.6, acceleration=1.5)

   arm.pass_poll(BEFORE_PASS_POLL)

   arm.switch_gripper_pose(CONTAINER_1,velocity=0.6, acceleration=1.5)

   arm.switch_gripper_pose(CONTAINER_0,velocity=0.6, acceleration=1.5)

   arm.moveJ_to_capture_right()
   
def main():

    rclpy.init()

    node = Node("move_to_container_test")
    arm = XArm7(node)

    for offset in [-20.0,-50.0,-70.0,-100.0,-120.0,-140.0,-160.0,-180.0,-200.0]:
        try:
            Move_to_Container(offset, arm)
        except RuntimeError as e:
            print("🚫 xArm error detected:", e)
            print("🛑 Stop program.")
            return

        input("次のオフセットに進みますか？")

    rclpy.shutdown()

    
if __name__ == "__main__":
   #offset = float(input("offset[mm]? ").strip().lower().replace("mm", ""))
   #Move_to_Container(-offset)
   
   main()

