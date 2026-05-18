from xarm7.control.xarm7 import XArm7
from xarm.wrapper import XArmAPI
import numpy as np
from pathlib import Path
import time
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook
from xarm7.control.xarm7 import XArm7
import math
from pathlib import Path
import os
import rclpy
from rclpy.node import Node
from xarm7.control.xarm_init_to_capture_integration import WaypointPlayerNode
from xarm7.control.xarm_monitor import XArmMonitor


#BOOK_BARCODE_2 = [-37.2, -84.8, 165.4, 86.2, 94.9, 13.1, -43]
BOOK_BARCODE_2 = [-36.9, -75.2, 159.3, 80.5, 83.8, 18.8, -34.9]
BOOK_CAPTURE = -210.0  #-310.0
CONTAINER_TILT_DEG = 13.0

def fatal_exit(arm, msg):
    print(f"\n🔥 FATAL: {msg}")
    try:
        arm.emergency_stop()
    except Exception:
        pass
    os._exit(1)

def Move_to_Container(offset: float, arm: XArm7, waypoint_node, HandMotors):

    print("[Move_to_Container] offset:", offset)

    #HandMotors = HandBook.init_dynamixels()

    #arm.switch_gripper_pose(BOOK_BARCODE_2, acceleration=0.1, velocity=0.7)

    waypoint_node.reset()

    waypoint_node.play_direct(
        "/home/book/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/move_to_container_t.yaml"
    )
    while not waypoint_node.is_finished():
        time.sleep(0.1)

    waypoint_node.reset()


    # --- 変更箇所：offset の値によって YAML ファイルを切り替える ---
    if offset < 30.0:
        yaml_file = "container_offset_30.0.yaml"
    elif offset < 60.0:
        yaml_file = "container_offset_60.0.yaml"
    elif offset < 90.0:
        yaml_file = "container_offset_90.0.yaml"    
    elif offset < 120.0:
        yaml_file = "container_offset_120.0.yaml"   
    elif offset < 150.0:
        yaml_file = "container_offset_150.0.yaml"  
    elif offset < 180.0:
        yaml_file = "container_offset_180.0.yaml"       
    elif offset < 210.0:       
        yaml_file = "container_offset_210.0.yaml"       
    elif offset < 240.0:       
        yaml_file = "container_offset_240.0.yaml"   
    elif offset < 270.0:       
        yaml_file = "container_offset_270.0.yaml"   
    elif offset < 300.0: 
        yaml_file = "container_offset_300.0.yaml"
    elif offset < 330.0:
        yaml_file = "container_offset_330.0.yaml"
    else:
        print('本がコンテナにいっぱいです') 
        waypoint_node.reset()
        waypoint_node.play_direct(
            "/home/book/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/init.yaml"
        )
        while not waypoint_node.is_finished():
            time.sleep(0.1)
            
        return  # ここで関数を抜け、これ以降の処理は行わない
        
    yaml_path = f"/home/book/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/{yaml_file}"
    print(f"[container_offset] 使用する軌道ファイル: {yaml_file}")
    waypoint_node.play_direct(yaml_path)
    while not waypoint_node.is_finished():
        time.sleep(0.1)
    waypoint_node.reset()
        
    # arm.container_offset(2)
    # input("Enter押すと開始")
    theta = math.radians(CONTAINER_TILT_DEG)
    # slide = offset + 16.0
    z_drop = BOOK_CAPTURE + offset * math.tan(theta)
    # print("[Move_to_Container] slide:", slide)
    print("[Move_to_Container] z_drop:", z_drop)
    # arm.container_offset(slide)
    arm.moveL_z_offset(z_drop)
    HandBook.open_until_full(HandMotors, asynchronous=False)

    #waypoint_node.play_direct(yaml_path)
    #while not waypoint_node.is_finished():
    #    time.sleep(0.1)
    waypoint_node.reset()
    waypoint_node.play_direct(
        "/home/book/pro_book/pro_hand_book_python/ros2_ws/src/xarm7_teaching/config/move_to_container_final.yaml"
    )
    HandBook.grasp(HandMotors)
    while not waypoint_node.is_finished():
        time.sleep(0.1)

    

#    arm.switch_gripper_pose(CONTAINER_1,velocity=0.6, acceleration=1.5)
#    print("phase1 finished")
#    
# arm.pass_poll(AFTER_PASS_POLL,velocity=100.0, acceleration=300.0)
#    arm.switch_gripper_pose(CONTAINER_3,velocity=0.6, acceleration=1.5)
#    arm.switch_gripper_pose(CONTAINER_5,velocity=0.5, acceleration=1.5)
#    #arm.switch_gripper_pose(CONTAINER_4)
#    print("phase4 finished")

#    slide = offset + 160.0
#    arm.container_offset(slide)
#    arm.moveL_z_offset(BOOK_CAPTURE - offset * np.tan(0.2267))
#    #上下機構 500mm から 1020mm
#    HandBook.open_until_full(HandMotors, asynchronous = False) # ハンド全開
#    time.sleep(1.0)

#    arm.moveL_z_offset(-(BOOK_CAPTURE - offset * np.tan(13)))
#    HandBook.grasp(HandMotors) # ハンド閉じる
#    arm.container_offset(-offset,velocity=80.0, acceleration=200.0)

#    arm.switch_gripper_pose(CONTAINER_3,velocity=0.6, acceleration=1.5)

#    arm.switch_gripper_pose(CONTAINER_2,velocity=0.6, acceleration=1.5)

#    arm.pass_poll(BEFORE_PASS_POLL)

#    arm.switch_gripper_pose(CONTAINER_1,velocity=0.6, acceleration=1.5)

#    arm.switch_gripper_pose(CONTAINER_0,velocity=0.6, acceleration=1.5)

#    arm.moveJ_to_capture_right()
   
def main():

    rclpy.init()

    node = Node("move_to_container_test")
    arm = XArm7(node)
    monitor = XArmMonitor(arm)

    waypoint_node = WaypointPlayerNode(
        node_name="waypoint_player",
        arm=arm,
        yaml_path="",
        monitor=monitor
    )

    try:
        HandMotors = HandBook.init_dynamixels()
        Move_to_Container(offset, arm, waypoint_node, HandMotors)
    except RuntimeError as e:
        print("🚫 xArm error detected:", e)
        print("🛑 Stop program.")
        return

    rclpy.shutdown()

    
if __name__ == "__main__":
   offset = float(input("offset[mm]? ").strip().lower().replace("mm", ""))

   main()

