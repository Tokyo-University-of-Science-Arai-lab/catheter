#!/usr/bin/env python3
"""グリッパーを閉じるだけのスクリプト。ハードウェアエラー状態でも reboot してから閉じる。"""
import time
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from Dynamixel_win_pro_hand_book.dynamixel_cross_platform import Dynamixel
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook_retrieval

PORT       = HandBook_retrieval.PORT
BAUDRATE   = HandBook_retrieval.BAUDRATE
GRIPPER_ID = HandBook_retrieval.GRIPPER_ID

# reboot でハードウェアエラーをクリア
dxl = Dynamixel(port=PORT, baudrate=BAUDRATE)
print("reboot...")
dxl.reboot(GRIPPER_ID)
dxl.close_port()   # 必ずポートを閉じてから再オープン
time.sleep(1.0)    # reboot完了待ち

# 再接続して閉じる
dxl = HandBook_retrieval.init_dynamixels()
HandBook_retrieval.close_to_home(dxl)
print("ハンドを閉じました")
dxl.close_port()
