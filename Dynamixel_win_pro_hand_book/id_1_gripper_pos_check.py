from .dynamixel_cross_platform import Dynamixel
from . import kbhit_lin as kbhit
import time


try:

    dxl = Dynamixel("/dev/book_hand", 57600) #インスタンス化
    kb = kbhit.KBHit() #キーボード入力のクラス立ち上げ
    time.sleep(0.5)  #通信が確立するまでちょっと待つ（待たなくても良いが高速すぎるとバッファが溢れ命令実行漏れが発生する）
    dxl.set_mode_ex_position(1) # 拡張位置制御モードに設定

    init_pos  = int(dxl.read_position(1)) #現在の位置を取得し目標値の初期値として代入
    print("current position = ", init_pos)
    
    curr_pos = init_pos
    curr_des_pos = init_pos
    last_sent_pos = curr_des_pos # added

    rot_gain = 50 # キー入力1つで何deg 回転させるか（1→0.0043 deg）50: 0.21 deg
    # motor 1回転で4096段階の位置があるらしい
    dxl.enable_torque(1) #トルクをオンにする（手で動かせなくなる）
    time.sleep(1)

    print("---------------------------------")
    print("   Dynamixel READY TO MOVE  ")
    print("---------------------------------")

    while True:

      if kb.kbhit(): #キーボードの打鍵があったら（無いときはここで待ち状態になる）
        ch = kb.getch()
        
        if ch == 'K': #L arrow
          curr_des_pos += rot_gain  # 目標量増加（グリッパ 開）
          print(dxl.read_position(1))
          
        elif ch == 'M': #R arrow
          curr_des_pos -=  rot_gain # 目標値減少（グリッパ 閉）
          print(dxl.read_position(1))

      if curr_des_pos != last_sent_pos:
        dxl.write_position(1, curr_des_pos)
        last_sent_pos = curr_des_pos

      # time.sleep(0.001)

except KeyboardInterrupt:
  
    print(dxl.read_position(1)) # last position check
    dxl.disable_torque(1)
    dxl.close_port()
    kb.set_normal_term()
