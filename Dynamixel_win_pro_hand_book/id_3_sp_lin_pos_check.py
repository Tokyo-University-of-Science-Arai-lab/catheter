from .dynamixel_cross_platform import Dynamixel
#import kbhit_lin as kbhit
import time
from . import kbhit_lin as kbhit
import ctypes

#  back : 1495
# front : -190359
SP_LIN_BACK = 511
SP_LIN_FRONT = SP_LIN_BACK - 196854

def check_max_min(dxl, kb):

    while 1:
  #   curr_des_pos = dxl.read_position(1)

      if kb.kbhit(): #キーボードの打鍵があったら（無いときはここで待ち状態になる）
        c = ord(kb.getch()) #キー入力を数値（番号）に変換
        if c==75: #L arrow #左矢印なら
            dxl.write_position(3,  SP_LIN_BACK)
        elif c==77: #R arrow
            dxl.write_position(3,  SP_LIN_FRONT)
        elif c == ord('q'):
            dxl.write_position(3, dxl.read_position(3))
        # curr_des_pos = max(min_1,min(curr_des_pos, max_1)) #数値的に上下限を超えないようにする
        # dxl.write_position(3, curr_des_pos)  #Dynamixelに数値を書き込む
        # print(dxl.read_position(1))

def position_check(dxl, kb):

      init_pos  = dxl.read_position(3) #現在の位置を取得し目標値の初期値として代入
      curr_des_pos = init_pos
      last_sent_pos = curr_des_pos
      
      print("current position = ", init_pos)
      # curr_pos = init_pos

      rot_gain = 100
      while 1:
        # curr_des_pos = dxl.read_position(3)

        if kb.kbhit(): #キーボードの打鍵があったら（無いときはここで待ち状態になる）
          ch = kb.getch() #キー入力を数値（番号）に変換
          if ch == 'K': # L arrow #左矢印なら
            curr_des_pos += rot_gain  
          elif ch == 'M': # R arrow
            curr_des_pos -=  rot_gain

        if curr_des_pos != last_sent_pos:
          dxl.write_position(3, curr_des_pos)
          last_sent_pos = curr_des_pos
          # print("current velocity = ", dxl.read_velocity(3))
          print("current position = ", dxl.read_position(3))



if __name__ == '__main__':
  try:
    ###---------
      dxl = Dynamixel("/dev/book_hand", 57600) #インスタンス化
      kb = kbhit.KBHit() #キーボード入力のクラス立ち上げ
      time.sleep(0.5)  #通信が確立するまでちょっと待つ（待たなくても良いが高速すぎるとバッファが溢れ命令実行漏れが発生する）
      dxl.set_mode_ex_position(3) # 拡張位置制御モードに設定
      time.sleep(1)

      print("current position = ", dxl.read_position(3))

      dxl.enable_torque(3) #トルクをオンにする（手で動かせなくなる）
      time.sleep(1)
      print("---------------------------------")
      print("   Dynamixel READY TO MOVE  ")
      print("---------------------------------")

      position_check(dxl, kb)
      # check_max_min(dxl, kb)

      dxl.disable_torque(3)
      dxl.close_port()

  except KeyboardInterrupt:
      # dxl.write_position(3, SP_LIN_BACK)
      # time.sleep(30)

      dxl.disable_torque(3)
      dxl.close_port()
      kb.set_normal_term()
      # print(dxl.read_position(1))


  # memo init_pos : 22680
