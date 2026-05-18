import dynamixel_classes_for_windows as dyna
import kbhit
import time

try:
  ###---------
  dxl = dyna.Dynamixel("COM7",57600) #インスタンス化
  kb = kbhit.KBHit() #キーボード入力のクラス立ち上げ
  time.sleep(0.5)  #通信が確立するまでちょっと待つ（待たなくても良いが高速すぎるとバッファが溢れ命令実行漏れが発生する）
  dxl.set_mode_position(1) #位置（角度）制御モードに設定
  min_1=0
  max_1=4095
  dxl.set_min_max_position(1,min_1,max_1) #位置の上下限を設定
  now_goal1=int(dxl.read_position(1)) #現在の位置を取得し目標値の初期値として代入
  dxl.enable_torque(1) #トルクをオンにする（手で動かせなくなる）
  print("---------------------------------")
  print("   Dynamixel READY TO MOVE  ")
  print("---------------------------------")

  while 1:
    if kb.kbhit(): #キーボードの打鍵があったら（無いときはここで待ち状態になる）
      c = ord(kb.getch()) #キー入力を数値（番号）に変換
      if c==75: #L arrow #左矢印なら
        now_goal1=now_goal1+50  #目標値を50増やす
      elif c==77: #R arrow
        now_goal1=now_goal1-50
      now_goal1=max(min_1,min(now_goal1,max_1)) #数値的に上下限を超えないようにする
      dxl.write_position(1,now_goal1)  #Dynamixelに数値を書き込む

except KeyboardInterrupt: #Ctrl+Cが押されたら
  dxl.disable_torque(1) #トルクをオフにする（手で動かせるようになる）
  dxl.close_port() #ポートを切断する
  kb.set_normal_term() #キー入力に対する応答を通常時に戻す？）
