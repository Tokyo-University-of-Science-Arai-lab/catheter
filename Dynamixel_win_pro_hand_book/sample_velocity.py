import dynamixel_classes_for_windows as dyna
import kbhit
import time

try:
    dxl = dyna.Dynamixel("COM7", 57600)  # インスタンス化
    kb = kbhit.KBHit()  # キーボード入力のクラス立ち上げ
    time.sleep(0.5)  # 通信が確立するまでちょっと待つ（待たなくても良いが高速すぎるとバッファが溢れ命令実行漏れが発生する）
    dxl.set_mode_velocity(3)  # 速度制御モードに設定
    max_1 = 300
    dxl.set_max_velocity(3, max_1)  # 速度の上限を設定
    dxl.enable_torque(3)  # トルクをオンにする（手で動かせなくなる）

    now_goal1 = 0

    print("---------------------------------")
    print("     Dynamixel READY TO MOVE   ")
    print("---------------------------------")

    while 1:
        if kb.kbhit():  # キーボードの打鍵があったら（無いときはここで待ち状態になる）
            c = ord(kb.getch())  # キー入力を数値（番号）に変換
            if c == 75:  # L arrow  #左矢印なら
                now_goal1 = now_goal1 + 5  # 目標値を50増やす
            elif c == 77:  # R arrow
                now_goal1 = now_goal1 - 5

            now_goal1 = max((-1) * max_1, min(now_goal1, max_1))  # 数値的に上下限を超えないようにする
            dxl.write_velocity(3, now_goal1)  # Dynamixelに数値を書き@込む

except KeyboardInterrupt:  # Ctrl+Cが押されたら
    dxl.disable_torque(3)  # トルクをオフにする（手で動かせるようになる）
    dxl.close_port()  # ポートを切断する
    kb.set_normal_term()  # キー入力に対する応答を通常時に戻す？）
