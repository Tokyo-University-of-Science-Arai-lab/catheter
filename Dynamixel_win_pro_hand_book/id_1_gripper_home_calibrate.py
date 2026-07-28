"""
グリッパー(ID=1)の「本当に閉じきった位置」を、目視ではなくストール検知(grasp()と同じ考え方)で
自動的に求めるキャリブレーションスクリプト．

手順:
  1. 現在位置から、閉方向へ確実に機械的な限界を超える分だけ目標値を送る．
  2. 速度が閾値(thresh.vel)を下回ったら「何か（機械的な限界）に当たった」と判定し停止する．
  3. その時点の位置を「真の閉じきった位置」として表示する．

reboot()は多回転カウンタをリセットしてしまうため呼ばない．
"""
import time
from .dynamixel_cross_platform import Dynamixel
from . import HandBook_Retrieval as cfgmod

GRIPPER_ID = 1
VELOCITY_THRESHOLD = cfgmod.VELOCITY_THRESHOLD
OVERSHOOT_MARGIN_STEP = 20000  # 現在位置からこの分だけ閉方向(負)へ目標を送る．機構の全可動域より確実に大きい値
SETTLE_TIME_SEC = 0.3          # 動き始める前の速度ゼロを誤検知しないための猶予
TIMEOUT_SEC = 8.0

dxl = Dynamixel("/dev/ttyUSB0", 57600)
dxl.set_mode_ex_position(GRIPPER_ID)

start_pos = dxl.read_position(GRIPPER_ID)
target_pos = start_pos - OVERSHOOT_MARGIN_STEP
print(f"start position = {start_pos}")
print(f"target position (closing direction) = {target_pos}")

dxl.enable_torque(GRIPPER_ID)
dxl.write_position(GRIPPER_ID, target_pos)

time.sleep(SETTLE_TIME_SEC)  # 動き始めの加速期間は速度判定しない

deadline = time.time() + TIMEOUT_SEC
stalled = False
while time.time() < deadline:
    vel = dxl.read_velocity(GRIPPER_ID)
    pos = dxl.read_position(GRIPPER_ID)
    print(f"  pos={pos}  vel={vel}")
    if abs(vel) < VELOCITY_THRESHOLD:
        stalled = True
        break
    time.sleep(0.05)

dxl.disable_torque(GRIPPER_ID)

final_pos = dxl.read_position(GRIPPER_ID)
print("---------------------------------")
if stalled:
    print("ストール検知: 機械的な限界に到達したと判定しました．")
else:
    print(f"タイムアウト({TIMEOUT_SEC}秒)：ストールを検知できませんでした．目標値やマージンを見直してください．")
print(f"final position = {final_pos}  (turn={final_pos // 4096}, phase={final_pos % 4096})")
print("この値を Dynamixel_config.yaml の pos.gripper.close に設定してください．")
print("---------------------------------")

dxl.close_port()
