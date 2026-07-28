"""
グリッパー(ID=1)の現在状態を1回だけ読んで表示する．
reboot()は多回転カウンタをリセットしてしまうため、ここでは呼ばない．
ハンドは動かさない（トルクもオンにしない）ので、閉じた状態のまま安全に実行できる．
"""
from .dynamixel_cross_platform import Dynamixel

GRIPPER_ID = 1

dxl = Dynamixel("/dev/ttyUSB0", 57600)

pos = dxl.read_position(GRIPPER_ID)
vel = dxl.read_velocity(GRIPPER_ID)
torque_on = dxl.read_torque_enable(GRIPPER_ID)
moving = dxl.read_moving(GRIPPER_ID)
pwm = dxl.read_pwm(GRIPPER_ID)
current = dxl.read_current(GRIPPER_ID)
temp = dxl.read_temperature(GRIPPER_ID)
homing_offset = dxl.read_homing_offset(GRIPPER_ID)

print("---------------------------------")
print(f"present position   = {pos}  (turn={pos // 4096}, phase={pos % 4096})")
print(f"present velocity   = {vel}")
print(f"torque_enable      = {torque_on}")
print(f"moving             = {moving}")
print(f"present PWM        = {pwm}")
print(f"present current    = {current}")
print(f"present temperature= {temp} C")
print(f"homing_offset      = {homing_offset}")
print("---------------------------------")

dxl.close_port()
