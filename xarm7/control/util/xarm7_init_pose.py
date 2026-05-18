from xarm.wrapper import XArmAPI
import time
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


#XARM_HOST = "192.168.1.197" #DC
XARM_HOST = "192.168.1.208" #AC

# INITIAL_POSE_Q = [
#     -22.9,
#     84.3,
#     -163.1,
#     56.3,
#     -154.6,
#     42.5,
#     -177.5
# ] # 2号館上から3段目,疑似上下機構940mm

#INITIAL_POSE_Q = [
#    -51.5,
#    58.5,
#    -209.2,
#    29.6,
#    -204.5,
#    95,
#    -114.1
#] # 2号館上から4段目,疑似上下機構940mm  収納帯引っ掛かり再現用
INITIAL_POSE_Q = [
    90.0,
    0.0,
    163.89,
    43.55,
    -357.58,
    40.8,
    70.73
]
# INITIAL_POSE_Q = [
#     -118.0,
#     -15.7,
#     4.9,
#     44.8,
#     1.0,
#     60.3,
#     66.0
# ] # 2号館上から3段目,上下機構400mm  逆側（進行方向左側）書籍

CONTAINER_1 = [
   -246.4,
   12.3,
   -244.9,
   50.4,
   -302.6,
   115.7,
   52.2
]

VELOCITY = 0.1
ACCELERATION = 0.1


def init_xarm(ip):
    #Instantiation API
    arm = XArmAPI(ip)
    time.sleep(0.5)
    
    #clean error and warn
    if arm.warn_code != 0:
        arm.clean_warn()
    if arm.error_code != 0:
        arm.clean_error()
    #Enable the robot
    arm.motion_enable(enable=True)
    #set mode and state
    arm.set_mode(0)
    arm.set_state(0)

    return arm

def del_xarm(arm):
    arm.disconnect()

if __name__ == "__main__":
    xarm = init_xarm(XARM_HOST)
    xarm.set_servo_angle(angle=INITIAL_POSE_Q, wait=True)
    #xarm.set_servo_angle(angle=CONTAINER_1, wait=True)
    del_xarm(xarm)