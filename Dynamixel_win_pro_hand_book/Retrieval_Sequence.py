from dynamixel_cross_platform import Dynamixel
from util.cfg_dict_loader import DynamixelCfg
import kbhit_lin as kbhit
import time


cfg = DynamixelCfg('./config/Dynamixel_config.yaml')

PORT = cfg.id.port
BAUDRATE = cfg.id.baudrate

GRIPPER_ID = cfg.id.gripper
SP_ROT_ID = cfg.id.spacer_rot
SP_LIN_ID = cfg.id.spacer_lin

GRIPPER_CLOSE = cfg.pos.gripper.close
GRIPPER_OPEN = cfg.pos.gripper.range + GRIPPER_CLOSE
GRIPPER_ROT_GAIN = cfg.cont_gain.gripper.pos_cont

VELOCITY_THRESHOLD = cfg.thresh.vel
POSITION_THRESHOLD = cfg.thresh.pos



def init_dynamixels():
    
    dxl = Dynamixel(port=PORT, baudrate=BAUDRATE)
    dxl.set_mode_ex_position(GRIPPER_ID)
    print(f'Gripper Position : {dxl.read_position(GRIPPER_ID)}')
    time.sleep(1)
    
    print("---------------------------------")
    print("   Dynamixel READY TO MOVE  ")
    print("---------------------------------")

    return dxl

def open_servo_key(dxl):

    kb = kbhit.KBHit()

    dxl.enable_torque(GRIPPER_ID)
    curr_des_pos = dxl.read_position(GRIPPER_ID)
    last_sent_pos = curr_des_pos

    while True:
        if kb.kbhit():
            ch = kb.getch()
            if ch == 'K': # L arrow : grippen open
                curr_des_pos += GRIPPER_ROT_GAIN
                
            elif ch == 'M': # R arrow : gripper close
                curr_des_pos -=  GRIPPER_ROT_GAIN
                
            elif ch in ('g', 'G'):  # g/G key : break and goto next sequence
                break
            
        curr_des_pos = max(GRIPPER_CLOSE, min(curr_des_pos, GRIPPER_OPEN)) # range limitation
        
        if curr_des_pos != last_sent_pos:
            dxl.write_position(GRIPPER_ID, curr_des_pos)  # set gripper position

def grasp(dxl):
    
    print("grasping object until detection")
    dxl.write_position(GRIPPER_ID, GRIPPER_CLOSE)
    time.sleep(0.05)
    while True:
        curr_vel = dxl.read_velocity(GRIPPER_ID)
        if abs(curr_vel) < VELOCITY_THRESHOLD: # grasp detection by velocity
            print('grasp detected')
            dxl.disable_torque(GRIPPER_ID)
            break
        
        elif GRIPPER_CLOSE - POSITION_THRESHOLD < dxl.read_position(GRIPPER_ID) < GRIPPER_CLOSE + POSITION_THRESHOLD:
            print('gripper close')
            dxl.disable_torque(GRIPPER_ID)
            break



if __name__ == '__main__':

    try:
        
        dxl = init_dynamixels()
        open_servo_key(dxl)

        grasp(dxl)
        time.sleep(1)
        print("gripper closed")
        dxl.disable_torque(GRIPPER_ID)
        dxl.close_port()
        
    except KeyboardInterrupt:
        
        dxl.disable_torque(GRIPPER_ID)
        dxl.close_port()




