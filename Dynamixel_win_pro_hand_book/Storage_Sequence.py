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
RELEASE_GAIN = cfg.pos.gripper.release

VELOCITY_THRESHOLD = cfg.thresh.vel
POSITION_THRESHOLD = cfg.thresh.pos

SP_ROT_0 = cfg.pos.spacer_rot.zero
# TODO SP_ROT_MAX = 4095*2 + SP_ROT_0
SP_ROT_GAIN = cfg.cont_gain.spacer_rot.pos_cont

SP_LIN_BACK = cfg.pos.spacer_lin.back
SP_LIN_FRONT = SP_LIN_BACK - cfg.pos.spacer_lin.range


def init_dynamixels():

    dxl = Dynamixel(port=PORT, baudrate=BAUDRATE)
    time.sleep(1)

    for dxl_id, name in [
        (GRIPPER_ID, "Gripper"),
        (SP_ROT_ID, "Spacer-Rot"),
        (SP_LIN_ID, "Spacer-Lin"),
    ]:
        print(f"[INIT] {name} ID={dxl_id}")

        try:
            dxl.disable_torque(dxl_id)
            time.sleep(0.1)
        except Exception as e:
            print(f"[WARN] disable_torque failed for {name}: {e}")

        dxl.set_mode_ex_position(dxl_id)
        time.sleep(0.1)

        print(f"{name} Position : {dxl.read_position(dxl_id)}")

    print("---------------------------------")
    print("   Dynamixel READY TO MOVE  ")
    print("---------------------------------")

    return dxl


def expand_sp_lin(dxl, kb):

    dxl.enable_torque(SP_LIN_ID)
    time.sleep(0.1)
    dxl.write_position(SP_LIN_ID, SP_LIN_FRONT)

    while True:

        if SP_LIN_FRONT - POSITION_THRESHOLD < dxl.read_position(SP_LIN_ID) < SP_LIN_FRONT + POSITION_THRESHOLD:
            print('spacer expanded')
            dxl.disable_torque(SP_LIN_ID)
            break

        elif kb.kbhit(): # emergency stop
            ch = kb.getch
            if ch in ('q', 'Q'):
                dxl.write_position(SP_LIN_ID, dxl.read_position(SP_LIN_ID))
                time.sleep(0.5)
                dxl.disable_torque(SP_LIN_ID)
                break

def rot_servo(dxl, kb):

    dxl.enable_torque(SP_ROT_ID)
    time.sleep(0.5)
    curr_des_pos = dxl.read_position(SP_ROT_ID)
    last_sent_pos = curr_des_pos
    print('spacer_rot activated')
    while True:

        if kb.kbhit():
            ch = kb.getch()
            if ch == 'K': # L arrow
                curr_des_pos -= SP_ROT_GAIN  # cw
            elif ch == 'M': # R arrow
                curr_des_pos +=  SP_ROT_GAIN # ccw
            elif ch in ('r', 'R'):  # r key next sequence
                print('rot_servo : done')
                break
        
        # TODO
        # curr_des_pos = max(~, min(rurr_des_pos, ~)) # range limitation
        
        if curr_des_pos != last_sent_pos:
            dxl.write_position(SP_ROT_ID, curr_des_pos)  # set sp_rot position

def reset_rot(dxl):

    dxl.write_position(SP_ROT_ID, SP_ROT_0)

    while True:

        if SP_ROT_0 - POSITION_THRESHOLD < dxl.read_position(SP_ROT_ID) < SP_ROT_0 + POSITION_THRESHOLD:
            dxl.disable_torque(SP_ROT_ID)
            print('reset_rot : done')
            break

def contract_sp_lin(dxl, kb):

    dxl.enable_torque(SP_LIN_ID)
    time.sleep(0.5)
    dxl.write_position(SP_LIN_ID, SP_LIN_BACK)

    while True:
        dxl.read_position(SP_LIN_ID)
        time.sleep(0.05)
        if SP_LIN_BACK - POSITION_THRESHOLD < dxl.read_position(SP_LIN_ID) < SP_LIN_BACK + POSITION_THRESHOLD:
            print('spacer contracted')
            dxl.disable_torque(SP_LIN_ID)
            break

        elif kb.kbhit():
            ch = kb.getch()
            if ch in ('q', 'Q'):
                dxl.write_position(SP_LIN_ID, dxl.read_position(SP_LIN_ID))
                time.sleep(0.5)
                dxl.disable_torque(SP_LIN_ID)
                break

def ungrasp(dxl, kb):

    dxl.enable_torque(GRIPPER_ID)
    time.sleep(0.5)
    gripper_curr_pos = dxl.read_position(GRIPPER_ID)
    gripper_ungrasp_pos = gripper_curr_pos + RELEASE_GAIN

    while True:
        if kb.kbhit():
            ch = kb.getch()
            if ch in('u', 'U'):
                dxl.write_position(GRIPPER_ID, gripper_ungrasp_pos)

        if gripper_ungrasp_pos - POSITION_THRESHOLD < dxl.read_position(GRIPPER_ID) < gripper_ungrasp_pos + POSITION_THRESHOLD:
            print('ungrasp : done')
            dxl.disable_torque(GRIPPER_ID)
            break

if __name__ == '__main__':

    try:
        dxl = init_dynamixels()
        kb = kbhit.KBHit()

        expand_sp_lin(dxl, kb)
        time.sleep(1)
        rot_servo(dxl, kb)
        reset_rot(dxl)
        contract_sp_lin(dxl, kb)
        ungrasp(dxl, kb)

        time.sleep(2)
        dxl.disable_torque(GRIPPER_ID)
        dxl.disable_torque(SP_ROT_ID)
        dxl.disable_torque(SP_LIN_ID)
        time.sleep(1)
        dxl.close_port()
        kb.set_normal_term()

        print('dynamixel deactivated')

    except KeyboardInterrupt:

        dxl.disable_torque(GRIPPER_ID)
        dxl.disable_torque(SP_ROT_ID)
        dxl.disable_torque(SP_LIN_ID)
        dxl.close_port()
        kb.set_normal_term()

