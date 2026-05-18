from .dynamixel_cross_platform import Dynamixel
from .util.cfg_dict_loader import DynamixelCfg
from . import kbhit_lin as kbhit
import time
import math
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

PKG_DIR = Path(__file__).resolve().parent
cfg_path = PKG_DIR / "config" / "Dynamixel_config.yaml"

cfg = DynamixelCfg(str(cfg_path))

PORT = cfg.id.port
BAUDRATE = cfg.id.baudrate

GRIPPER_ID = cfg.id.gripper
SP_ROT_ID = cfg.id.spacer_rot
SP_LIN_ID = cfg.id.spacer_lin

GRIPPER_BL = cfg.pos.gripper.backlash # backlash
GRIPPER_CLOSE = cfg.pos.gripper.close
GRIPPER_FULL_OPEN = cfg.pos.gripper.range + GRIPPER_CLOSE + GRIPPER_BL
GRIPPER_ROT_GAIN = cfg.cont_gain.gripper.pos_cont
GRIPPER_THETA_0 = cfg.pos.gripper.theta_zero
GRIPPER_R2S = cfg.pos.gripper.rad_to_step
GRIPPER_GR = cfg.pos.gripper.gear_ratio
GRIPPER_CALIB_A = cfg.pos.gripper.calib_width_a
GRIPPER_CALIB_B = cfg.pos.gripper.calib_width_b

VELOCITY_THRESHOLD = cfg.thresh.vel
POSITION_THRESHOLD = cfg.thresh.pos



def init_dynamixels():
    
    dxl = Dynamixel(port=PORT, baudrate=BAUDRATE)
    dxl.set_mode_ex_position(GRIPPER_ID)
    print(f'Gripper Position : {dxl.read_position(GRIPPER_ID)}')
    time.sleep(0.2)
    
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
            
        curr_des_pos = max(GRIPPER_CLOSE, min(curr_des_pos, GRIPPER_FULL_OPEN)) # range limitation
        
        if curr_des_pos != last_sent_pos:
            dxl.write_position(GRIPPER_ID, curr_des_pos)  # set gripper position
            
def calib_width(w_real):
    return GRIPPER_CALIB_A*w_real + GRIPPER_CALIB_B
    
def open_until_width(dxl, width, gravity=False):

    width = calib_width(width)

    dxl.enable_torque(GRIPPER_ID)

    sinx = (width + 20) / 80.0
    sinx = max(-1.0, min(1.0, sinx))

    d_theta = GRIPPER_GR * (math.asin(sinx) - GRIPPER_THETA_0)
    d_step  = int(d_theta * GRIPPER_R2S + GRIPPER_BL)

    # ★絶対開口に統一
    des_pos = GRIPPER_CLOSE + d_step

    if gravity:
        des_pos = des_pos - GRIPPER_BL

    des_pos = max(GRIPPER_CLOSE, min(des_pos, GRIPPER_FULL_OPEN))

    dxl.write_position(GRIPPER_ID, des_pos)
    

def grasp(dxl, timeout_sec=3.0):

    dxl.enable_torque(GRIPPER_ID)
    print("grasping object until detection")

    try:
        dxl.write_position(GRIPPER_ID, GRIPPER_CLOSE - GRIPPER_BL-500)
        print("grasping")
    except Exception as e:
        raise RuntimeError(f"Dynamixel write failed: {e}")

    time.sleep(0.05)

    timeout = time.time() + timeout_sec

    while True:
        # ----- タイムアウト -----
        if time.time() > timeout:
            raise RuntimeError("Grasp timeout (Dynamixel not responding)")

        # ----- 通信エラー検知 -----
        try:
            curr_vel = dxl.read_velocity(GRIPPER_ID)
            curr_pos = dxl.read_position(GRIPPER_ID)
        except Exception as e:
            raise RuntimeError(f"Dynamixel read failed: {e}")

        # ----- 把持判定 -----
        if abs(curr_vel) < VELOCITY_THRESHOLD:
            print('grasp detected')
            dxl.disable_torque(GRIPPER_ID)
            break

        elif GRIPPER_CLOSE - POSITION_THRESHOLD < curr_pos < GRIPPER_CLOSE + POSITION_THRESHOLD:
            print('gripper close')
            dxl.disable_torque(GRIPPER_ID)
            break


def open_until_full(dxl, asynchronous=False):

    print("gripper open until full")
    dxl.enable_torque(GRIPPER_ID)
    dxl.write_position(GRIPPER_ID, GRIPPER_FULL_OPEN)
    time.sleep(0.05)

    if asynchronous:
        return

    timeout = time.time() + 3.0  # 3秒タイムアウト

    while True:
        if time.time() > timeout:
            raise RuntimeError("Dynamixel timeout during open_until_full")

        try:
            curr_vel = dxl.read_velocity(GRIPPER_ID)
            curr_pos = dxl.read_position(GRIPPER_ID)
        except Exception as e:
            raise RuntimeError(f"Dynamixel communication lost: {e}")

        if curr_vel is None or curr_pos is None:
            raise RuntimeError("Dynamixel returned None")

        if abs(curr_vel) < VELOCITY_THRESHOLD:
            print("gripper full open")
            break

        if GRIPPER_FULL_OPEN - POSITION_THRESHOLD < curr_pos < GRIPPER_FULL_OPEN + POSITION_THRESHOLD:
            print("gripper full open")
            break

    dxl.disable_torque(GRIPPER_ID)

def calib_width_collect(dxl):

    print("\n====== GRIPPER CALIBRATION (絶対開口版) ======")
    print("指定mm入力 → 常に閉位置基準で開く")
    print("終了は q")
    print("=============================================\n")

    w_real_list = []
    w_hat_list  = []

    while True:

        cmd = input("開きたい幅(mm) or q: ")

        if cmd.lower() == 'q':
            break

        try:
            target_width = float(cmd)
        except:
            print("数値入力してください")
            continue

        dxl.enable_torque(GRIPPER_ID)

        # ===== 正しい逆算 =====
        sinx = (target_width + 20) / 80.0
        sinx = max(-1.0, min(1.0, sinx))  # 安全化

        d_theta = GRIPPER_GR * (math.asin(sinx) - GRIPPER_THETA_0)
        d_step  = int(d_theta * GRIPPER_R2S + GRIPPER_BL)

        # 絶対開口
        des_pos = GRIPPER_CLOSE + d_step
        des_pos = max(GRIPPER_CLOSE, min(des_pos, GRIPPER_FULL_OPEN))

        dxl.write_position(GRIPPER_ID, des_pos)

        time.sleep(1.0)

        measured = input("実測値(mm): ")

        try:
            w_real = float(measured)
        except:
            print("スキップ")
            continue

        curr_pos = dxl.read_position(GRIPPER_ID)

        # ===== 順方向（理論幅） =====
        d_theta_now = (curr_pos - GRIPPER_CLOSE - GRIPPER_BL) / GRIPPER_R2S
        w_hat = 80 * math.sin(d_theta_now / GRIPPER_GR + GRIPPER_THETA_0) - 20

        print(f"記録 -> w_real={w_real:.3f}, w_hat={w_hat:.3f}")

        w_real_list.append(w_real)
        w_hat_list.append(w_hat)

    if len(w_real_list) < 2:
        print("データ不足")
        return

    A, B = np.polyfit(w_real_list, w_hat_list, 1)

    print("\n========= RESULT =========")
    print(f"GRIPPER_CALIB_A = {A}")
    print(f"GRIPPER_CALIB_B = {B}")
    print("==========================")

    # ===== 可視化 =====
    w_real_arr = np.array(w_real_list, dtype=float)
    w_hat_arr  = np.array(w_hat_list, dtype=float)

    x_line = np.linspace(w_real_arr.min(), w_real_arr.max(), 200)
    y_line = A * x_line + B

    plt.figure()
    plt.scatter(w_real_arr, w_hat_arr, label="measured points")
    plt.plot(x_line, y_line, label="fit: w_hat = A*w_real + B")
    plt.xlabel("w_real (mm)")
    plt.ylabel("w_hat (mm)")
    plt.title("Gripper calibration")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    residual = w_hat_arr - (A * w_real_arr + B)

    plt.figure()
    plt.scatter(w_real_arr, residual, label="residual")
    plt.axhline(0.0)
    plt.xlabel("w_real (mm)")
    plt.ylabel("residual (mm)")
    plt.title("Calibration residual")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    return A, B

if __name__ == '__main__':
    try:
        
        dxl = init_dynamixels()

        print("1: 通常テスト")
        print("2: キャリブレーション")
        mode = input("mode: ")

        if mode == "2":
            calib_width_collect(dxl)
            grasp(dxl)
        else:
            open_until_width(dxl, 25)
            input()
            grasp(dxl)

        dxl.disable_torque(GRIPPER_ID)
        dxl.close_port()
        
    except KeyboardInterrupt:
        grasp(dxl)
        dxl.disable_torque(GRIPPER_ID)
        dxl.close_port()




