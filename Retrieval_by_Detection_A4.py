import ur5e.control.rtde_ur5e as UR5e
from rs_d435i.get_book_position import GetBookSpinePosition
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook
from pathlib import Path
from detection.pro_handbook.sam_py_demo.sam_infer_module import SamBatchInfer, SamConfig
import time
import cv2
import numpy as np

# initial TCP_pose when 1st perception
# [-0.367414913194101, -0.11505935911584259, 0.3268820798200269, -0.00033660564673279066, -1.571207211242841, 0.0009980046969578957]

# x 方向，ズレの校正用 [mm]
# DX_ADJ =  -3.424 # 0824実験からの値
DY_ADJ = -4.5
ROT_THRESH = np.deg2rad(2.5) 

def main():
    try:
        # initialize modules
        print('start sequence')
        PosGetter = GetBookSpinePosition()
        PosGetter.uv_right, PosGetter.uv_left = [None, 230], [None, 230]
        PosGetter.uv_top, PosGetter.uv_bottom = [None, 400], [None, 120]
        HandMotors = HandBook.init_dynamixels() # TODO classにする
        UR5e.moveJ_to_init_Q(joints_deg = UR5e.INIT_Q_A4)
        UR5e.moveL_to_init_pose(pose = UR5e.INIT_TCP_POSE_A4)
        time.sleep(0.2)
        BookDetector = SamBatchInfer(SamConfig(
            encoder_path="models/sam_vit_h_4b8939.encoder.onnx",
            decoder_path="models/sam_vit_h_4b8939.decoder.onnx",
            device="gpu",        # cpuなら "cpu"
            pts_side=30,
            min_area=500,
            iou_thr=0.8,
        ))
        # sequence
        PosGetter.get_frames()
        ####### 可視化したい　######################
        rgb_frame = np.asanyarray(PosGetter.color_frame.get_data())
        #print(type(rgb_frame))
        #cv2.imshow('rgb', rgb_frame)
        #cv2.waitKey(1)
        ########################################
        # TODO
        ## こうせいがやること、rgb_frame を入力として、認識システムによる、書籍両端の検出と、uv_left, uv_right の決定
        # rgb_frame = np.asanyarray(PosGetter.color_frame.get_data())
        # cv2.imshow('rgb', rgb_frame); cv2.waitKey(1)  # 任意

        # ★ 推論：書籍の左右端を検出（所持上で左右逆にしたいなら swap_lr_output=True）
        res = BookDetector.run_on_rgb_frame(
            rgb_frame,
            stage_save_dir=Path("./captures"),
            swap_lr_output=True,           # 必要なら
            stem_for_save="live",
        )

        PosGetter.uv_right[0] = res["uv_right"] # input u_right value by checking RGB frame
        PosGetter.uv_left[0] =  res["uv_left"]# input u_left value by checking RGB frame
        PosGetter.uv_top[0] = res.get("bottom")
        PosGetter.uv_bottom[0] = res.get("top")

        #####################################################
        print('uv_right, uv_left, uv_top, uv_bottom : ', PosGetter.uv_right, PosGetter.uv_left, PosGetter.uv_top, PosGetter.uv_bottom)
                
        PosGetter.deproject()

        print('check depth', PosGetter.c_p_right[2], 
              PosGetter.c_p_left[2],
              PosGetter.c_p_top[2],
              PosGetter.c_p_bottom[2]) # depth の値チェック
        
        print('check depth 2', PosGetter.c_p_right[2], 
              PosGetter.c_p_left[2],
              PosGetter.c_p_top[2],
              PosGetter.c_p_bottom[2]) # depth の値チェック
        
        time.sleep(0.5)
        # culculate gripper width
        book_width = PosGetter.get_book_width() * 1000 # m to mm
        print('book_width : ',book_width)
        print('gripper open')
        HandBook.open_until_width(HandMotors, book_width)
        # culculate UR5e next_pose
        dy = PosGetter.get_dy() + 0.001* -1* DY_ADJ
        print('UR5e will move')
        next_pose_diff = UR5e.culc_pose_diff_from_dy(dy)
        d_roll = PosGetter.get_d_roll()
        # 回転が3deg 未満なら無視
        ###########################################################
        if abs(d_roll) < ROT_THRESH:
            d_roll = 0.0
        ###########################################################

        UR5e.moveL_diff(next_pose_diff)
        UR5e.moveL_to_2nd_pos()
        UR5e.moveL_rot_diff(d_roll)

        try: # 明らかに挿入不可の位置に来たら，ctrl+c でスキップ
            input('insert:enter, avoid:ctrl+c') # enterキーで挿入，ctrl+c でスキップ
            UR5e.moveL_to_insert()
            time.sleep(0.2)
            HandBook.grasp(HandMotors)
            UR5e.moveL_post_grasp(velocity=0.05)
            UR5e.moveL_to_init_pose(pose = UR5e.INIT_TCP_POSE_A4)
            UR5e.moveJ_to_init_Q(joints_deg = UR5e.INIT_Q_A4)
            HandMotors.close_port()


        except KeyboardInterrupt:
            print('skip insertion')
            HandBook.grasp(HandMotors)
            HandMotors.disable_torque(HandBook.GRIPPER_ID)
        
        time.sleep(0.2)
        UR5e.moveL_post_grasp(velocity=0.05)
        UR5e.moveL_to_init_pose(pose = UR5e.INIT_TCP_POSE_A4)
        print('sequence done')
        HandMotors.close_port()
        # PosGetter.pipe.stop()
    
    except KeyboardInterrupt:
        HandMotors.disable_torque(HandBook.GRIPPER_ID)
        HandMotors.close_port()

if __name__ == '__main__':
    
    main()
    