import ur5e.control.rtde_ur5e as UR5e
from rs_d435i.get_book_position import GetBookSpinePosition
import Dynamixel_win_pro_hand_book.HandBook_Retrieval as HandBook

import time

# initial TCP_pose when 1st perception
# [-0.367414913194101, -0.11505935911584259, 0.3268820798200269, -0.00033660564673279066, -1.571207211242841, 0.0009980046969578957]

# x 方向，ズレの校正用 [mm]
# DX_ADJ =  -3.424 # 0824実験からの値
# DX_ADJ = -5.5
DY_ADJ = -4.5

def main():
    try:
        # initialize modules
        print('start sequence')
        PosGetter = GetBookSpinePosition()
        PosGetter.uv_right, PosGetter.uv_left = [None, 230], [None, 230]
        HandMotors = HandBook.init_dynamixels() # TODO classにする
        UR5e.moveL_to_init_pose()
        time.sleep(1)
        
        # sequence
        PosGetter.get_frames()
        PosGetter.uv_right[0] = int(input('u_right ? : ')) # input u_right value by checking RGB frame
        PosGetter.uv_left[0] = int(input('uv_left ? : ')) # input u_left value by checking RGB frame
        PosGetter.uv_top[0] = int(input('uv_top ? : '))
        PosGetter.uv_bottom[0] = int(input('uv_bottom ? : '))
        
        PosGetter.deproject()
        
        print('check depth', PosGetter.c_p_right[2], 
              PosGetter.c_p_left[2],
              PosGetter.c_p_top[2],
              PosGetter.c_p_bottom[2]) # depth の値チェック

        time.sleep(2)
        # culculate gripper width
        book_width = PosGetter.get_book_width() * 1000 # m to mm
        print('book_width : ',book_width)
        print('gripper open')
        HandBook.open_until_width(HandMotors, book_width)
        # culculate UR5e next_pose
        dy = PosGetter.get_dy() + 0.001* -1* DY_ADJ
        
        print('UR5e will move')
        
        # ここの next_pose_diff にrollを埋め込めれば良い
        next_pose_diff = UR5e.culc_pose_diff_from_dy(dy)
        d_roll = PosGetter.get_d_roll()

        UR5e.moveL_diff(next_pose_diff)
        UR5e.moveL_rot_diff(d_roll)
        UR5e.moveL_to_2nd_pos()
        try: # 明らかに挿入不可の位置に来たら，ctrl+c でスキップ
            input('insert:enter, avoid:ctrl+c') # enterキーで挿入，ctrl+c でスキップ
            UR5e.moveL_to_insert()
            time.sleep(1)
            HandBook.grasp(HandMotors)
        except KeyboardInterrupt:
            print('skip insertion')
            HandBook.grasp(HandMotors)
            HandMotors.disable_torque(HandBook.GRIPPER_ID)
        
        time.sleep(2)
        UR5e.moveL_post_grasp(velocity=0.05)
        UR5e.moveL_to_init_pose()
        print('sequence done')
        HandMotors.close_port()
        # PosGetter.pipe.stop()
    
    except KeyboardInterrupt:
        HandMotors.disable_torque(HandBook.GRIPPER_ID)
        HandMotors.close_port()

if __name__ == '__main__':
    
    main()
    