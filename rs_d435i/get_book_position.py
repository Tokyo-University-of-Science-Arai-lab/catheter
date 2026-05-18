# get book position in camera coordinate from pixel position

import pyrealsense2 as rs

import numpy as np
import time
import cv2
import math
from pathlib import Path
from datetime import datetime
import csv


class GetBookSpinePosition:
    
    def __init__(self):
        
        self.conf = rs.config()
        self.conf.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6) # bgr8 → mjpeg
        self.conf.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6) # z16 ってなんだ？

        self.pipe = rs.pipeline()
        self.prof = self.pipe.start(self.conf)
        time.sleep(1) # カメラの起動待ち
        self.align = rs.align(rs.stream.color)
        
        self.intr = rs.video_stream_profile(self.prof.get_stream(rs.stream.color)).get_intrinsics()
        
        # pyrealsense内のクラスとして出力される画像を持つひとたち
        self.depth_frame = None
        self.filled_d_frame = None
        self.color_frame = None
        
        # pyrealsense2 へのピクセル位置入力は(列, 行)で行うっぽい
        self.uv_left = [None, int(230)] # 上下反転を反映した，書架内左側の点 # 836 536
        self.uv_right = [None, int(230)] # 高さ固定，230pixel をとっている
        self.uv_top = [None, int(400)]
        self.uv_bottom = [None, int(120)]

        self.c_p_Z_right = None
        self.c_p_Z_left = None
        self.c_p_Z_top = None
        self.c_p_Z_bottom = None


        # とりあえず，TCP 位置の画像座標における高さ方向が230pixel 前後であることはわかったわ 目測360mm くらい
        self.c_p_left = [None, None, None]
        self.c_p_right = [None, None, None]
        self.c_p_top = [None, None, None]
        self.c_p_bottom = [None, None, None]
        
        self.R_cam_to_robot = Rz(-90.0) @ Rx(-100.0)
        # self.R_cam_to_robot = Rx(-10.0)
        self.r_p_top = [None, None, None] # 姿勢のみ反映，位置は未反映
        self.r_p_bottom = [None, None, None]
        
        self.save_root = Path('captures')
        self.outdif = None


    def __del__(self):

        self.pipe.stop()
        
        
    def depth_filter(self, raw_depth_frame):
        
        # rs-viewer と同じフィルタ処理
        # decimation filter : 解像度を落とす処理があり，使用するとRGBフレームと解像度が合わずエラーが出る
        decim = rs.decimation_filter()
        decim.set_option(rs.option.filter_magnitude, 2.0)
        
        # spatial filter
        spat = rs.spatial_filter()
        spat.set_option(rs.option.filter_magnitude, 2.0)
        spat.set_option(rs.option.filter_smooth_alpha, 0.5)
        spat.set_option(rs.option.filter_smooth_delta, 20.0)
        
        # hole_filling はデフォルトではオフ？
        hole_fill = rs.hole_filling_filter()
        
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        
        # filterd = decim.process(raw_depth_frame)
        filterd = depth_to_disparity.process(raw_depth_frame)
        filterd = spat.process(filterd)
        filterd = disparity_to_depth.process(filterd)
        filterd = hole_fill.process(filterd)

        
        return filterd.as_depth_frame()
        
        
    def get_frames(self):
        
        frames = self.pipe.wait_for_frames()
        # time.sleep(10) これしてもRGBがめっちゃ緑になる
        align_frames = self.align.process(frames)
        depth_frame = align_frames.get_depth_frame()
        self.depth_frame = self.depth_filter(depth_frame)
    
        self.color_frame = align_frames.get_color_frame()
        # time.sleep(0.1)
        
        # save frames
        # ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f") mili sec 
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.outdir = (self.save_root / ts)
        self.outdir.mkdir(parents=True, exist_ok=True)

        # 画像取得（numpy化）
        color_np = np.asanyarray(self.color_frame.get_data())      # BGR
        depth_np_u16 = np.asanyarray(self.depth_frame.get_data())  # uint16 (Z16)

        # 保存：RGB (png)
        cv2.imwrite(str(self.outdir / "rgb.png"), color_np)

        # 保存：Depth (npy: raw uint16)
        np.save(self.outdir / "depth.npy", depth_np_u16)

        # 保存：Depth (csv: meter換算 float32)
        depth_m = depth_np_u16.astype(np.float32)
        np.savetxt(self.outdir / "depth.csv", depth_m, delimiter=",", fmt="%.6f")

        
    def deproject(self): # input u, v, Z, f 画像座標→カメラ座標 の変換
        
        # # 生のdepth を使用しているので抜けが発生して返り値が0になる場合がある(左は9回中5回，右は9回中9回抜けた)
        self.c_p_Z_right = self.depth_frame.get_distance(self.uv_right[0], self.uv_right[1])
        self.c_p_Z_left = self.depth_frame.get_distance(self.uv_left[0], self.uv_left[1])
        self.c_p_Z_top = self.depth_frame.get_distance(self.uv_top[0], self.uv_top[1])
        self.c_p_Z_bottom = self.depth_frame.get_distance(self.uv_bottom[0], self.uv_bottom[1])

        # logging
        msg = (
            f"before thresh check depth "
            f"r {self.c_p_Z_right} "
            f"l {self.c_p_Z_left} "
            f"t {self.c_p_Z_top} "
            f"b {self.c_p_Z_bottom}"
        )

        # 画面にも出力
        print(msg)

        # CSV ファイルに保存（追記）
        csv_path = self.outdir / "depth_info.csv"
        write_header = not csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)

            if write_header:
                writer.writerow(["r", "l", "t", "b"])  # ヘッダー行

            writer.writerow([
                self.c_p_Z_right,
                self.c_p_Z_left,
                self.c_p_Z_top,
                self.c_p_Z_bottom
            ])

        # depth 飛び対策
        #####################################################
        if 0.32 > self.c_p_Z_right or self.c_p_Z_right > 0.41:
            self.c_p_Z_right = 0.36
        if 0.32 > self.c_p_Z_left or self.c_p_Z_left > 0.41:
            self.c_p_Z_left = 0.36
        if 0.32 > self.c_p_Z_top or self.c_p_Z_top > 0.41:
            self.c_p_Z_top = 0.36
        if 0.32 > self.c_p_Z_bottom or self.c_p_Z_bottom > 0.41:
            self.c_p_Z_bottom = 0.376
        #####################################################
                
        self.c_p_right = rs.rs2_deproject_pixel_to_point(self.intr , self.uv_right, self.c_p_Z_right)
        self.c_p_left = rs.rs2_deproject_pixel_to_point(self.intr , self.uv_left, self.c_p_Z_left)
        self.c_p_top = rs.rs2_deproject_pixel_to_point(self.intr , self.uv_top, self.c_p_Z_top)
        self.c_p_bottom = rs.rs2_deproject_pixel_to_point(self.intr , self.uv_bottom, self.c_p_Z_bottom)
        # print('c_p_right = ', self.c_p_right)
        
        
    def get_book_width(self):
        # カメラ座標系における書籍幅 本来距離を求めるべきだが，
        # 現状URの並進量算出にもx位置のみで実行→幅もx位置のみを使用して算出
        book_width = abs(self.c_p_right[0] - self.c_p_left[0]) # m
        
        return book_width
        
        
    def get_dy(self): # カメラ座標系におけるx方向位置
        
        return self.c_p_right[0]
         
        
    def get_d_roll(self):
        # カメラ→ロボット座標へ変換
        c_p_top_np    = np.asarray(self.c_p_top,    dtype=float)
        c_p_bottom_np = np.asarray(self.c_p_bottom, dtype=float)

        self.r_p_top    = (self.R_cam_to_robot @ c_p_top_np).tolist()
        self.r_p_bottom = (self.R_cam_to_robot @ c_p_bottom_np).tolist()

        print('r_p_top    = ', self.r_p_top)
        print('r_p_bottom = ', self.r_p_bottom)

        # 方向ベクトル（上端→下端。向きを変えると符号が反転するだけ）
        v  = np.asarray(self.r_p_top, dtype=float) - np.asarray(self.r_p_bottom, dtype=float)
        vy, vz = v[1], v[2]

        # yz 平面での退化チェック
        if float(np.hypot(vy, vz)) < 1e-9:
            self.d_roll_deg = None
            print("warn: yz平面成分が小さすぎるため角度を計算できません。")
            return None

        # 左手系の軸向き・回転は右ねじ：基準を z− とする → atan2(vy, -vz)
        theta_rad = np.arctan2(vy, -vz)
        theta_deg = float(np.degrees(theta_rad))
        print(theta_rad)

        # [-180, 180) に正規化
        if theta_deg >= 180.0:
            theta_deg -= 360.0
        elif theta_deg < -180.0:
            theta_deg += 360.0

        self.d_roll_deg = theta_deg
        print(f'd_roll_deg (LH axes, RH rotation about x, z−=0°) = {theta_deg:.3f} deg')
        return -1*theta_rad        
    
    
    def main(self):
        
        self.get_frames()
        self.deproject()
        self.get_d_roll()
        
def Rx(deg):
    rad = math.radians(deg)
    Rx = np.array([[1, 0, 0],
                    [0, math.cos(rad), -math.sin(rad)],
                    [0, math.sin(rad), math.cos(rad)]])
    return Rx

def Rz(deg):
    rad = math.radians(deg)
    Rz = np.array([[math.cos(rad), -math.sin(rad), 0],
                    [math.sin(rad), math.cos(rad), 0],
                    [0, 0, 1]])
    return Rz

        
if __name__ == '__main__':
    
    DT = GetBookSpinePosition()
    DT.main()