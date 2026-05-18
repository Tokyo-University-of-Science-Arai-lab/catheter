import pyrealsense2  as rs

import numpy as np
import time
import cv2
import math
from pathlib import Path
from datetime import datetime


class Deproject():
    
    def __init__(self, serial: str | None = None):
        self.conf = rs.config()

        if serial is not None:
            self.conf.enable_device(serial)   # ←これが「区別」

        #self.conf.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
        self.conf.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 6)
        self.pipe = rs.pipeline()
        self.prof = self.pipe.start(self.conf)
        time.sleep(1) # カメラの起動待ち
        self.align = rs.align(rs.stream.color)

        self.intr = rs.video_stream_profile(
            self.prof.get_stream(rs.stream.color)
        ).get_intrinsics()

        self.filled_d_frame = None
        self.color_frame = None
    
        
    
    def get_frames(self, shot_dir):
        
        frames = self.pipe.wait_for_frames()
        # time.sleep(10) これしてもRGBがめっちゃ緑になる
        align_frames = self.align.process(frames)
        self.color_frame = align_frames.get_color_frame()
        # time.sleep(0.1)
        
        # save frames
        outdir = shot_dir
        #ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        #outdir = (self.save_root / ts)
        #outdir.mkdir(parents=True, exist_ok=True)

        # 画像取得（numpy化）
        color_np = np.asanyarray(self.color_frame.get_data())      # BGR
        # 保存：RGB (png)
        cv2.imwrite(str(outdir / "rgb_under.png"), color_np)
        self.pipe.stop()
        return color_np