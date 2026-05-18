#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pyrealsense2 as rs

def main():
    # ===== 1) パイプラインと設定 =====
    pipe = rs.pipeline()
    cfg = rs.config()

    width = 1280
    height = 720
    fps = 6

    # あなたの既存コードと同じ設定：
    # カラーとデプスを 1280x720, 6fps で有効化
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # depth を color に揃える align も作っておく
    align = rs.align(rs.stream.color)

    # ===== 2) ストリーム開始 =====
    profile = pipe.start(cfg)

    # 自動露出などが安定するまで数フレーム捨てる
    for _ in range(10):
        pipe.wait_for_frames()

    # 1フレーム取得
    frames = pipe.wait_for_frames()
    aligned = align.process(frames)

    depth_frame = aligned.get_depth_frame()
    if not depth_frame:
        raise RuntimeError("depth_frame が取得できませんでした")

    # ===== 3) depth のプロファイルから intrinsics を取得 =====
    dprof = rs.video_stream_profile(depth_frame.get_profile())
    intr = dprof.get_intrinsics()

    # depth_scale も取得（Z16 → [m] に変換するときに使うやつ）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    pipe.stop()

    # ===== 4) 結果表示 =====
    print("=== RealSense Intrinsics (aligned depth) ===")
    print(f"width  = {intr.width}")
    print(f"height = {intr.height}")
    print(f"fx     = {intr.fx}")
    print(f"fy     = {intr.fy}")
    print(f"ppx    = {intr.ppx}")
    print(f"ppy    = {intr.ppy}")
    print(f"depth_scale = {depth_scale}  # [m / depth_count]")

if __name__ == "__main__":
    main()