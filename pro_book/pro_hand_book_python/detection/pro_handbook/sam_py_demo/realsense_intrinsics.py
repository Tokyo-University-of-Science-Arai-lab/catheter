#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RealSense → 逆投影に必要な定数だけを1ファイルJSONに保存
保存するキー: fx, fy, cx, cy, depth_scale  （必要最小限）内部パラメータを読み込んでるだけやね！！
※マスクがRGB基準なら aligned_depth_to_color の intrinsics を使うのが安全
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="calib/camera_calib.json", help="保存先 JSON")
    ap.add_argument("--w", type=int, default=1280, help="ストリーム幅")
    ap.add_argument("--h", type=int, default=720,  help="ストリーム高さ")
    ap.add_argument("--fps", type=int, default=6,  help="フレームレート")
    ap.add_argument("--serial", type=str, default=None, help="シリアル指定（任意）")
    ap.add_argument("--warmup", type=int, default=5, help="ウォームアップ枚数")
    ap.add_argument("--include-size", action="store_true",
                    help="幅高さ（width,height）もオプションで保存")
    args = ap.parse_args()

    # 遅延 import（pyrealsense2 未インストール環境でも読み込み時エラー回避）
    import pyrealsense2 as rs

    cfg = rs.config()
    if args.serial:
        cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, args.w, args.h, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.w, args.h, rs.format.z16, args.fps)

    pipe = rs.pipeline()
    try:
        prof = pipe.start(cfg)
        align = rs.align(rs.stream.color)

        # depth scale（raw Z16 → [m] の係数）
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

        # ウォームアップ
        for _ in range(max(0, args.warmup)):
            pipe.wait_for_frames()

        # 1フレーム取得して depth を color に整列
        frames = pipe.wait_for_frames()
        aligned = align.process(frames)
        depth_aligned = aligned.get_depth_frame()

        # 整列後Depthの intrinsics（←これを逆投影に使う）
        dprof = rs.video_stream_profile(depth_aligned.get_profile())
        di = dprof.get_intrinsics()  # fx, fy, ppx(=cx), ppy(=cy), width, height

        payload = {
            "fx": float(di.fx),
            "fy": float(di.fy),
            "cx": float(di.ppx),
            "cy": float(di.ppy),
            "depth_scale": float(depth_scale),
            # 参考: この値は aligned_depth_to_color（カラー座標系のDepth）
            "mode": "aligned_depth_to_color",
        }
        if args.include_size:
            payload["width"]  = int(di.width)
            payload["height"] = int(di.height)

        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf_8")

        print("✔ saved:", out.resolve())
        print(f"  fx={di.fx:.3f}, fy={di.fy:.3f}, cx={di.ppx:.3f}, cy={di.ppy:.3f}")
        if args.include_size:
            print(f"  size=({di.width}x{di.height})")
        print("  depth_scale =", depth_scale)
        print("  mode =", payload["mode"])

    finally:
        try:
            pipe.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()
