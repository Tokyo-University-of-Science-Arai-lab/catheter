#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from pyzbar.pyzbar import decode
import pyrealsense2 as rs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--serial", type=str, default="", help="RealSense serial (optional)")
    ap.add_argument("--timeout-ms", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=30, help="number of frames to skip")
    ap.add_argument("--out", type=str, default="./rs_test_capture.png")
    ap.add_argument("--show", action="store_true", help="show window (requires GUI)")
    ap.add_argument("--decode", action="store_true", help="try barcode decode")
    return ap.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)

    pipe = rs.pipeline()
    cfg = rs.config()

    if args.serial.strip():
        cfg.enable_device(args.serial.strip())

    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    try:
        profile = pipe.start(cfg)
    except Exception as e:
        print("[ERROR] pipeline.start failed:", e)
        return 2

    try:
        # intrinsics 表示
        vsp = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = vsp.get_intrinsics()
        print(f"[INFO] intrinsics fx={intr.fx:.3f} fy={intr.fy:.3f} cx={intr.ppx:.3f} cy={intr.ppy:.3f}")
        print(f"[INFO] stream {args.width}x{args.height}@{args.fps} format=bgr8")

        # warmup
        for _ in range(max(0, args.warmup)):
            pipe.wait_for_frames()

        frames = pipe.wait_for_frames(args.timeout_ms)
        color = frames.get_color_frame()
        if not color:
            print("[ERROR] get_color_frame() returned None (no color frame)")
            return 3

        bgr = np.asanyarray(color.get_data())  # (H,W,3) uint8 BGR
        if bgr is None or bgr.size == 0:
            print("[ERROR] captured image is empty")
            return 4

        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), bgr)
        print(f"[INFO] saved: {out_path} (imwrite={ok})  shape={bgr.shape} dtype={bgr.dtype}")

        if args.decode:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            blur = cv2.GaussianBlur(contrast, (3, 3), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            decoded = decode(binary)
            print(f"[INFO] decode count: {len(decoded)}")
            for i, d in enumerate(decoded):
                s = d.data.decode("utf-8", errors="ignore")
                r = d.rect
                print(f"  [{i}] code='{s}' bbox=(left={r.left}, top={r.top}, w={r.width}, h={r.height})")

        if args.show:
            cv2.imshow("realsense_color", bgr)
            print("[INFO] press any key on the image window to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0

    except rs.error as e:
        print("[ERROR] RealSense rs.error:", e)
        return 5
    except Exception as e:
        print("[ERROR] unexpected error:", e)
        return 6
    finally:
        try:
            pipe.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
