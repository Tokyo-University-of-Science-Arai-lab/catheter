import os, glob
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

def find_video_devices_by_name(keyword="Depstech") -> list[str]:
    key = keyword.lower()
    devs = []
    for v in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_path = os.path.join(v, "name")
        try:
            name = open(name_path, "r").read().strip()
        except Exception:
            continue
        if key in name.lower():
            devs.append("/dev/" + os.path.basename(v))
    if not devs:
        raise RuntimeError(f'keyword "{keyword}" を含む video デバイスが見つかりません')
    return devs

def find_first_openable_video_device(keyword="Depstech") -> str:
    """
    keyword を含む /dev/video* を列挙して、
    実際に OpenCV で open できた最初のデバイスパスだけ返す。
    """
    for dev in find_video_devices_by_name(keyword):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.release()
            return dev
        cap.release()
    raise RuntimeError(f"{keyword} は見つかったが、どの /dev/video* も open できませんでした")


def capture_one_depstech(save_path: Path, width=3840, height=2160):
    dev = find_first_openable_video_device("Depstech")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {dev}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path_ts = str(save_path.with_name(f"{save_path.stem}_{ts}{save_path.suffix}"))
    # 安定化のため少し捨てる
    print("capturing...")
    for _ in range(80):
        cap.read()

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"capture failed: {dev}")
    print("captured")

    cv2.imwrite(str(save_path_ts), frame)
    return frame

if __name__ == "__main__":
    save_path = "/home/book/Pictures/barcode_capture.png"
    frame, dev = capture_one_depstech(save_path)
    print(f"Captured image from {dev}, shape={frame.shape}")