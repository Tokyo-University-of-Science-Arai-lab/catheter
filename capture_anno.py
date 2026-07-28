#!/usr/bin/env python3
"""
RealSense D435i でアノテーション用画像を1枚ずつ撮影するスクリプト。

Enter を押すたびに1枚撮影し、Ctrl+C で中断するまで無限に続く。

保存先: anno-catheter/<連番>/<年月日_時分秒>.png
連番フォルダは起動時に1つだけ作られ、中断するまでの撮影は
すべて同じフォルダに保存される。連番は既存の数字フォルダの
続きから自動で決まる（1 があれば 2、2 があれば 3、…）。
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── 設定 ────────────────────────────────────────────────────
SAVE_ROOT = Path("anno-catheter")
N_WARMUP  = 30    # 自動露出が安定するまで捨てるフレーム数
WIDTH     = 1280
HEIGHT    = 720
FPS       = 6
# ────────────────────────────────────────────────────────────


def next_start_index(save_root: Path) -> int:
    """既存の連番フォルダを調べ、次に使うべきインデックスを返す。"""
    if not save_root.exists():
        return 1
    existing = [
        int(p.name) for p in save_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    return max(existing) + 1 if existing else 1


def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    outdir = SAVE_ROOT / str(next_start_index(SAVE_ROOT))
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"保存先: {outdir.resolve()}")
    print("Enter で撮影、Ctrl+C で終了\n")

    # ── カメラ初期化 ────────────────────────────────────────
    conf = rs.config()
    conf.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    pipe = rs.pipeline()
    pipe.start(conf)

    print(f"カメラ起動中... ウォームアップ {N_WARMUP} フレーム")
    for _ in range(N_WARMUP):
        pipe.wait_for_frames()
    print("準備完了\n")

    saved = 0
    try:
        while True:
            input(f"[{saved + 1} 枚目] Enter で撮影... ")

            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("  フレーム取得失敗 → もう一度 Enter を押してください")
                continue

            color_np = np.asanyarray(color_frame.get_data())

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outpath = outdir / f"{stamp}.png"
            # 同一秒内の連続撮影でも上書きしないよう枝番を付ける
            n = 1
            while outpath.exists():
                outpath = outdir / f"{stamp}_{n}.png"
                n += 1
            cv2.imwrite(str(outpath), color_np)

            print(f"  保存 → {outpath}")
            saved += 1
    except KeyboardInterrupt:
        print(f"\n中断しました。")
    finally:
        pipe.stop()
        print(f"撮影完了: {saved} 枚保存")


if __name__ == "__main__":
    main()
