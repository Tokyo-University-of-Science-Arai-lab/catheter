#!/usr/bin/env python3
"""
RealSense D435i で 100 枚バースト撮影するスクリプト。
10 枚撮影するたびに一時停止し、Enter で次の 10 枚へ進む。

保存先: captures/100test/<連番>/
  after_init_rgb.png        : RGB画像 (BGR PNG)
  after_init_depth.npy      : 深度画像 (uint16, Z16)
  camera_params.json        : カメラ内部パラメータ

連番は既存の captures/100test/ の続きから自動開始する。
offline_pointcloud_debug.py が読むファイル名・構造と一致する。
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── 設定 ────────────────────────────────────────────────────
SAVE_ROOT    = Path("captures") / "100test"
N_TOTAL      = 100   # 撮影する総枚数
BATCH_SIZE   = 10    # 1バッチあたりの枚数（Enter待ちの単位）
N_WARMUP     = 30    # 自動露出が安定するまで捨てるフレーム数
WIDTH        = 1280
HEIGHT       = 720
FPS          = 6
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
    start_idx = next_start_index(SAVE_ROOT)
    already_done = start_idx - 1
    remaining = N_TOTAL - already_done

    if remaining <= 0:
        print(f"すでに {N_TOTAL} 枚撮影済みです（captures/100test/1〜{N_TOTAL}）。")
        print("最初から撮り直す場合は: rm -rf captures/100test/")
        return

    end_idx = start_idx + remaining - 1

    print(f"撮影済み       : {already_done} 枚")
    print(f"今回の範囲     : {start_idx} 〜 {end_idx}")
    print(f"バッチサイズ   : {BATCH_SIZE} 枚ごとに Enter 待ち")
    print(f"保存先         : {SAVE_ROOT.resolve()}")

    # ── カメラ初期化 ────────────────────────────────────────
    conf = rs.config()
    conf.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    conf.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    pipe = rs.pipeline()
    prof = pipe.start(conf)
    align = rs.align(rs.stream.color)

    intr = rs.video_stream_profile(prof.get_stream(rs.stream.color)).get_intrinsics()
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()

    camera_params = {
        "width":       WIDTH,
        "height":      HEIGHT,
        "fx":          intr.fx,
        "fy":          intr.fy,
        "ppx":         intr.ppx,
        "ppy":         intr.ppy,
        "depth_scale": depth_scale,
        "fps":         FPS,
    }

    print(f"\nカメラ起動中... ウォームアップ {N_WARMUP} フレーム")
    for _ in range(N_WARMUP):
        pipe.wait_for_frames()
    print("準備完了\n")

    # ── バッチ撮影ループ ─────────────────────────────────────
    idx     = start_idx
    saved   = 0
    batch   = 0

    while idx <= end_idx:
        batch += 1
        batch_end = min(idx + BATCH_SIZE - 1, end_idx)

        print(f"── バッチ {batch}  ({idx} 〜 {batch_end}、残り {end_idx - idx + 1} 枚) ──")
        input("  Enter を押して撮影開始... ")

        batch_saved = 0
        while idx <= batch_end:
            frames  = pipe.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                print(f"  [{idx}] フレーム取得失敗 → スキップ")
                idx += 1
                continue

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())

            outdir = SAVE_ROOT / str(idx)
            outdir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(outdir / "after_init_rgb.png"), color_np)
            np.save(outdir / "after_init_depth.npy", depth_np)
            (outdir / "camera_params.json").write_text(
                json.dumps(camera_params, indent=2, ensure_ascii=False)
            )

            print(f"  [{idx:3d}/{end_idx}] 保存 → {outdir}")
            batch_saved += 1
            saved += 1
            idx += 1

        print(f"  バッチ {batch} 完了: {batch_saved} 枚\n")

    pipe.stop()
    print(f"撮影完了: {saved} 枚保存  ({SAVE_ROOT}/{start_idx} 〜 {end_idx})")


if __name__ == "__main__":
    main()
