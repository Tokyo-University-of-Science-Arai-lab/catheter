#!/usr/bin/env python3
"""
オフライン認識評価用に、同じ棚を N 枚撮り直すスクリプト。

1枚ごとにライブプレビューを表示して止まるので、その間にカテーテルの配置や向きを変え、
プレビューウィンドウ上で Enter を押すと撮影する（ESC で中断）。
配置を変えずに撮ると 5 枚が同じ画像になり、ばらつきの評価にならないため。

保存先: captures/5shot_catheter/<実行日時>/<1〜N>/   （実行のたびに新しいフォルダ、上書きしない）
  after_init_rgb.png    : RGB画像 (BGR PNG)
  after_init_depth.npy  : 深度画像 (uint16, Z16)
  camera_params.json    : カメラ内部パラメータ

使い方（ターミナルから対話実行）:
  source .pro_hand_book_fixed/bin/activate
  python capture_5shot.py                 # 1〜5番を撮り直す（新しいフォルダに保存）
  python capture_5shot.py --n-shots 3     # 1〜3番だけ
  python capture_5shot.py --start 3       # 3番から5番まで（同じ実行内での一部撮り直し）
  python capture_5shot.py --run-dir captures/5shot_catheter/20260824_120000 --start 3
                                           # 既存フォルダに追記・上書きしたい場合に明示指定
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── 設定 ────────────────────────────────────────────────────
DEFAULT_SAVE_ROOT = Path("captures") / "5shot_catheter"
N_SHOTS   = 5     # 撮影枚数
N_WARMUP  = 30    # 自動露出が安定するまで捨てるフレーム数
N_SETTLE  = 10    # Enter後、配置変更の手ブレが収まるまで捨てるフレーム数
WIDTH     = 1280
HEIGHT    = 720
FPS       = 6

# 配置変更で長く待つ間にカメラが止まることがあるため、
# 既定の5秒より長く待ち、失敗したらパイプラインを開き直して復帰する。
FRAME_TIMEOUT_MS = 15000
N_RETRY          = 3
# ────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="評価用に棚を撮り直す（Enterごとに1枚）")
    parser.add_argument("--save-root", type=Path, default=DEFAULT_SAVE_ROOT,
                        help=f"新しい実行フォルダを作る親ディレクトリ（デフォルト: {DEFAULT_SAVE_ROOT}）")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="このフォルダに直接保存する（新規フォルダを作らず、既存の実行に"
                             "追記・上書きしたい場合に指定）")
    parser.add_argument("--n-shots", type=int, default=N_SHOTS,
                        help=f"撮影する枚数（デフォルト: {N_SHOTS}）")
    parser.add_argument("--start", type=int, default=1,
                        help="開始番号。途中だけ撮り直すときに使う（デフォルト: 1）")
    return parser.parse_args()


def make_unique_run_dir(base_dir: Path, timestamp: str) -> Path:
    """base_dir/<timestamp> を作る。同じ秒に複数回実行した場合は _001, _002 ... を付ける。"""
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir = base_dir / timestamp
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return run_dir

    for i in range(1, 1000):
        candidate = base_dir / f"{timestamp}_{i:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

    raise RuntimeError(f"unique run dir を作れませんでした: {base_dir}/{timestamp}_XXX")


class Camera:
    """RealSense をラップし、フレーム取得に失敗したら開き直して復帰する。"""

    def __init__(self):
        self.pipe = None
        self.align = rs.align(rs.stream.color)
        self.camera_params = None
        self.start()

    def start(self):
        conf = rs.config()
        conf.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        conf.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

        self.pipe = rs.pipeline()
        prof = self.pipe.start(conf)

        intr = rs.video_stream_profile(prof.get_stream(rs.stream.color)).get_intrinsics()
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.camera_params = {
            "width":       WIDTH,
            "height":      HEIGHT,
            "fx":          intr.fx,
            "fy":          intr.fy,
            "ppx":         intr.ppx,
            "ppy":         intr.ppy,
            "depth_scale": depth_scale,
            "fps":         FPS,
        }

    def stop(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.pipe = None

    def restart(self):
        """USBが不安定でフレームが来なくなったときの復帰処理。"""
        print("  カメラを開き直しています...")
        self.stop()
        time.sleep(2.0)
        self.start()
        for _ in range(N_WARMUP):
            self.pipe.wait_for_frames(FRAME_TIMEOUT_MS)
        print("  復帰しました")

    def warmup(self, n: int):
        for _ in range(n):
            self.pipe.wait_for_frames(FRAME_TIMEOUT_MS)

    def preview_color_frame(self):
        """プレビュー表示用に、整列済みのcolorフレームを1枚返す（取得失敗時はNone）。"""
        try:
            aligned = self.align.process(self.pipe.wait_for_frames(FRAME_TIMEOUT_MS))
            color_frame = aligned.get_color_frame()
            if color_frame:
                return np.asanyarray(color_frame.get_data())
        except RuntimeError:
            pass
        return None

    def grab(self, n_settle: int):
        """
        整列済みの (color, depth) を返す。直前の n_settle 枚は捨てる。

        wait_for_frames はタイムアウトすると RuntimeError を投げるので、
        例外も「取得失敗」として扱い、リトライ・再起動で復帰させる。
        """
        last_error = None

        for attempt in range(1, N_RETRY + 1):
            try:
                for _ in range(n_settle):
                    self.pipe.wait_for_frames(FRAME_TIMEOUT_MS)

                aligned = self.align.process(self.pipe.wait_for_frames(FRAME_TIMEOUT_MS))
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if color_frame and depth_frame:
                    return (np.asanyarray(color_frame.get_data()),
                            np.asanyarray(depth_frame.get_data()))
                last_error = "フレームが空です"
            except RuntimeError as e:
                last_error = str(e)

            print(f"  フレーム取得に失敗（{attempt}/{N_RETRY}回目）: {last_error}")
            if attempt < N_RETRY:
                try:
                    self.restart()
                except Exception as e:
                    print(f"  再起動にも失敗しました: {e}")

        raise RuntimeError(f"フレームを取得できませんでした: {last_error}")


PREVIEW_WINDOW = "capture_5shot preview (Enter:撮影 / ESC:中断)"


def wait_for_capture_key(cam: "Camera") -> bool:
    """ライブプレビューを表示しつつキー入力を待つ。EnterならTrue、ESCならFalseを返す。"""
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = cam.preview_color_frame()
            if frame is not None:
                cv2.imshow(PREVIEW_WINDOW, frame)
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # Enter
                return True
            if key == 27:  # ESC
                return False
            if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                return False
    finally:
        cv2.destroyWindow(PREVIEW_WINDOW)


def save_shot(outdir: Path, color_np, depth_np, camera_params):
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "after_init_rgb.png"), color_np)
    np.save(outdir / "after_init_depth.npy", depth_np)
    (outdir / "camera_params.json").write_text(
        json.dumps(camera_params, indent=2, ensure_ascii=False)
    )


def main():
    args = parse_args()
    start_idx = args.start
    end_idx = args.start + args.n_shots - 1

    if args.run_dir is not None:
        run_dir = args.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        overwrite_note = "（このフォルダ内で既存ファイルは上書き）"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = make_unique_run_dir(args.save_root, timestamp)
        overwrite_note = "（新規フォルダなので上書きなし）"

    print("=" * 55)
    print("  評価用 撮り直しスクリプト")
    print("=" * 55)
    print(f"保存先   : {run_dir.resolve()}")
    print(f"撮影範囲 : {start_idx} 〜 {end_idx} {overwrite_note}")
    print()
    print("  ※ 1枚ごとにプレビューウィンドウが開きます。その間にカテーテルの配置を")
    print("    変え、ウィンドウ上で Enter を押すと撮影します（ESCで中断）。")
    print("  ※ 全20種がフレームに入っているか確認してください。")
    print("=" * 55)

    cam = Camera()
    saved, failed = [], []
    try:
        print(f"\nカメラ起動中... ウォームアップ {N_WARMUP} フレーム")
        cam.warmup(N_WARMUP)
        print("準備完了")

        idx = start_idx
        while idx <= end_idx:
            outdir = run_dir / str(idx)
            mark = "（上書き）" if (outdir / "after_init_rgb.png").exists() else "（新規）"

            print(f"\n── {idx} 枚目 {mark} ──")
            print("  配置を変えたら、プレビューウィンドウでEnterを押して撮影 > ")
            try:
                if not wait_for_capture_key(cam):
                    print("\n中断しました。")
                    break
            except KeyboardInterrupt:
                print("\n中断しました。")
                break

            # 1枚失敗しても全体を止めない。その番号だけ後で撮り直せるようにする。
            try:
                color_np, depth_np = cam.grab(N_SETTLE)
            except RuntimeError as e:
                print(f"  [ERROR] {idx} 枚目を撮影できませんでした: {e}")
                print("  USB接続を確認してください。Enterでこの番号を再試行します。")
                failed.append(idx)
                continue

            save_shot(outdir, color_np, depth_np, cam.camera_params)
            if idx in failed:
                failed.remove(idx)
            saved.append(idx)

            valid_ratio = float((depth_np > 0).mean())
            print(f"  保存 → {outdir}")
            print(f"  深度の有効画素: {valid_ratio * 100:.1f}%")
            if valid_ratio < 0.5:
                print("  [WARN] 深度が欠けています。照明や距離を確認してください。")

            idx += 1
    finally:
        cam.stop()

    print(f"\n撮影完了: {len(saved)} 枚  → {run_dir.resolve()}")
    if failed:
        print(f"未取得の番号: {sorted(set(failed))}")
        print(
            f"  撮り直し例: python capture_5shot.py --run-dir {run_dir} "
            f"--start {sorted(set(failed))[0]} --n-shots 1"
        )
        return 1

    print("次のコマンドで評価を実行できます:")
    print("  python offline_pointcloud_debug_SAM3_catheter.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
