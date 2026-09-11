#!/usr/bin/env python3
"""
RealSense D435i でアノテーション用の撮影データを1枚ずつ撮影するスクリプト。

capture_5shot.pyと同じ操作方式: ライブプレビューウィンドウを見ながら位置決めし、
ウィンドウ上で Enter を押すと1枚撮影する。撮影後は自動で次の撮影のプレビューに戻り、
ESC または Ctrl+C で終了するまで無限に続く。

【2026-08-25、ユーザー要望】RGBのみの保存から、reco/<dataset>/depth_shots/形式と同じ
RGB+depth+camera_params.jsonの3点セット保存に変更した。あわせて、CVAT等でのアノテーション
用にRGBだけを集めたimages/フォルダも並行して作る(reco/<dataset>/images/と同じ命名:
<shot番号>.png)。ここで作った1回分のフォルダ(depth_shots/ + images/)は、そのまま
reco/<新データセット名>/ 配下に配置すれば既存の評価スクリプト(width_mm_validation.py等)
にそのまま使える構成にしてある。

保存先: anno-catheter/<連番>/
  depth_shots/<shot番号>/
    after_init_rgb.png    : RGB画像 (BGR PNG)
    after_init_depth.npy  : 深度画像 (uint16, Z16)
    camera_params.json    : カメラ内部パラメータ
  images/<shot番号>.png   : RGBのみを集めたコピー(アノテーションツール用)

連番フォルダ(anno-catheter/<連番>/)は起動時に1つだけ作られ、中断するまでの撮影は
すべて同じフォルダに保存される。連番は既存の数字フォルダの続きから自動で決まる
（1 があれば 2、2 があれば 3、…）。shot番号は実行のたびに1から始まる。
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── 設定 ────────────────────────────────────────────────────
SAVE_ROOT = Path("anno-catheter")
N_WARMUP  = 30    # 自動露出が安定するまで捨てるフレーム数
N_SETTLE  = 10    # Enter後、配置変更の手ブレが収まるまで捨てるフレーム数
WIDTH     = 1280
HEIGHT    = 720
FPS       = 6

# 配置変更で長く待つ間にカメラが止まることがあるため、
# 既定の5秒より長く待ち、失敗したらパイプラインを開き直して復帰する(capture_5shot.pyと同じ)。
FRAME_TIMEOUT_MS = 15000
N_RETRY          = 3
# ────────────────────────────────────────────────────────────

PREVIEW_WINDOW = "capture_anno preview (Enter:撮影 / ESC:終了)"


def next_start_index(save_root: Path) -> int:
    """既存の連番フォルダを調べ、次に使うべきインデックスを返す。"""
    if not save_root.exists():
        return 1
    existing = [
        int(p.name) for p in save_root.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    return max(existing) + 1 if existing else 1


class Camera:
    """RealSense(color+depth整列)をラップし、フレーム取得に失敗したら開き直して復帰する
    (capture_5shot.pyと同じ設計)。"""

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


def wait_for_capture_key(cam: "Camera") -> bool:
    """ライブプレビューを表示しつつキー入力を待つ。EnterならTrue、ESCならFalseを返す。"""
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, int(WIDTH * 0.9), int(HEIGHT * 0.9))
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


def save_shot(depth_shots_dir: Path, images_dir: Path, shot_idx: int,
              color_np, depth_np, camera_params) -> Path:
    shot_dir = depth_shots_dir / str(shot_idx)
    shot_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(shot_dir / "after_init_rgb.png"), color_np)
    np.save(shot_dir / "after_init_depth.npy", depth_np)
    (shot_dir / "camera_params.json").write_text(
        json.dumps(camera_params, indent=2, ensure_ascii=False)
    )

    images_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / f"{shot_idx}.png"), color_np)

    return shot_dir


def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = SAVE_ROOT / str(next_start_index(SAVE_ROOT))
    depth_shots_dir = run_dir / "depth_shots"
    images_dir = run_dir / "images"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"保存先: {run_dir.resolve()}")
    print("  depth_shots/<N>/ : RGB+depth+camera_params.jsonの3点セット")
    print("  images/<N>.png   : RGBのみ(アノテーション用)")
    print("プレビューウィンドウでEnterを押すと撮影、ESCまたはCtrl+Cで終了\n")

    cam = Camera()
    saved = 0
    try:
        print(f"カメラ起動中... ウォームアップ {N_WARMUP} フレーム")
        cam.warmup(N_WARMUP)
        print("準備完了\n")

        while True:
            shot_idx = saved + 1
            print(f"[{shot_idx} 枚目] 配置を決めたら、プレビューウィンドウでEnterを押して撮影 > ")
            try:
                if not wait_for_capture_key(cam):
                    print("\n終了します。")
                    break
            except KeyboardInterrupt:
                print("\n中断しました。")
                break

            try:
                color_np, depth_np = cam.grab(N_SETTLE)
            except RuntimeError as e:
                print(f"  [ERROR] {shot_idx} 枚目を撮影できませんでした: {e}")
                print("  USB接続を確認してください。Enterでもう一度撮影してください。")
                continue

            shot_dir = save_shot(depth_shots_dir, images_dir, shot_idx,
                                  color_np, depth_np, cam.camera_params)
            saved += 1

            valid_ratio = float((depth_np > 0).mean())
            print(f"  保存 → {shot_dir}, {images_dir / f'{shot_idx}.png'}")
            print(f"  深度の有効画素: {valid_ratio * 100:.1f}%")
            if valid_ratio < 0.5:
                print("  [WARN] 深度が欠けています。照明や距離を確認してください。")
    except KeyboardInterrupt:
        print("\n中断しました。")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"\n撮影完了: {saved} 枚保存 → {run_dir.resolve()}")


if __name__ == "__main__":
    main()
