#!/usr/bin/env python3
"""
OCR精度が解像度によってどう変わるかを実験するための撮影スクリプト(2026-10-06追加)。

capture_anno.pyと同じ操作方式(ライブプレビュー+Enterで撮影)だが、
Enterを1回押すたびに、同じ画角のまま複数の解像度で連続撮影する
(RealSenseは稼働中のパイプラインの解像度を変更できないため、解像度ごとに
パイプラインを停止→再起動している。1枚ごとに数秒のラグが出る)。

深度は使わないので、保存するのはRGB画像のみ。

保存先: captures/resolution_test/<連番>/<width>x<height>/<shot番号>.png
  例: captures/resolution_test/1/1920x1080/1.png
      captures/resolution_test/1/1280x720/1.png

連番フォルダは起動時に1つだけ作られ、中断するまでの撮影はすべて同じフォルダに
保存される。連番は既存の数字フォルダの続きから自動で決まる。shot番号は実行のたびに
1から始まる。

実験のもう1つのデータ(スマホ撮影分)は、このスクリプトの対象外。別途用意すること。
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── 設定 ────────────────────────────────────────────────────
SAVE_ROOT = Path("captures/resolution_test")
RESOLUTIONS = [(1920, 1080), (1280, 720)]  # 撮りたい解像度のリスト
PREVIEW_WIDTH, PREVIEW_HEIGHT = 1280, 720  # ライブプレビュー用(低い方で十分)
FPS = 6
N_WARMUP  = 30    # 自動露出が安定するまで捨てるフレーム数
N_SETTLE  = 10    # Enter後、解像度切り替え直後の手ブレ・自動露出再調整が収まるまで捨てるフレーム数

FRAME_TIMEOUT_MS = 15000
N_RETRY          = 3
# ────────────────────────────────────────────────────────────

PREVIEW_WINDOW = "capture_resolution_test preview (Enter:撮影 / ESC:終了)"


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
    """RealSenseのcolorストリームだけをラップする。解像度を変えるたびに
    stop() → 別設定でstart() する必要がある(稼働中の切り替えはできない)。"""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.pipe = None
        self.start()

    def start(self):
        conf = rs.config()
        conf.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, FPS)
        self.pipe = rs.pipeline()
        self.pipe.start(conf)

    def stop(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.pipe = None

    def switch_resolution(self, width: int, height: int):
        self.stop()
        time.sleep(0.5)
        self.width, self.height = width, height
        self.start()

    def warmup(self, n: int):
        for _ in range(n):
            self.pipe.wait_for_frames(FRAME_TIMEOUT_MS)

    def preview_color_frame(self):
        try:
            frame = self.pipe.wait_for_frames(FRAME_TIMEOUT_MS).get_color_frame()
            if frame:
                return np.asanyarray(frame.get_data())
        except RuntimeError:
            pass
        return None

    def grab(self, n_settle: int):
        """colorフレームを1枚返す。直前のn_settle枚は捨てる。失敗時はリトライする。"""
        last_error = None
        for attempt in range(1, N_RETRY + 1):
            try:
                for _ in range(n_settle):
                    self.pipe.wait_for_frames(FRAME_TIMEOUT_MS)
                color_frame = self.pipe.wait_for_frames(FRAME_TIMEOUT_MS).get_color_frame()
                if color_frame:
                    return np.asanyarray(color_frame.get_data())
                last_error = "フレームが空です"
            except RuntimeError as e:
                last_error = str(e)

            print(f"  フレーム取得に失敗（{attempt}/{N_RETRY}回目）: {last_error}")
            if attempt < N_RETRY:
                print("  カメラを開き直しています...")
                self.stop()
                time.sleep(2.0)
                self.start()
                self.warmup(N_WARMUP)

        raise RuntimeError(f"フレームを取得できませんでした: {last_error}")


def wait_for_capture_key(cam: "Camera") -> bool:
    """ライブプレビューを表示しつつキー入力を待つ。EnterならTrue、ESCならFalseを返す。"""
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, int(cam.width * 0.9), int(cam.height * 0.9))
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


def save_shot(run_dir: Path, width: int, height: int, shot_idx: int, color_np) -> Path:
    res_dir = run_dir / f"{width}x{height}"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / f"{shot_idx}.png"
    cv2.imwrite(str(out_path), color_np)
    return out_path


def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = SAVE_ROOT / str(next_start_index(SAVE_ROOT))
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"保存先: {run_dir.resolve()}")
    print(f"撮影する解像度: {', '.join(f'{w}x{h}' for w, h in RESOLUTIONS)}")
    print("プレビューウィンドウでEnterを押すと、全解像度を自動で連続撮影します")
    print("ESCまたはCtrl+Cで終了\n")

    cam = Camera(PREVIEW_WIDTH, PREVIEW_HEIGHT)
    saved = 0
    try:
        print(f"カメラ起動中... ウォームアップ {N_WARMUP} フレーム")
        cam.warmup(N_WARMUP)
        print("準備完了\n")

        while True:
            shot_idx = saved + 1
            print(f"[{shot_idx} 枚目] 配置を決めたら、プレビューウィンドウでEnterを押して撮影 > ")
            try:
                if cam.width != PREVIEW_WIDTH or cam.height != PREVIEW_HEIGHT:
                    cam.switch_resolution(PREVIEW_WIDTH, PREVIEW_HEIGHT)
                    cam.warmup(N_WARMUP)
                if not wait_for_capture_key(cam):
                    print("\n終了します。")
                    break
            except KeyboardInterrupt:
                print("\n中断しました。")
                break

            ok = True
            for width, height in RESOLUTIONS:
                print(f"  {width}x{height} へ切り替え中...")
                try:
                    cam.switch_resolution(width, height)
                    cam.warmup(N_WARMUP)
                    color_np = cam.grab(N_SETTLE)
                except RuntimeError as e:
                    print(f"  [ERROR] {width}x{height} の撮影に失敗しました: {e}")
                    ok = False
                    break

                out_path = save_shot(run_dir, width, height, shot_idx, color_np)
                print(f"  保存 → {out_path}")

            if ok:
                saved += 1
            else:
                print("  この回は一部解像度が撮れませんでした。Enterでもう一度撮影してください。")
    except KeyboardInterrupt:
        print("\n中断しました。")
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"\n撮影完了: {saved} 回分(解像度ごとに1枚ずつ) → {run_dir.resolve()}")


if __name__ == "__main__":
    main()
