# from __future__ import annotations

# from pathlib import Path
# import json
# import cv2
# import numpy as np
# import sys
# from paddleocr import PaddleOCR


# def save_json(path: Path, obj) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# def draw_ocr_results(img_bgr, out, *, font_scale=0.7, thickness=2):
#     """out: [{"quad": [[x,y]x4], "text": str, "score": float}, ...]"""
#     vis = img_bgr.copy()

#     for item in out:
#         quad = item["quad"]  # [[x,y],...]
#         text = item["text"]
#         score = item.get("score", None)

#         pts = cv2.UMat(cv2.UMat.get(cv2.UMat(quad))) if False else None  # ダミー（無視してOK）

#         pts = cv2.convexHull(
#             cv2.UMat(cv2.UMat.get(cv2.UMat(quad))) if False else
#             cv2.UMat(0)
#         ) if False else None  # ダミー（無視してOK）

#         # quad -> np.int32 (OpenCV用)
#         import numpy as np
#         pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))

#         # 四角形を描画
#         cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

#         # テキスト表示位置（quadの左上っぽい点）
#         x0 = int(min(p[0] for p in quad))
#         y0 = int(min(p[1] for p in quad))
#         label = f"{text}"
#         if score is not None:
#             label += f" ({score:.2f})"

#         # 画像外にはみ出しにくいように少し上へ
#         y_text = max(0, y0 - 5)

#         # 背景付きで文字を描く（見やすい）
#         (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
#         x_bg1, y_bg1 = x0, max(0, y_text - th - baseline)
#         x_bg2, y_bg2 = x0 + tw, y_text + baseline
#         cv2.rectangle(vis, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)  # 黒背景
#         cv2.putText(vis, label, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

#     return vis


# def rotate270_cw(img_bgr: np.ndarray) -> np.ndarray:
#     # （あなたの意図する向きが逆だったので）反対方向へ回す
#     return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)


# def sharpen_unsharp(img_bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.2) -> np.ndarray:
#     """
#     Unsharp mask によるシャープ化
#     sigma: ぼかし強さ
#     amount: シャープ量
#     """
#     blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
#     sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
#     return sharp

# def enhance_contrast_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
#     """BGR画像を高コントラスト化（CLAHE）。返り値もBGR。"""
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     l2 = clahe.apply(l)
#     lab2 = cv2.merge([l2, a, b])
#     return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# def OCR_main(shot_dir: str | Path):
#     shot_dir = Path(shot_dir)

#     ocr = PaddleOCR(
#         ocr_version="PP-OCRv5",
#         lang="japan",
#         use_doc_orientation_classify=False,  # ★勝手に回さない
#         use_doc_unwarping=False,
#         use_textline_orientation=False,
#     )

#     img_path = shot_dir / "after_init_rgb.png"
#     img = cv2.imread(str(img_path))
#     if img is None:
#         raise FileNotFoundError(img_path)

#     img = rotate270_cw(img)  # ★必ず270°回す（時計回り）

#     # （任意：あなたの前処理を使うならここ）
#     # img = enhance_contrast_for_ocr(img)
#     # img = sharpen_unsharp(img)

#     result = ocr.predict(img)

#     res0 = result[0]
#     cv2.imwrite(str(shot_dir / "before_init_rgb_rot270.png"), img)  # ★確認用
#     json_path = shot_dir / "ocr_result.json"
#     vis_path  = shot_dir / "ocr_overlay.png"
#     res0.save_to_json(str(json_path))
#     res0.save_to_img(str(vis_path))
#     return res0

# if __name__ == "__main__":
#     shot_dir = Path(sys.argv[1]).expanduser()
#     OCR_main(shot_dir)

from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import time

os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np
from paddleocr import PaddleOCR

try:
    import paddle
except Exception:  # paddle環境以外での構文チェック用
    paddle = None


_OCR_MODEL = None
_OCR_CREATE_SEC = None
SAVE_ROTATED_OCR_INPUT_DEBUG = False


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_ocr_results(img_bgr, out, *, font_scale=0.7, thickness=2):
    """out: [{"quad": [[x,y]x4], "text": str, "score": float}, ...]"""
    vis = img_bgr.copy()

    for item in out:
        quad = item["quad"]
        text = item["text"]
        score = item.get("score", None)

        pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        x0 = int(min(p[0] for p in quad))
        y0 = int(min(p[1] for p in quad))
        label = f"{text}"
        if score is not None:
            label += f" ({score:.2f})"

        y_text = max(0, y0 - 5)
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )
        x_bg1, y_bg1 = x0, max(0, y_text - th - baseline)
        x_bg2, y_bg2 = x0 + tw, y_text + baseline
        cv2.rectangle(vis, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
        cv2.putText(
            vis,
            label,
            (x0, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return vis


def rotate270_cw(img_bgr: np.ndarray) -> np.ndarray:
    # 現行コードの仕様を維持: OCR入力は必ず時計回り90度回転する．
    return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)


def sharpen_unsharp(img_bgr: np.ndarray, sigma: float = 1.2, amount: float = 1.2) -> np.ndarray:
    """Unsharp mask によるシャープ化．"""
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(img_bgr, 1.0 + amount, blurred, -amount, 0)
    return sharp


def enhance_contrast_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """BGR画像を高コントラスト化（CLAHE）．返り値もBGR．"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _configure_paddle_device_once() -> None:
    """GPU対応Paddleなら gpu:0 を明示する．失敗してもOCR本体に任せる．"""
    if paddle is None:
        return

    try:
        print(f"[OCR CACHE] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[OCR CACHE] paddle.is_compiled_with_cuda={paddle.is_compiled_with_cuda()}", flush=True)
        if paddle.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
        else:
            paddle.set_device("cpu")
        print(f"[OCR CACHE] paddle.device={paddle.device.get_device()}", flush=True)
    except Exception as e:
        print(f"[OCR CACHE] paddle device setup skipped: {e}", flush=True)


def create_ocr_model() -> PaddleOCR:
    """現行OCR設定をそのまま使って，PaddleOCRモデルを1回だけ作る．"""
    _configure_paddle_device_once()

    return PaddleOCR(
        ocr_version="PP-OCRv5",
        lang="japan",
        use_doc_orientation_classify=False,  # 勝手に回さない
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def get_ocr_model() -> PaddleOCR:
    """常駐worker内でOCRモデルをキャッシュする．"""
    global _OCR_MODEL, _OCR_CREATE_SEC

    if _OCR_MODEL is None:
        print(f"[OCR CACHE] create OCR model once. pid={os.getpid()}", flush=True)
        t0 = time.perf_counter()
        _OCR_MODEL = create_ocr_model()
        _OCR_CREATE_SEC = time.perf_counter() - t0
        print(f"[OCR CACHE] OCR model created in {_OCR_CREATE_SEC:.3f} sec", flush=True)
    else:
        print(f"[OCR CACHE] reuse OCR model. pid={os.getpid()}", flush=True)

    return _OCR_MODEL


def _run_ocr_with_model(shot_dir: str | Path, ocr: PaddleOCR):
    shot_dir = Path(shot_dir).expanduser().resolve()

    img_path = shot_dir / "after_init_rgb.png"
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)

    img = rotate270_cw(img)

    # 現行仕様では前処理は未使用．必要なら以下を戻す．
    # img = enhance_contrast_for_ocr(img)
    # img = sharpen_unsharp(img)

    t0 = time.perf_counter()
    result = ocr.predict(img)
    predict_sec = time.perf_counter() - t0
    print(f"[OCR CACHE] OCR predict finished in {predict_sec:.3f} sec", flush=True)

    res0 = result[0]
    if SAVE_ROTATED_OCR_INPUT_DEBUG:
        cv2.imwrite(str(shot_dir / "before_init_rgb_rot270.png"), img)
    json_path = shot_dir / "ocr_result.json"
    vis_path = shot_dir / "ocr_overlay.png"

    t_save = time.perf_counter()
    res0.save_to_json(str(json_path))
    res0.save_to_img(str(vis_path))
    save_sec = time.perf_counter() - t_save
    print(f"[OCR CACHE] OCR result save finished in {save_sec:.3f} sec", flush=True)

    # 計測確認用．get_book_points側のcleanupで画像制限してもjsonは残る．
    save_json(
        shot_dir / "ocr_runtime_info.json",
        {
            "cached_model": True,
            "pid": int(os.getpid()),
            "ocr_model_create_sec": None if _OCR_CREATE_SEC is None else float(_OCR_CREATE_SEC),
            "ocr_predict_sec": float(predict_sec),
            "ocr_save_sec": float(save_sec),
            "input_image": str(img_path),
            "json_path": str(json_path),
            "overlay_path": str(vis_path),
            "save_rotated_input_debug": bool(SAVE_ROTATED_OCR_INPUT_DEBUG),
        },
    )
    return res0


def OCR_main_cached(shot_dir: str | Path):
    """常駐worker用．OCRモデルを作り直さずに使い回す．"""
    ocr = get_ocr_model()
    return _run_ocr_with_model(shot_dir, ocr)


def OCR_main(shot_dir: str | Path):
    """
    互換用エントリポイント．

    単発実行でもこの関数を使えるようにする．同一プロセス内で複数回呼ばれた場合は
    OCR_main_cached() と同じくモデルを再利用する．
    """
    return OCR_main_cached(shot_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python paddle_ocr_test.py <shot_dir>")
    shot_dir = Path(sys.argv[1]).expanduser().resolve()
    OCR_main(shot_dir)
