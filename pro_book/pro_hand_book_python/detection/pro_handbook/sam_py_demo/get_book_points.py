#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RealSense 1ショット撮影 → SAM(infer_for_storage.infer_masks)で書籍領域認識
→ 対象書籍のRGB/Depth保存 → intrinsics+depthで点群化 → PLY保存
→ PCAで背表紙方向などの特徴量を計算

モジュールとしての使い方例:
  from rs_book_capture_and_pointcloud import run_capture_and_pca

  theta_rad, p_min, p_max = run_capture_and_pca(out_dir="captures")

スクリプトとしての使い方例:
  python rs_book_capture_and_pointcloud.py --out captures
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import open3d as o3d
import numpy as np
import cv2
from PIL import Image
import json
import subprocess
import re
import traceback
from .OCR.only_one import find_similar_books
from .OCR.only_one_tilted import match_text_to_mask_main

# ===== RealSense =====
import pyrealsense2 as rs

# ===== SAM: infer_for_storage の infer_masks を使う =====
from .infer_for_storage import SamConfig, SamBatchInfer_storage, StageSaveCfg
from modules.overlay_io import _render_overlay_bgr, _save_points_and_overlay
#from bar_code.code_1_pic import detect_barcode
# 点群モジュール
from modules.pointcloud_utils import masked_depth_to_points, save_ply_ascii
from modules.calculate_3D_point_or_RANSAC import calculate_yaw
from modules.pca_vector import pca_axes_fix_dir
from modules.book_width import estimate_book_width
from modules.grip_point import find_target_point
from modules.open3d_view import visualize_points_and_target_open3d

ALLOWABLE_RANGE_Z = 0.07

def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def depth_filter_like_viewer(raw_depth_frame: rs.depth_frame) -> rs.depth_frame:
    """
    rs-viewer に近いフィルタ処理（あなたの depth_filter と同等）
    """
    decim = rs.decimation_filter()
    decim.set_option(rs.option.filter_magnitude, 2.0)

    spat = rs.spatial_filter()
    spat.set_option(rs.option.filter_magnitude, 2.0)
    spat.set_option(rs.option.filter_smooth_alpha, 0.5)
    spat.set_option(rs.option.filter_smooth_delta, 20.0)

    hole_fill = rs.hole_filling_filter()

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # viewer準拠: decimation は RGB 解像とズレる危険があるのでスキップ
    filtered = depth_to_disparity.process(raw_depth_frame)
    filtered = spat.process(filtered)
    filtered = disparity_to_depth.process(filtered)
    filtered = hole_fill.process(filtered)
    return filtered.as_depth_frame()


# def save_masked_and_cropped(
#     rgb_bgr: np.ndarray,
#     depth_u16: np.ndarray,
#     mask01: np.ndarray,
#     outdir: Path,
#     stem: str,
# ):
#     """
#     対象書籍のみの RGB/Depth を保存（背景0マスク）
#     """
#     outdir.mkdir(parents=True, exist_ok=True)

#     # --- マスク適用（背景=0） ---
#     rgb_masked = rgb_bgr.copy()
#     rgb_masked[mask01 == 0] = 0
#     depth_masked = depth_u16.copy()
#     depth_masked[mask01 == 0] = 0

#     cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
#     np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

#     # 深度可視化（0 以外の範囲を 0–255 に正規化）
#     nonzero = depth_masked[depth_masked > 0]
#     if nonzero.size > 0:
#         zmin, zmax = int(nonzero.min()), int(nonzero.max())
#         zrange = max(1, zmax - zmin)
#         depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
#         depth_vis[depth_masked > 0] = (
#             (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
#         ).astype(np.uint8)
#         cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

#     return depth_masked # 変更（追加）

def save_masked_and_cropped(
    rgb_bgr: np.ndarray,
    depth_u16: np.ndarray,
    mask01: np.ndarray,
    outdir: Path,
    stem: str,
    z_tolerance_raw: int = 30,  # Z16値での許容幅（例: ±80カウント ≈ ±8cm 程度）
):
    """
    対象書籍のみの RGB/Depth を保存（背景0マスク ＋ 深度外れ値除去）
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # --- マスク適用（背景=0） ---
    rgb_masked = rgb_bgr.copy()
    rgb_masked[mask01 == 0] = 0

    depth_masked = depth_u16.copy()
    depth_masked[mask01 == 0] = 0  # まずマスク外を 0 にする

    # --- ここから: マスク内の depth の外れ値除去（1D簡易版 RANSAC 的なもの） ---
    # マスク内で depth が 0 でない画素だけ取り出す
    nonzero = depth_masked[depth_masked > 0]

    if nonzero.size > 0:
        # 主体（本）の「代表的な距離」を中央値で近似
        z_med = int(np.median(nonzero))

        # 中央値から外れすぎた depth を 0 にする
        #   -> 本より手前/奥にある別の物体を消す目的
        z_min_keep = z_med - z_tolerance_raw
        z_max_keep = z_med + z_tolerance_raw

        # 残したい領域（Trueが残す）
        keep = (depth_masked >= z_min_keep) & (depth_masked <= z_max_keep)

        # keep==False のところを 0 にする
        depth_masked[~keep] = 0

    # --- ファイル保存 ---
    cv2.imwrite(str(outdir / f"{stem}_rgb_masked.png"), rgb_masked)
    np.save(outdir / f"{stem}_depth_masked.npy", depth_masked)

    # 深度可視化（0 以外の範囲を 0–255 に正規化） ※外れ値除去後の結果を可視化
    nonzero = depth_masked[depth_masked > 0]
    if nonzero.size > 0:
        zmin, zmax = int(nonzero.min()), int(nonzero.max())
        zrange = max(1, zmax - zmin)
        depth_vis = np.zeros_like(depth_masked, dtype=np.uint8)
        depth_vis[depth_masked > 0] = (
            (depth_masked[depth_masked > 0] - zmin) * 255 // zrange
        ).astype(np.uint8)
        cv2.imwrite(str(outdir / f"{stem}_depth_masked_vis.png"), depth_vis)

    return depth_masked  # ここで外れ値除去済みの depth を返す

def capture_one_shot(pipe, cfg, align, shot_dir, *, stem: str, color_only: bool = False):
    profile = pipe.start(cfg)

    for _ in range(10):
        pipe.wait_for_frames()

    frames = pipe.wait_for_frames()

    if color_only:
        color_frame = frames.get_color_frame()
        color_np = np.asanyarray(color_frame.get_data())

        # 左右反転しないで保存
        cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)

        pipe.stop()
        return color_np, None, None, None

    # depthも使う場合（2回目）
    align_frames = align.process(frames)
    depth_frame = depth_filter_like_viewer(align_frames.get_depth_frame())
    color_frame = align_frames.get_color_frame()

    color_np = np.asanyarray(color_frame.get_data())
    depth_np_u16 = np.asanyarray(depth_frame.get_data())

    # 左右反転しないで保存
    cv2.imwrite(str(shot_dir / f"{stem}_rgb.png"), color_np)
    np.save(shot_dir / f"{stem}_depth.npy", depth_np_u16)

    dprof = rs.video_stream_profile(depth_frame.get_profile())
    intr = dprof.get_intrinsics()
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

    pipe.stop()
    return color_np, depth_np_u16, intr, depth_scale #後で調べる
def run_ocr_subprocess(shot_dir: Path):
    # OCR用仮想環境の python へのパス（あなたの環境に合わせて変更）
    # Linux venv例: /home/book/venv/ocr/bin/python
    OCR_PY = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python"
    OCR_SCRIPT = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py"

    subprocess.run([OCR_PY, OCR_SCRIPT, str(shot_dir)], check=True)
    print(f"✔ OCR done: {shot_dir / 'ocr_result.json'}")
def start_ocr_subprocess(shot_dir: Path):
    """
    OCR を非同期で開始して Popen を返す。
    後で communicate() / wait() して終了を待つ。
    """
    OCR_PY = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/.paadle_ocr/bin/python"
    OCR_SCRIPT = "/home/book/pro_book/pro_hand_book_python/detection/pro_handbook/sam_py_demo/OCR/paddle_ocr_test.py"

    proc = subprocess.Popen(
        [OCR_PY, OCR_SCRIPT, str(shot_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def wait_ocr_subprocess(proc: subprocess.Popen, *, timeout: float | None = None) -> str:
    """
    OCR サブプロセスの終了待ち。
    正常終了しなければ例外を投げる。
    """
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        raise RuntimeError(f"OCR subprocess timeout.\n{stdout}")

    if proc.returncode != 0:
        raise RuntimeError(f"OCR subprocess failed (code={proc.returncode}).\n{stdout}")

    return stdout or ""


def merge_ocr_and_masks(
    query: str,
    masks,
    shot_dir: Path,
    interactive: bool = True,
    threshold: int = 40,
):
    """
    OCR 結果(json) と SAM マスクを統合して、選択マスクを返す。
    """
    results = match_text_to_mask_main(query, masks, shot_dir, threshold=threshold)

    book_name = results[0]["name"] if results else None

    sel_idx = None
    if not interactive:
        sel_idx = 1

    if book_name:
        m = re.search(r"(\d+)$", book_name)  # 末尾の連続数字
        if m:
            sel_idx = int(m.group(1))

    if sel_idx is None:
        raise RuntimeError("マスクIDが選べませんでした（OCR結果なし or 対応付け失敗）")

    sel_mask = masks[sel_idx - 1]
    mask01 = (np.asarray(sel_mask) > 0).astype(np.uint8)

    return {
        "results": results,
        "book_name": book_name,
        "sel_idx": sel_idx,
        "sel_mask": sel_mask,
        "mask01": mask01,
    }



def _poly_from_any(value):
    """
    OCR bbox/polygon の表現ゆれを吸収して，4点以上の polygon を返す．
    返せない場合は None．
    """
    if value is None:
        return None

    # dict形式: {x1,y1,x2,y2} は axis-aligned box として4点化する．
    if isinstance(value, dict):
        keys = set(value.keys())
        if {"x1", "y1", "x2", "y2"}.issubset(keys):
            x1 = float(value["x1"])
            y1 = float(value["y1"])
            x2 = float(value["x2"])
            y2 = float(value["y2"])
            return np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            )

        for key in ("poly", "points", "bbox", "box", "dt_poly", "quadrilateral"):
            if key in value:
                poly = _poly_from_any(value[key])
                if poly is not None:
                    return poly
        return None

    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None

    # 例: [[x1,y1],...[x4,y4]]
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 4:
        return arr[:4].astype(np.float32)

    # 例: [[[x1,y1],...]] のように一段深い場合
    if arr.ndim == 3 and arr.shape[-1] == 2 and arr.shape[-2] >= 4:
        return arr.reshape(-1, arr.shape[-2], 2)[0, :4].astype(np.float32)

    # 例: [x1,y1,x2,y2,x3,y3,x4,y4]
    if arr.ndim == 1 and arr.size >= 8 and arr.size % 2 == 0:
        return arr.reshape(-1, 2)[:4].astype(np.float32)

    return None


def _polygon_center_and_axis(poly: np.ndarray):
    """
    4点 polygon から中心，長辺方向，短辺長，長辺長を返す．
    OCR polygon が傾いていれば，その傾きに沿った軸が返る．
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 4:
        return None, None, None, None

    poly = poly[:4]
    center = poly.mean(axis=0).astype(np.float64)

    edges = []
    for i in range(4):
        v = poly[(i + 1) % 4] - poly[i]
        length = float(np.linalg.norm(v))
        if length > 1e-6:
            edges.append((length, v.astype(np.float64) / length))

    if not edges:
        return center, None, None, None

    edges_sorted = sorted(edges, key=lambda x: x[0], reverse=True)
    long_len, long_axis = edges_sorted[0]
    short_len = edges_sorted[-1][0]

    # 符号は任意なので，後段で使いやすいように正規化だけする．
    n = float(np.linalg.norm(long_axis))
    if n < 1e-9:
        return center, None, short_len, long_len

    return center, long_axis / n, short_len, long_len


# ===== OCR polygon 座標系補正 =====
# ocr_result.json 内の polygon が after_init_rgb.png と 90度ずれている場合に備え，
# 複数の座標変換を試し，選択SAMマスクや merged box と最も整合するものを自動選択する．
_OCR_POLY_TRANSFORM_MODES = (
    "identity",
    "ocr_cw_to_rgb",
    "ocr_ccw_to_rgb",
    "ocr_180_to_rgb",
)


def _transform_ocr_poly_to_rgb(poly: np.ndarray, image_shape: tuple[int, int], mode: str) -> np.ndarray:
    """
    OCR polygon を RGB画像座標系へ写す．

    mode:
      identity        : 変換なし
      ocr_cw_to_rgb   : OCR画像がRGB画像を90度時計回りに回転したものだった場合の逆変換
      ocr_ccw_to_rgb  : OCR画像がRGB画像を90度反時計回りに回転したものだった場合の逆変換
      ocr_180_to_rgb  : OCR画像がRGB画像を180度回転したものだった場合の逆変換

    注意:
      ここでは after_init_rgb.png の shape=(H,W) を基準にする．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return None

    H, W = int(image_shape[0]), int(image_shape[1])
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2).copy()
    x = p[:, 0].copy()
    y = p[:, 1].copy()

    if mode == "identity":
        out = np.stack([x, y], axis=1)
    elif mode == "ocr_cw_to_rgb":
        # RGB -> OCR が cv2.ROTATE_90_CLOCKWISE だったと仮定した逆変換．
        # OCR(xr,yr) -> RGB(x=yr, y=H-1-xr)
        out = np.stack([y, (H - 1) - x], axis=1)
    elif mode == "ocr_ccw_to_rgb":
        # RGB -> OCR が cv2.ROTATE_90_COUNTERCLOCKWISE だったと仮定した逆変換．
        # OCR(xr,yr) -> RGB(x=W-1-yr, y=xr)
        out = np.stack([(W - 1) - y, x], axis=1)
    elif mode == "ocr_180_to_rgb":
        out = np.stack([(W - 1) - x, (H - 1) - y], axis=1)
    else:
        raise ValueError(f"unknown OCR polygon transform mode: {mode}")

    return out.astype(np.float32)


def _poly_image_valid_score(poly: np.ndarray, image_shape: tuple[int, int]) -> float:
    """
    polygon が画像内にどの程度入っているかを 0〜1 で返す．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return 0.0
    H, W = int(image_shape[0]), int(image_shape[1])
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    inside = (p[:, 0] >= 0) & (p[:, 0] < W) & (p[:, 1] >= 0) & (p[:, 1] < H)
    return float(np.mean(inside))


def _center_distance_score(center: np.ndarray | None, target_center: np.ndarray | None, image_shape: tuple[int, int]) -> float:
    """
    center が target_center に近いほど 1 に近いスコアを返す．
    target_center が無ければ 0．
    """
    if center is None or target_center is None:
        return 0.0
    H, W = int(image_shape[0]), int(image_shape[1])
    scale = max(80.0, 0.20 * float(max(H, W)))
    d = float(np.linalg.norm(np.asarray(center, dtype=np.float64) - np.asarray(target_center, dtype=np.float64)))
    return float(np.exp(-d / scale))


def _choose_axis_consistent_with_mask(
    ocr_axis: np.ndarray,
    mask_axis: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """
    OCR polygon の長辺方向と，その直交方向のうち，SAMマスク主軸に近い方を選ぶ．

    理由:
      OCR polygon の長辺は「文字列方向」を表す場合があり，背表紙方向と90度ずれることがある．
      そのため，SAMマスクの主軸を参照して，長辺方向か短辺方向かを自動選択する．
    """
    axis = np.asarray(ocr_axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return axis, "invalid"
    axis = axis / n

    if mask_axis is None:
        return axis, "ocr_long_axis_no_mask_reference"

    m = np.asarray(mask_axis, dtype=np.float64)
    mn = float(np.linalg.norm(m))
    if mn < 1e-9:
        return axis, "ocr_long_axis_no_mask_reference"
    m = m / mn

    perp = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    score_long = abs(float(np.dot(axis, m)))
    score_perp = abs(float(np.dot(perp, m)))

    if score_perp > score_long:
        return perp, "ocr_perpendicular_axis_aligned_to_mask_pca"
    return axis, "ocr_long_axis_aligned_to_mask_pca"


def _mask_centroid(mask01: np.ndarray):
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0:
        return None
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _mask_pca_axis(mask01: np.ndarray):
    """
    SAMマスクの2D画素から主軸を求める．
    OCR polygon が使えない場合のフォールバック．
    """
    ys, xs = np.where(mask01 > 0)
    if len(xs) < 20:
        return None

    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0, keepdims=True)

    try:
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        return None

    axis = vecs[:, int(np.argmax(vals))].astype(np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return None
    return axis / n


def _text_similarity(a: str, b: str) -> float:
    """
    OCR文字列と query の緩い類似度．
    タイトル全体でなく一部だけ読めた場合も拾えるようにする．
    """
    from difflib import SequenceMatcher

    def norm(s: str) -> str:
        s = str(s)
        s = re.sub(r"\s+", "", s)
        s = s.replace("［", "[").replace("］", "]")
        s = s.replace("（", "(").replace("）", ")")
        return s.lower()

    a = norm(a)
    b = norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return float(SequenceMatcher(None, a, b).ratio())


def _iter_ocr_items(obj):
    """
    ocr_result.json の形式差を吸収して，text, bbox/poly, score を持つ候補を列挙する．

    対応例:
      - PaddleOCR: [[poly, (text, score)], ...]
      - PaddleX系: {"rec_texts": [...], "rec_scores": [...], "dt_polys": [...]}
      - 独自形式: {"text": ..., "bbox": ...}
    """
    if obj is None:
        return

    if isinstance(obj, dict):
        # PaddleX / PP-OCR 系でよくある並列配列形式
        text_keys = ("rec_texts", "texts", "text")
        poly_keys = ("dt_polys", "dt_boxes", "boxes", "polys", "points", "bbox")
        score_keys = ("rec_scores", "scores", "confidences")

        rec_texts = None
        for k in text_keys:
            v = obj.get(k)
            if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], str)):
                rec_texts = v
                break

        rec_polys = None
        for k in poly_keys:
            v = obj.get(k)
            if isinstance(v, list) and len(v) == (len(rec_texts) if rec_texts is not None else len(v)):
                # text が単一文字列の bbox ではなく，並列配列っぽいものだけ採用
                if rec_texts is not None:
                    rec_polys = v
                    break

        rec_scores = None
        for k in score_keys:
            v = obj.get(k)
            if isinstance(v, list):
                rec_scores = v
                break

        if rec_texts is not None and rec_polys is not None:
            for i, text in enumerate(rec_texts):
                score = None
                if rec_scores is not None and i < len(rec_scores):
                    try:
                        score = float(rec_scores[i])
                    except Exception:
                        score = None
                yield {"text": str(text), "bbox": rec_polys[i], "score": score, "source": "parallel_arrays"}

        # 独自形式: text + bbox/poly が同じdict内にある
        text = obj.get("text") or obj.get("rec_text") or obj.get("label") or obj.get("transcription")
        bbox = None
        for key in ("poly", "points", "bbox", "box", "dt_poly", "quadrilateral"):
            if key in obj:
                bbox = obj[key]
                break
        score = obj.get("score") or obj.get("confidence") or obj.get("rec_score")
        if text is not None and bbox is not None:
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            yield {"text": str(text), "bbox": bbox, "score": score, "source": "dict_item"}

        # ネストを再帰的に探索
        for key in ("results", "ocr_results", "data", "items", "pages", "res", "output"):
            if key in obj:
                yield from _iter_ocr_items(obj[key])

    elif isinstance(obj, (list, tuple)):
        # PaddleOCR: [box, (text, score)]
        if len(obj) >= 2:
            maybe_poly = _poly_from_any(obj[0])
            maybe_rec = obj[1]
            if maybe_poly is not None and isinstance(maybe_rec, (list, tuple)) and len(maybe_rec) >= 1:
                if isinstance(maybe_rec[0], str):
                    score = None
                    if len(maybe_rec) >= 2:
                        try:
                            score = float(maybe_rec[1])
                        except Exception:
                            score = None
                    yield {"text": maybe_rec[0], "bbox": maybe_poly, "score": score, "source": "paddle_pair"}
                    return

        for v in obj:
            yield from _iter_ocr_items(v)


def _polygon_overlap_ratio_with_mask(poly: np.ndarray, mask01: np.ndarray) -> float:
    """
    OCR polygon が選択SAMマスクとどの程度重なっているかを返す．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return 0.0
    h, w = mask01.shape[:2]
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts], 1)
    area = int(canvas.sum())
    if area <= 0:
        return 0.0
    overlap = int(((canvas > 0) & (mask01 > 0)).sum())
    return float(overlap / area)


def _load_best_ocr_polygon_for_mask(
    shot_dir: Path,
    query: str,
    mask01: np.ndarray,
    target_center: np.ndarray | None = None,
    max_debug_candidates: int = 30,
):
    """
    ocr_result.json から，queryに近く，かつ選択SAMマスク・merged boxに近いOCR polygonを選ぶ．

    重要:
      ocr_result.json のpolygon座標が after_init_rgb.png と90度ずれている場合があるため，
      identity / 90度CW逆変換 / 90度CCW逆変換 / 180度変換をすべて試し，
      選択SAMマスクとの重なりと merged box 中心への近さから最も整合する座標系を選ぶ．
    """
    ocr_path = Path(shot_dir) / "ocr_result.json"
    if not ocr_path.exists():
        return None, {"reason": f"ocr_result.json not found: {ocr_path}", "candidates": []}

    try:
        obj = json.loads(ocr_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, {"reason": f"failed to read ocr_result.json: {e}", "candidates": []}

    h, w = mask01.shape[:2]
    image_shape = (h, w)
    candidates = []

    for item in _iter_ocr_items(obj):
        text = item.get("text", "")
        raw_poly = _poly_from_any(item.get("bbox"))
        if raw_poly is None:
            continue

        sim = _text_similarity(query, text)
        rec_score = item.get("score")
        try:
            rec_score = float(rec_score) if rec_score is not None else 0.0
        except Exception:
            rec_score = 0.0

        for transform_mode in _OCR_POLY_TRANSFORM_MODES:
            poly = _transform_ocr_poly_to_rgb(raw_poly, image_shape, transform_mode)
            if poly is None:
                continue

            valid_score = _poly_image_valid_score(poly, image_shape)
            if valid_score <= 0.0:
                continue

            center, raw_axis, short_len, long_len = _polygon_center_and_axis(poly)
            if center is None:
                continue

            cx, cy = float(center[0]), float(center[1])
            in_image_center = (0 <= int(round(cx)) < w) and (0 <= int(round(cy)) < h)
            inside_center = bool(mask01[int(round(cy)), int(round(cx))] > 0) if in_image_center else False
            overlap = _polygon_overlap_ratio_with_mask(poly, mask01)
            center_score = _center_distance_score(center, target_center, image_shape)

            # query一致を最重視．そのうえで，マスクとの重なり，merged box中心への近さ，画像内妥当性を使う．
            rank_score = (
                2.0 * sim
                + 0.55 * float(inside_center)
                + 0.55 * float(overlap)
                + 0.45 * float(center_score)
                + 0.20 * float(valid_score)
                + 0.05 * float(rec_score)
            )

            candidates.append({
                "rank_score": float(rank_score),
                "text": str(text),
                "sim": float(sim),
                "inside_center": bool(inside_center),
                "overlap": float(overlap),
                "center_distance_score": float(center_score),
                "image_valid_score": float(valid_score),
                "score": float(rec_score),
                "poly": poly.astype(np.float32),
                "raw_poly": raw_poly.astype(np.float32),
                "center": np.asarray(center, dtype=np.float64),
                "axis": None if raw_axis is None else np.asarray(raw_axis, dtype=np.float64),
                "short_len": None if short_len is None else float(short_len),
                "long_len": None if long_len is None else float(long_len),
                "source": item.get("source", "unknown"),
                "transform_mode": transform_mode,
            })

    if not candidates:
        return None, {"reason": "no OCR polygon candidates", "candidates": []}

    candidates.sort(key=lambda d: d["rank_score"], reverse=True)
    best = candidates[0]

    # queryにもマスクにもmerged中心にも合わないOCRは使わない．
    if best["sim"] < 0.15 and best["overlap"] < 0.20 and not best["inside_center"] and best["center_distance_score"] < 0.20:
        dbg = []
        for c in candidates[:max_debug_candidates]:
            dbg.append({
                "text": c["text"],
                "rank_score": c["rank_score"],
                "sim": c["sim"],
                "inside_center": c["inside_center"],
                "overlap": c["overlap"],
                "center_distance_score": c["center_distance_score"],
                "transform_mode": c["transform_mode"],
                "source": c["source"],
            })
        return None, {"reason": "no reliable OCR polygon", "candidates": dbg}

    dbg = []
    for c in candidates[:max_debug_candidates]:
        dbg.append({
            "text": c["text"],
            "rank_score": c["rank_score"],
            "sim": c["sim"],
            "inside_center": c["inside_center"],
            "overlap": c["overlap"],
            "center_distance_score": c["center_distance_score"],
            "image_valid_score": c["image_valid_score"],
            "score": c["score"],
            "source": c["source"],
            "transform_mode": c["transform_mode"],
            "center": [float(c["center"][0]), float(c["center"][1])],
            "axis": None if c["axis"] is None else [float(c["axis"][0]), float(c["axis"][1])],
            "short_len": c["short_len"],
            "long_len": c["long_len"],
        })

    return best, {"reason": "ok", "candidates": dbg}

def _extract_merged_box_center(merged: dict, mask01: np.ndarray):
    """
    OCR/SAM統合結果の axis-aligned box から中心を取得する．
    中心位置は merged の box が比較的安定していたため，まずこれを使う．
    """
    results = merged.get("results", []) if isinstance(merged, dict) else []
    result = results[0] if results else {}
    box = result.get("box") if isinstance(result, dict) else None

    if isinstance(box, dict) and {"x1", "y1", "x2", "y2"}.issubset(set(box.keys())):
        try:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            return np.asarray([0.5 * (x1 + x2), 0.5 * (y1 + y2)], dtype=np.float64), {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        except Exception:
            pass

    # result に4点polygonがあれば中心だけ使う
    if isinstance(result, dict):
        for key in ("poly", "points", "bbox"):
            poly = _poly_from_any(result.get(key))
            if poly is not None:
                center, _, _, _ = _polygon_center_and_axis(poly)
                if center is not None:
                    return np.asarray(center, dtype=np.float64), None

    c = _mask_centroid(mask01)
    return c, None


def _filter_small_components(mask01: np.ndarray, min_area_ratio: float = 0.003):
    """
    帯補正後に出る小さな孤立領域を除去する．
    削りすぎ防止のため，極小成分のみを対象にする．
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    area = int(mask01.sum())
    if area <= 0:
        return mask01, {"enabled": False, "reason": "empty mask"}

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 2:
        return mask01, {"enabled": True, "num_components": max(0, num_labels - 1), "removed_area": 0}

    min_area = max(10, int(round(area * float(min_area_ratio))))
    out = np.zeros_like(mask01, dtype=np.uint8)
    removed_area = 0
    kept_components = 0
    for label in range(1, num_labels):
        comp_area = int(stats[label, cv2.CC_STAT_AREA])
        if comp_area >= min_area:
            out[labels == label] = 1
            kept_components += 1
        else:
            removed_area += comp_area

    if int(out.sum()) <= 0:
        return mask01, {
            "enabled": True,
            "num_components": num_labels - 1,
            "removed_area": 0,
            "reason": "all components would be removed; reverted",
        }

    return out, {
        "enabled": True,
        "num_components": num_labels - 1,
        "kept_components": kept_components,
        "min_area": int(min_area),
        "removed_area": int(removed_area),
    }



def _suppress_side_protrusions_axis_profile(
    mask01: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    bin_size_px: float = 8.0,
    slice_width_percentiles: tuple[float, float] = (10.0, 90.0),
    typical_width_percentile: float = 55.0,
    profile_width_margin: float = 1.20,
    min_profile_half_width_px: float = 18.0,
    min_bin_pixels: int = 12,
    max_profile_shrink_ratio: float = 0.30,
):
    """
    OCR軸帯だけでは残る「側面の張り出し」を，背表紙方向の断面幅から控えめに削る．

    考え方:
      - 背表紙方向を u，それに直交する方向を v とする．
      - u方向に細かくスライスし，各スライス内の v 方向の幅を求める．
      - 背表紙の大部分では幅が安定する一方，側面が混入したスライスだけ幅が大きくなる．
      - そこで，全スライスの典型幅を基準として，各スライスの中心線周辺だけを残す．

    削りすぎ防止:
      - この処理だけで max_profile_shrink_ratio を超えて面積が減る場合は元に戻す．
    """
    mask01 = (mask01 > 0).astype(np.uint8)
    area_before = int(mask01.sum())
    info = {
        "enabled": True,
        "area_before": area_before,
        "bin_size_px": float(bin_size_px),
        "slice_width_percentiles": [float(slice_width_percentiles[0]), float(slice_width_percentiles[1])],
        "typical_width_percentile": float(typical_width_percentile),
        "profile_width_margin": float(profile_width_margin),
        "min_profile_half_width_px": float(min_profile_half_width_px),
        "min_bin_pixels": int(min_bin_pixels),
        "max_profile_shrink_ratio": float(max_profile_shrink_ratio),
    }

    if area_before <= 0:
        info.update({"used": False, "reason": "empty mask"})
        return mask01, info

    axis = np.asarray(axis, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        info.update({"used": False, "reason": "invalid axis"})
        return mask01, info
    axis = axis / n
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    ys, xs = np.where(mask01 > 0)
    if len(xs) < 50:
        info.update({"used": False, "reason": "too few pixels"})
        return mask01, info

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    u = rel @ axis
    v = rel @ normal

    u_min = float(np.min(u))
    bin_size_px = max(2.0, float(bin_size_px))
    bins = np.floor((u - u_min) / bin_size_px).astype(np.int32)
    num_bins = int(bins.max()) + 1

    valid_bin_ids = []
    bin_centers_v = []
    bin_widths_v = []

    low_p, high_p = float(slice_width_percentiles[0]), float(slice_width_percentiles[1])
    for b in range(num_bins):
        vv = v[bins == b]
        if vv.size < int(min_bin_pixels):
            continue
        q_low, q_high = np.percentile(vv, [low_p, high_p])
        width = float(q_high - q_low)
        if width <= 1.0:
            continue
        valid_bin_ids.append(b)
        # 中心は中央値を使う．側面の張り出しがあっても平均より引っ張られにくい．
        bin_centers_v.append(float(np.median(vv)))
        bin_widths_v.append(width)

    if len(valid_bin_ids) < 5:
        info.update({"used": False, "reason": "too few valid slices", "num_valid_slices": int(len(valid_bin_ids))})
        return mask01, info

    valid_bin_ids = np.asarray(valid_bin_ids, dtype=np.float64)
    bin_centers_v = np.asarray(bin_centers_v, dtype=np.float64)
    bin_widths_v = np.asarray(bin_widths_v, dtype=np.float64)

    typical_width = float(np.percentile(bin_widths_v, float(typical_width_percentile)))
    if typical_width <= 1.0:
        info.update({"used": False, "reason": "invalid typical width"})
        return mask01, info

    half_clip = max(float(min_profile_half_width_px), 0.5 * typical_width * float(profile_width_margin))

    # 各画素のuスライスに対応する中心線位置を線形補間で求める．
    # 端では最近傍の中心を使う．
    center_v_for_pixel = np.interp(
        bins.astype(np.float64),
        valid_bin_ids,
        bin_centers_v,
        left=float(bin_centers_v[0]),
        right=float(bin_centers_v[-1]),
    )

    keep = np.abs(v - center_v_for_pixel) <= half_clip
    out = np.zeros_like(mask01, dtype=np.uint8)
    out[ys[keep], xs[keep]] = 1

    # 極小成分は最後に削る．ただしここでも削りすぎない．
    out, comp_info = _filter_small_components(out, min_area_ratio=0.002)

    area_after = int(out.sum())
    remain_ratio = float(area_after / max(area_before, 1))
    shrink_ratio = float(1.0 - remain_ratio)

    info.update({
        "used": True,
        "reason": "ok",
        "area_after": area_after,
        "remain_ratio": remain_ratio,
        "shrink_ratio": shrink_ratio,
        "num_valid_slices": int(len(valid_bin_ids)),
        "typical_width_px": typical_width,
        "half_clip_px": float(half_clip),
        "slice_width_min_px": float(np.min(bin_widths_v)),
        "slice_width_median_px": float(np.median(bin_widths_v)),
        "slice_width_max_px": float(np.max(bin_widths_v)),
        "component_filter": comp_info,
    })

    if area_after <= 0:
        info.update({"used": False, "reason": "all removed; reverted", "area_after": area_before, "remain_ratio": 1.0, "shrink_ratio": 0.0})
        return mask01, info

    if shrink_ratio > float(max_profile_shrink_ratio):
        info.update({
            "used": False,
            "reason": f"profile suppression removed too much: shrink_ratio={shrink_ratio:.3f}",
            "area_after": area_before,
            "remain_ratio": 1.0,
            "shrink_ratio": 0.0,
        })
        return mask01, info

    return out, info

def refine_mask_by_ocr_axis_band(
    mask01: np.ndarray,
    merged: dict,
    image_shape: tuple[int, int],
    shot_dir: str | Path | None = None,
    query: str | None = None,
    mask_width_ratio: float = 0.90,
    min_keep_ratio: float = 0.65,
    width_percentiles: tuple[float, float] = (5.0, 95.0),
    remove_small_components: bool = True,
    small_component_area_ratio: float = 0.003,
    use_ocr_short_width: bool = True,
    ocr_short_to_half_width_scale: float = 1.35,
    min_ocr_half_width_px: float = 28.0,
    suppress_side_protrusions: bool = False,
    profile_bin_size_px: float = 8.0,
    profile_width_margin: float = 1.20,
    min_profile_half_width_px: float = 18.0,
    max_profile_shrink_ratio: float = 0.30,
    return_info: bool = False,
):
    """
    SAM2マスクから，OCR中心線に沿う帯領域を控えめに残す．

    今回の反省を反映した設計:
      1. merged box は中心位置として使う．
      2. ocr_result.json の polygon は，RGB画像と90度ずれていることがあるため，複数の回転補正を自動試行する．
      3. OCR polygon の長辺方向が背表紙方向と90度ずれる場合があるため，SAMマスクPCA主軸に近い方向を選ぶ．
      4. polygonが使えない場合は SAMマスクPCA主軸を使う．
      5. forced_angle は最後のフォールバックとしてのみ使う．
      6. 帯幅は，OCR文字領域の短辺幅を優先して決める．文字領域の長さには依存しない．
      7. OCR短辺幅が使えない場合のみ，SAMマスクの幅から決める．
      8. 局所スライスごとの側面抑制は斜め削りの原因になりやすいため，デフォルトでは使わない．
      9. 残存率が低すぎる場合は元マスクへ戻す．
    """
    h, w = image_shape
    mask01 = (mask01 > 0).astype(np.uint8)
    mask_area = int(mask01.sum())

    info = {
        "used": False,
        "reason": "",
        "mask_area_before": mask_area,
        "mask_width_ratio": float(mask_width_ratio),
        "min_keep_ratio": float(min_keep_ratio),
        "width_percentiles": [float(width_percentiles[0]), float(width_percentiles[1])],
        "use_ocr_short_width": bool(use_ocr_short_width),
        "ocr_short_to_half_width_scale": float(ocr_short_to_half_width_scale),
        "min_ocr_half_width_px": float(min_ocr_half_width_px),
        "suppress_side_protrusions": bool(suppress_side_protrusions),
        "profile_bin_size_px": float(profile_bin_size_px),
        "profile_width_margin": float(profile_width_margin),
        "min_profile_half_width_px": float(min_profile_half_width_px),
        "max_profile_shrink_ratio": float(max_profile_shrink_ratio),
    }

    if mask_area <= 0:
        info["reason"] = "empty mask"
        return (mask01, info) if return_info else mask01

    results = merged.get("results", []) if isinstance(merged, dict) else []
    result = results[0] if results else {}

    center, merged_box = _extract_merged_box_center(merged, mask01)
    if center is None:
        info["reason"] = "failed to determine center"
        return (mask01, info) if return_info else mask01

    axis = None
    axis_source = None
    ocr_best = None
    ocr_debug = None

    # SAMマスク主軸は，OCR polygonの長辺/短辺のどちらを使うか判定するためにも使う．
    mask_axis = _mask_pca_axis(mask01)

    # ===== 1) ocr_result.json の4点polygonを最優先 =====
    if shot_dir is not None and query is not None:
        ocr_best, ocr_debug = _load_best_ocr_polygon_for_mask(
            shot_dir=Path(shot_dir),
            query=str(query),
            mask01=mask01,
            target_center=center,
        )
        info["ocr_polygon_search"] = ocr_debug
        if ocr_best is not None and ocr_best.get("axis") is not None:
            raw_ocr_axis = np.asarray(ocr_best["axis"], dtype=np.float64)
            axis, axis_kind = _choose_axis_consistent_with_mask(raw_ocr_axis, mask_axis)
            axis_source = f"ocr_result_polygon_{axis_kind}"
            # 中心は merged bbox を優先するが，mergedが使えないときのみOCR中心を使う．
            if merged_box is None:
                center = np.asarray(ocr_best["center"], dtype=np.float64)

    # ===== 2) merged resultにpolygonが含まれていれば使う =====
    if axis is None and isinstance(result, dict):
        for key in ("poly", "points", "bbox"):
            poly = _poly_from_any(result.get(key))
            if poly is not None:
                c_poly, axis_poly, short_len, long_len = _polygon_center_and_axis(poly)
                if axis_poly is not None:
                    axis, axis_kind = _choose_axis_consistent_with_mask(axis_poly, mask_axis)
                    axis = np.asarray(axis, dtype=np.float64)
                    axis_source = f"merged_{key}_polygon_{axis_kind}"
                    break

    # ===== 3) SAMマスクのPCA主軸 =====
    if axis is None and mask_axis is not None:
        axis = np.asarray(mask_axis, dtype=np.float64)
        axis_source = "mask_pca"

    # ===== 4) 最後のフォールバックとしてだけ forced_angle / bbox縦横比 =====
    if axis is None:
        forced_angle = result.get("forced_angle", None) if isinstance(result, dict) else None
        if forced_angle is not None:
            angle_rad = np.deg2rad(float(forced_angle))
            axis_source = "forced_angle_fallback"
        elif merged_box is not None:
            bw = abs(float(merged_box["x2"]) - float(merged_box["x1"]))
            bh = abs(float(merged_box["y2"]) - float(merged_box["y1"]))
            angle_rad = np.pi / 2.0 if bh >= bw else 0.0
            axis_source = "bbox_aspect_fallback"
        else:
            info["reason"] = "failed to determine axis"
            return (mask01, info) if return_info else mask01
        axis = np.asarray([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float64)

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        info["reason"] = "invalid axis"
        return (mask01, info) if return_info else mask01
    axis = axis / axis_norm

    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)

    ys, xs = np.where(mask01 > 0)
    if len(xs) < 20:
        info["reason"] = "too few mask pixels"
        return (mask01, info) if return_info else mask01

    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    normal_coords = (pts - center) @ normal

    p_low, p_high = float(width_percentiles[0]), float(width_percentiles[1])
    q_low, q_high = np.percentile(normal_coords, [p_low, p_high])
    mask_width_px = float(q_high - q_low)
    if mask_width_px <= 1.0:
        info["reason"] = "invalid mask width"
        return (mask01, info) if return_info else mask01

    # ===== 帯幅の決定 =====
    # これまでの問題点:
    #   - SAMマスク幅だけで帯幅を決めると，側面部が混入した分だけ帯が広がり，側面が残る．
    #   - OCR文字領域の「長さ」は本全体を覆わないことがあるため信用しない．
    # 改善方針:
    #   - OCR文字領域の短辺幅は比較的安定しているため，これを背表紙幅の手掛かりにする．
    #   - OCR短辺幅から決めた帯を，本の主軸方向へ無限に延長して使う．
    #   - これにより，OCR文字領域の長さが短くても，本全体の上下方向は削らない．
    half_width_from_mask_px = 0.5 * mask_width_px * float(mask_width_ratio)

    ocr_short_len_px = None
    if isinstance(ocr_best, dict) and ocr_best.get("short_len") is not None:
        try:
            ocr_short_len_px = float(ocr_best.get("short_len"))
        except Exception:
            ocr_short_len_px = None

    half_width_from_ocr_px = None
    if bool(use_ocr_short_width) and ocr_short_len_px is not None and ocr_short_len_px > 1.0:
        half_width_from_ocr_px = max(
            float(min_ocr_half_width_px),
            float(ocr_short_len_px) * float(ocr_short_to_half_width_scale),
        )
        # OCR短辺幅は側面混入の影響を受けにくいので，SAM幅由来の値より小さい場合は優先する．
        # ただし，OCR幅が大きすぎる場合はSAM幅由来の安全上限で抑える．
        half_width_px = min(float(half_width_from_mask_px), float(half_width_from_ocr_px))
        band_width_source = "ocr_short_width_limited_by_mask_width"
    else:
        half_width_px = float(half_width_from_mask_px)
        band_width_source = "mask_width"

    keep_pts = np.abs(normal_coords) <= half_width_px

    refined = np.zeros_like(mask01, dtype=np.uint8)
    refined[ys[keep_pts], xs[keep_pts]] = 1

    component_info = None
    if remove_small_components:
        refined, component_info = _filter_small_components(
            refined,
            min_area_ratio=float(small_component_area_ratio),
        )

    side_suppression_info = None
    if suppress_side_protrusions:
        refined, side_suppression_info = _suppress_side_protrusions_axis_profile(
            refined,
            axis=axis,
            center=center,
            bin_size_px=float(profile_bin_size_px),
            profile_width_margin=float(profile_width_margin),
            min_profile_half_width_px=float(min_profile_half_width_px),
            max_profile_shrink_ratio=float(max_profile_shrink_ratio),
        )

    refined_area = int(refined.sum())
    remain_ratio = float(refined_area / max(mask_area, 1))

    info.update({
        "used": True,
        "reason": "ok",
        "axis_source": axis_source,
        "merged_box": merged_box,
        "center_source": "merged_box" if merged_box is not None else "fallback_center",
        "ocr_center": [float(center[0]), float(center[1])],
        "axis": [float(axis[0]), float(axis[1])],
        "normal": [float(normal[0]), float(normal[1])],
        "angle_deg": float(np.degrees(np.arctan2(axis[1], axis[0]))),
        "mask_pca_axis": None if mask_axis is None else [float(mask_axis[0]), float(mask_axis[1])],
        "mask_width_px": float(mask_width_px),
        "half_width_from_mask_px": float(half_width_from_mask_px),
        "ocr_short_len_px": None if ocr_short_len_px is None else float(ocr_short_len_px),
        "half_width_from_ocr_px": None if half_width_from_ocr_px is None else float(half_width_from_ocr_px),
        "band_width_source": band_width_source,
        "half_width_px": float(half_width_px),
        "mask_area_after": refined_area,
        "remain_ratio": remain_ratio,
        "component_filter": component_info,
        "side_suppression": side_suppression_info,
    })

    if ocr_best is not None:
        info["selected_ocr_polygon"] = {
            "text": ocr_best.get("text"),
            "rank_score": float(ocr_best.get("rank_score", 0.0)),
            "sim": float(ocr_best.get("sim", 0.0)),
            "inside_center": bool(ocr_best.get("inside_center", False)),
            "overlap": float(ocr_best.get("overlap", 0.0)),
            "center_distance_score": float(ocr_best.get("center_distance_score", 0.0)),
            "image_valid_score": float(ocr_best.get("image_valid_score", 0.0)),
            "source": ocr_best.get("source", "unknown"),
            "transform_mode": ocr_best.get("transform_mode", "unknown"),
            "center": [float(ocr_best["center"][0]), float(ocr_best["center"][1])],
            "axis_raw_from_polygon": None if ocr_best.get("axis") is None else [float(ocr_best["axis"][0]), float(ocr_best["axis"][1])],
            "poly": np.asarray(ocr_best["poly"], dtype=np.float32).tolist(),
            "raw_poly": np.asarray(ocr_best["raw_poly"], dtype=np.float32).tolist() if ocr_best.get("raw_poly") is not None else None,
        }

    # 削りすぎたら補正を破棄
    if remain_ratio < float(min_keep_ratio):
        info["used"] = False
        info["reason"] = f"too much removed: remain_ratio={remain_ratio:.3f}"
        info["mask_area_after"] = mask_area
        info["remain_ratio"] = 1.0
        return (mask01, info) if return_info else mask01

    return (refined, info) if return_info else refined


def _longest_occupied_s_segment(
    s_values: np.ndarray,
    s_origin: float,
    s_bin_size_px: float,
    gap_allow_bins: int,
):
    """
    1本のt列に含まれる点のs座標から，gapを少し許容した最長連続区間を返す．
    戻り値は dict または None．
    """
    s_values = np.asarray(s_values, dtype=np.float64).reshape(-1)
    if s_values.size <= 0:
        return None

    s_bin_size_px = max(float(s_bin_size_px), 1.0)
    gap_allow_bins = max(int(gap_allow_bins), 0)

    bins = np.floor((s_values - float(s_origin)) / s_bin_size_px).astype(np.int32)
    if bins.size <= 0:
        return None

    uniq = np.unique(bins)
    if uniq.size <= 0:
        return None

    best_start = int(uniq[0])
    best_end = int(uniq[0])
    cur_start = int(uniq[0])
    cur_end = int(uniq[0])

    for b in uniq[1:]:
        b = int(b)
        # 例: gap_allow_bins=2 なら，2binまでの穴は同じ連続区間とみなす．
        if b - cur_end <= gap_allow_bins + 1:
            cur_end = b
        else:
            if (cur_end - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = b

    if (cur_end - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, cur_end

    s_min = float(s_origin + best_start * s_bin_size_px)
    s_max = float(s_origin + (best_end + 1) * s_bin_size_px)
    length_px = float(max(0.0, s_max - s_min))

    # 最長区間内に実際にある点数も数える．
    in_run = (bins >= best_start) & (bins <= best_end)

    return {
        "start_bin": int(best_start),
        "end_bin": int(best_end),
        "s_min": s_min,
        "s_max": s_max,
        "length_px": length_px,
        "point_count_in_run": int(np.count_nonzero(in_run)),
        "occupied_bin_count": int(uniq.size),
    }


def _find_consecutive_groups(indices: np.ndarray):
    """昇順整数配列を連続グループに分ける．"""
    indices = np.asarray(indices, dtype=np.int32).reshape(-1)
    if indices.size == 0:
        return []
    indices = np.unique(indices)
    groups = []
    start = int(indices[0])
    prev = int(indices[0])
    for v in indices[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
        else:
            groups.append((start, prev))
            start = prev = v
    groups.append((start, prev))
    return groups


def refine_mask_by_spine_column_length_after_depth(
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    refine_info: dict | None,
    image_shape: tuple[int, int],
    *,
    t_bin_size_px: float = 4.0,
    s_bin_size_px: float = 5.0,
    s_gap_allow_px: float = 18.0,
    length_reference_percentile: float = 85.0,
    min_length_ratio: float = 0.65,
    relaxed_edge_length_ratio: float = 0.55,
    min_points_per_t_bin: int = 12,
    min_selected_t_bins: int = 2,
    expand_selected_t_bins: int = 0,
    s_margin_px: float = 8.0,
    min_valid_keep_ratio: float = 0.30,
    return_info: bool = False,
):
    """
    Depth中央値±3cm補正後の点群列長さを使って，側面・局所張り出しを削る．

    考え方:
      1. depth_masked > 0 の点だけを，実際に点群化される有効点として扱う．
      2. OCR/SAMから得た背表紙方向axisをs軸，直交方向をt軸とする．
      3. t方向に細かくbin分割し，各t列についてs方向の最長連続長を求める．
      4. 十分な長さを持つt列だけを候補にする．
      5. OCR seedのt座標に近い連続したt列グループだけを残す．

    幅そのものは固定しない．各書籍ごとに「背表紙方向に長い点群列が存在する範囲」を幅として間接的に求める．
    """
    h, w = image_shape
    mask01 = (mask01 > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)
    valid = (mask01 > 0) & (depth_masked > 0)
    valid_count = int(np.count_nonzero(valid))

    info = {
        "used": False,
        "reason": "",
        "valid_count_before": valid_count,
        "t_bin_size_px": float(t_bin_size_px),
        "s_bin_size_px": float(s_bin_size_px),
        "s_gap_allow_px": float(s_gap_allow_px),
        "length_reference_percentile": float(length_reference_percentile),
        "min_length_ratio": float(min_length_ratio),
        "relaxed_edge_length_ratio": float(relaxed_edge_length_ratio),
        "min_points_per_t_bin": int(min_points_per_t_bin),
        "min_selected_t_bins": int(min_selected_t_bins),
        "expand_selected_t_bins": int(expand_selected_t_bins),
        "s_margin_px": float(s_margin_px),
        "min_valid_keep_ratio": float(min_valid_keep_ratio),
    }

    if valid_count < 50:
        info["reason"] = "too few valid depth points"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    rinfo = refine_info or {}
    axis = rinfo.get("axis", None)
    center = rinfo.get("ocr_center", None)

    if axis is None:
        axis = _mask_pca_axis(mask01)
        info["axis_source"] = "mask_pca_fallback"
    else:
        info["axis_source"] = str(rinfo.get("axis_source", "refine_info_axis"))

    if center is None:
        center = _mask_centroid(mask01)
        info["center_source"] = "mask_centroid_fallback"
    else:
        info["center_source"] = "ocr_center"

    if axis is None or center is None:
        info["reason"] = "axis or center is unavailable"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    axis = np.asarray(axis, dtype=np.float64).reshape(2)
    an = float(np.linalg.norm(axis))
    if an < 1e-9:
        info["reason"] = "invalid axis"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)
    axis = axis / an
    normal = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(2)

    ys, xs = np.where(valid)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    s_coords = rel @ axis
    t_coords = rel @ normal

    t_bin_size_px = max(float(t_bin_size_px), 1.0)
    s_bin_size_px = max(float(s_bin_size_px), 1.0)
    gap_allow_bins = int(round(float(s_gap_allow_px) / s_bin_size_px))

    t_min_all = float(np.min(t_coords))
    t_max_all = float(np.max(t_coords))
    s_min_all = float(np.min(s_coords))
    s_max_all = float(np.max(s_coords))
    n_t_bins = int(np.floor((t_max_all - t_min_all) / t_bin_size_px)) + 1

    if n_t_bins < 1:
        info["reason"] = "invalid t bin count"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    t_bins = np.floor((t_coords - t_min_all) / t_bin_size_px).astype(np.int32)
    t_bins = np.clip(t_bins, 0, n_t_bins - 1)

    column_records = []
    lengths = []
    counts = []

    for b in range(n_t_bins):
        idx = np.where(t_bins == b)[0]
        count = int(idx.size)
        record = {
            "t_bin": int(b),
            "t_min": float(t_min_all + b * t_bin_size_px),
            "t_max": float(t_min_all + (b + 1) * t_bin_size_px),
            "point_count": count,
            "length_px": 0.0,
            "s_min": None,
            "s_max": None,
            "is_good": False,
            "is_relaxed_good": False,
        }
        if count >= int(min_points_per_t_bin):
            seg = _longest_occupied_s_segment(
                s_coords[idx],
                s_origin=s_min_all,
                s_bin_size_px=s_bin_size_px,
                gap_allow_bins=gap_allow_bins,
            )
            if seg is not None:
                record.update({
                    "length_px": float(seg["length_px"]),
                    "s_min": float(seg["s_min"]),
                    "s_max": float(seg["s_max"]),
                    "start_bin": int(seg["start_bin"]),
                    "end_bin": int(seg["end_bin"]),
                    "point_count_in_run": int(seg["point_count_in_run"]),
                    "occupied_bin_count": int(seg["occupied_bin_count"]),
                })
        column_records.append(record)
        lengths.append(float(record["length_px"]))
        counts.append(count)

    lengths_np = np.asarray(lengths, dtype=np.float64)
    positive_lengths = lengths_np[lengths_np > 0]
    if positive_lengths.size < 2:
        info["reason"] = "too few positive length columns"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    ref_percentile = float(np.clip(length_reference_percentile, 50.0, 100.0))
    reference_length = float(np.percentile(positive_lengths, ref_percentile))
    max_length = float(np.max(positive_lengths))
    if reference_length <= 1.0:
        info["reason"] = "invalid reference length"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    length_threshold = float(reference_length * float(min_length_ratio))
    relaxed_threshold = float(reference_length * float(relaxed_edge_length_ratio))

    good_bins = []
    relaxed_bins = []
    for rec in column_records:
        if rec["point_count"] >= int(min_points_per_t_bin) and rec["length_px"] >= length_threshold:
            rec["is_good"] = True
            good_bins.append(int(rec["t_bin"]))
        if rec["point_count"] >= int(min_points_per_t_bin) and rec["length_px"] >= relaxed_threshold:
            rec["is_relaxed_good"] = True
            relaxed_bins.append(int(rec["t_bin"]))

    good_bins = np.asarray(good_bins, dtype=np.int32)
    if good_bins.size == 0:
        info.update({
            "reason": "no long columns found",
            "reference_length_px": reference_length,
            "max_length_px": max_length,
            "length_threshold_px": length_threshold,
        })
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    seed_t = float((center - center) @ normal)  # center自身なので0．明示のため残す．
    seed_bin = int(np.floor((seed_t - t_min_all) / t_bin_size_px))
    seed_bin = int(np.clip(seed_bin, 0, n_t_bins - 1))

    groups = _find_consecutive_groups(good_bins)
    if not groups:
        info["reason"] = "failed to make good-bin groups"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    # OCR seed に最も近い長い列グループを採用する．
    selected_group = None
    best_group_score = -1e18
    for g0, g1 in groups:
        if g0 <= seed_bin <= g1:
            dist = 0
        elif seed_bin < g0:
            dist = g0 - seed_bin
        else:
            dist = seed_bin - g1
        width_bins = g1 - g0 + 1
        mean_len = float(np.mean(lengths_np[g0:g1 + 1]))
        # seedから近く，幅があり，長いグループを優先する．
        score = -10.0 * float(dist) + 0.5 * float(width_bins) + 0.01 * mean_len
        if score > best_group_score:
            best_group_score = score
            selected_group = (int(g0), int(g1))

    if selected_group is None:
        info["reason"] = "failed to select t group"
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    g0, g1 = selected_group

    # 境界だけは少し緩めの列を取り込めるようにする．
    relaxed_set = set(int(v) for v in relaxed_bins)
    for _ in range(max(int(expand_selected_t_bins), 0)):
        if g0 - 1 in relaxed_set:
            g0 -= 1
        if g1 + 1 in relaxed_set:
            g1 += 1

    selected_bin_count = int(g1 - g0 + 1)
    if selected_bin_count < int(min_selected_t_bins):
        info.update({
            "reason": "selected t group is too narrow",
            "selected_group": [int(g0), int(g1)],
            "selected_bin_count": selected_bin_count,
        })
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    keep_point = np.zeros_like(t_bins, dtype=bool)
    s_margin_px = max(float(s_margin_px), 0.0)

    for b in range(g0, g1 + 1):
        rec = column_records[b]
        if rec.get("s_min") is None or rec.get("s_max") is None:
            continue
        idx = np.where(t_bins == b)[0]
        if idx.size == 0:
            continue
        s0 = float(rec["s_min"]) - s_margin_px
        s1 = float(rec["s_max"]) + s_margin_px
        keep_point[idx] = (s_coords[idx] >= s0) & (s_coords[idx] <= s1)

    mask_after = np.zeros_like(mask01, dtype=np.uint8)
    mask_after[ys[keep_point], xs[keep_point]] = 1

    # 小さな孤立成分は除去する．ただし主たる背表紙成分まで削らないよう，弱めにする．
    mask_after, component_info = _filter_small_components(mask_after, min_area_ratio=0.001)

    depth_after = depth_masked.copy()
    depth_after[mask_after == 0] = 0

    valid_after_count = int(np.count_nonzero(depth_after > 0))
    valid_keep_ratio = float(valid_after_count / max(valid_count, 1))

    info.update({
        "used": True,
        "reason": "ok",
        "axis": [float(axis[0]), float(axis[1])],
        "normal": [float(normal[0]), float(normal[1])],
        "center": [float(center[0]), float(center[1])],
        "t_min_all": float(t_min_all),
        "t_max_all": float(t_max_all),
        "s_min_all": float(s_min_all),
        "s_max_all": float(s_max_all),
        "n_t_bins": int(n_t_bins),
        "gap_allow_bins": int(gap_allow_bins),
        "reference_length_px": float(reference_length),
        "max_length_px": float(max_length),
        "length_threshold_px": float(length_threshold),
        "relaxed_threshold_px": float(relaxed_threshold),
        "good_bins": [int(v) for v in good_bins.tolist()],
        "groups": [[int(a), int(b)] for a, b in groups],
        "seed_t": float(seed_t),
        "seed_bin": int(seed_bin),
        "selected_group": [int(g0), int(g1)],
        "selected_bin_count": int(selected_bin_count),
        "selected_t_min": float(t_min_all + g0 * t_bin_size_px),
        "selected_t_max": float(t_min_all + (g1 + 1) * t_bin_size_px),
        "valid_count_after": int(valid_after_count),
        "valid_keep_ratio": float(valid_keep_ratio),
        "component_filter": component_info,
        # JSONが巨大になりすぎないよう，column_recordsは必要情報のみに制限する．
        "column_records": [
            {
                "t_bin": int(r["t_bin"]),
                "point_count": int(r["point_count"]),
                "length_px": float(r["length_px"]),
                "s_min": None if r.get("s_min") is None else float(r["s_min"]),
                "s_max": None if r.get("s_max") is None else float(r["s_max"]),
                "is_good": bool(r.get("is_good", False)),
                "is_relaxed_good": bool(r.get("is_relaxed_good", False)),
            }
            for r in column_records
        ],
    })

    if valid_keep_ratio < float(min_valid_keep_ratio):
        info["used"] = False
        info["reason"] = f"too much valid depth removed: valid_keep_ratio={valid_keep_ratio:.3f}"
        info["valid_count_after"] = valid_count
        info["valid_keep_ratio"] = 1.0
        return (mask01, depth_masked, info) if return_info else (mask01, depth_masked)

    return (mask_after, depth_after, info) if return_info else (mask_after, depth_after)


def save_spine_column_length_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    depth_before: np.ndarray,
    depth_after: np.ndarray,
    column_info: dict | None,
    stem: str,
):
    """Depth補正後の主成分方向点群列フィルタのデバッグ画像を保存する．"""
    debug_dir = Path(shot_dir) / "debug_ocr_band"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_before = (mask_before > 0).astype(np.uint8)
    mask_after = (mask_after > 0).astype(np.uint8)
    valid_before = ((mask_before > 0) & (np.asarray(depth_before) > 0)).astype(np.uint8)
    valid_after = ((mask_after > 0) & (np.asarray(depth_after) > 0)).astype(np.uint8)

    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_mask_before.png"), mask_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_mask_after.png"), mask_after * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_valid_before.png"), valid_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_valid_after.png"), valid_after * 255)

    removed = ((valid_before == 1) & (valid_after == 0)).astype(np.uint8)
    kept = ((valid_before == 1) & (valid_after == 1)).astype(np.uint8)

    overlay = color_np.copy()
    overlay[kept == 1] = (0, 255, 0)
    overlay[removed == 1] = (0, 0, 255)
    blended = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_overlay_kept_removed.png"), blended)

    # 軸，選択t列境界，補正後輪郭を描く．
    axis_img = color_np.copy()
    info = column_info or {}
    center = info.get("center")
    axis = info.get("axis")
    normal = info.get("normal")
    selected_t_min = info.get("selected_t_min")
    selected_t_max = info.get("selected_t_max")
    if center is not None and axis is not None:
        cx, cy = float(center[0]), float(center[1])
        ax, ay = float(axis[0]), float(axis[1])
        length = 1200.0
        p1 = (int(round(cx - ax * length)), int(round(cy - ay * length)))
        p2 = (int(round(cx + ax * length)), int(round(cy + ay * length)))
        cv2.line(axis_img, p1, p2, (0, 0, 255), 2)
        cv2.circle(axis_img, (int(round(cx)), int(round(cy))), 6, (255, 0, 0), -1)

        if normal is not None and selected_t_min is not None and selected_t_max is not None:
            nx, ny = float(normal[0]), float(normal[1])
            for tval in (float(selected_t_min), float(selected_t_max)):
                ox = nx * tval
                oy = ny * tval
                q1 = (int(round(cx + ox - ax * length)), int(round(cy + oy - ay * length)))
                q2 = (int(round(cx + ox + ax * length)), int(round(cy + oy + ay * length)))
                cv2.line(axis_img, q1, q2, (255, 0, 255), 2)

    contours, _ = cv2.findContours(mask_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(axis_img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_spine_column_axis_selected.png"), axis_img)

    # t列ごとの長さプロファイルを簡易グラフで保存する．
    records = info.get("column_records", [])
    if records:
        graph_w, graph_h = 1000, 320
        margin_l, margin_r, margin_t, margin_b = 50, 20, 20, 45
        graph = np.full((graph_h, graph_w, 3), 255, dtype=np.uint8)
        plot_w = graph_w - margin_l - margin_r
        plot_h = graph_h - margin_t - margin_b
        max_len = max(float(info.get("max_length_px", 1.0)), 1.0)
        n = len(records)

        # 閾値線
        th = float(info.get("length_threshold_px", 0.0))
        y_th = int(round(margin_t + plot_h * (1.0 - min(th / max_len, 1.0))))
        cv2.line(graph, (margin_l, y_th), (graph_w - margin_r, y_th), (0, 0, 255), 1)
        cv2.putText(graph, "length threshold", (margin_l + 5, max(15, y_th - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        selected = info.get("selected_group", None)
        for i, rec in enumerate(records):
            x0 = int(round(margin_l + plot_w * i / max(n, 1)))
            x1 = int(round(margin_l + plot_w * (i + 1) / max(n, 1)))
            length_px = float(rec.get("length_px", 0.0))
            bar_h = int(round(plot_h * min(length_px / max_len, 1.0)))
            y0 = margin_t + plot_h - bar_h
            y1 = margin_t + plot_h
            color = (180, 180, 180)
            if bool(rec.get("is_relaxed_good", False)):
                color = (180, 220, 180)
            if bool(rec.get("is_good", False)):
                color = (0, 180, 0)
            if selected is not None and int(selected[0]) <= i <= int(selected[1]):
                color = (0, 120, 255)
            cv2.rectangle(graph, (x0, y0), (max(x0 + 1, x1 - 1), y1), color, -1)

        cv2.line(graph, (margin_l, margin_t), (margin_l, margin_t + plot_h), (0, 0, 0), 1)
        cv2.line(graph, (margin_l, margin_t + plot_h), (graph_w - margin_r, margin_t + plot_h), (0, 0, 0), 1)
        cv2.putText(graph, "t-bin", (graph_w // 2 - 30, graph_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(graph, "s length", (5, margin_t + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imwrite(str(debug_dir / f"{stem}_spine_column_length_profile.png"), graph)

    save_json(debug_dir / f"{stem}_spine_column_log.json", info)
    print(f"✔ Saved spine-column debug files: {debug_dir}")


def _project_points_to_pixels(pts_f: np.ndarray, intr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    カメラ座標点群 (x,y,z) を RGB 画像座標 (u,v) に投影する．
    戻り値: u, v, valid_z
    """
    pts = np.asarray(pts_f, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0,), dtype=bool)

    z = pts[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-9)

    u = np.zeros((pts.shape[0],), dtype=np.int32)
    v = np.zeros((pts.shape[0],), dtype=np.int32)
    u[valid_z] = np.round(float(intr.fx) * pts[valid_z, 0] / z[valid_z] + float(intr.ppx)).astype(np.int32)
    v[valid_z] = np.round(float(intr.fy) * pts[valid_z, 1] / z[valid_z] + float(intr.ppy)).astype(np.int32)
    return u, v, valid_z


def save_colored_ply_ascii(path: str | Path, pts_f: np.ndarray, rgb_colors: np.ndarray) -> None:
    """
    XYZRGB付きPLYをASCII形式で保存する．
    rgb_colors は (N,3) RGB, uint8想定．
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(pts_f, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(rgb_colors).reshape(-1, 3)
    if colors.shape[0] != pts.shape[0]:
        raise ValueError(f"colors length mismatch: pts={pts.shape[0]}, colors={colors.shape[0]}")
    colors = np.clip(colors, 0, 255).astype(np.uint8)

    with path.open('w', encoding='utf-8') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {pts.shape[0]}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(pts, colors):
            f.write(f'{x:.9f} {y:.9f} {z:.9f} {int(r)} {int(g)} {int(b)}\n')


def save_final_pointcloud_rgb_link_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask01: np.ndarray,
    depth_masked: np.ndarray,
    pts_f: np.ndarray,
    intr,
    stem: str,
):
    """
    最終的に取得された点群とRGB画像の対応を保存する．

    保存内容:
      - final_valid_depth_region_overlay.png
          最終mask & depth>0 の領域．点群化に使われる入力画素．
      - final_pointcloud_projection_overlay.png
          calculate_yaw 後の最終点群 pts_f をRGB画像へ再投影した領域．
      - final_pointcloud_projection_mask.png
          再投影された画素マスク．
      - final_rgb_masked_by_pointcloud.png
          再投影点群領域だけを残したRGB画像．
      - final_pointcloud_colored.ply
          RGB色付き点群．Open3Dなどで点群と色を同時確認できる．
      - final_pointcloud_rgb_link_log.json
          対応付け情報．
    """
    debug_dir = Path(shot_dir) / "debug_final_pointcloud_rgb"
    debug_dir.mkdir(parents=True, exist_ok=True)

    color_np = np.asarray(color_np)
    h, w = color_np.shape[:2]
    mask01 = (np.asarray(mask01) > 0).astype(np.uint8)
    depth_masked = np.asarray(depth_masked)

    # 1) 最終フィルタ後，Depthが有効な画素．
    valid_input = ((mask01 > 0) & (depth_masked > 0)).astype(np.uint8)
    cv2.imwrite(str(debug_dir / f"{stem}_final_valid_depth_region_mask.png"), valid_input * 255)

    input_overlay = color_np.copy()
    input_overlay[valid_input == 1] = (0, 255, 0)
    input_blend = cv2.addWeighted(color_np, 0.65, input_overlay, 0.35, 0)
    contours, _ = cv2.findContours(valid_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(input_blend, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_final_valid_depth_region_overlay.png"), input_blend)

    rgb_masked_input = np.zeros_like(color_np)
    rgb_masked_input[valid_input == 1] = color_np[valid_input == 1]
    cv2.imwrite(str(debug_dir / f"{stem}_final_valid_depth_rgb_masked.png"), rgb_masked_input)

    # 2) calculate_yaw等を通った最終点群 pts_f をRGB画像へ投影．
    pts = np.asarray(pts_f, dtype=np.float64).reshape(-1, 3)
    u, v, valid_z = _project_points_to_pixels(pts, intr)
    in_img = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    proj_mask = np.zeros((h, w), dtype=np.uint8)
    if np.any(in_img):
        proj_mask[v[in_img], u[in_img]] = 1

    # 見やすいように少しだけ膨張した表示用マスクも作る．実データ自体はproj_mask．
    kernel = np.ones((3, 3), np.uint8)
    proj_vis_mask = cv2.dilate(proj_mask, kernel, iterations=1)

    cv2.imwrite(str(debug_dir / f"{stem}_final_pointcloud_projection_mask.png"), proj_mask * 255)

    proj_overlay = color_np.copy()
    proj_overlay[proj_vis_mask == 1] = (255, 255, 0)  # BGR: cyan
    proj_blend = cv2.addWeighted(color_np, 0.65, proj_overlay, 0.35, 0)
    contours, _ = cv2.findContours(proj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(proj_blend, contours, -1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_final_pointcloud_projection_overlay.png"), proj_blend)

    rgb_masked_proj = np.zeros_like(color_np)
    rgb_masked_proj[proj_vis_mask == 1] = color_np[proj_vis_mask == 1]
    cv2.imwrite(str(debug_dir / f"{stem}_final_rgb_masked_by_pointcloud.png"), rgb_masked_proj)

    # 3) 最終点群にRGB色を付与してPLY保存．投影外点は灰色にする．
    colors_rgb = np.full((pts.shape[0], 3), 180, dtype=np.uint8)
    if np.any(in_img):
        bgr = color_np[v[in_img], u[in_img], :]
        rgb = bgr[:, ::-1]
        colors_rgb[in_img] = rgb.astype(np.uint8)

    colored_ply_path = debug_dir / f"{stem}_final_pointcloud_colored.ply"
    save_colored_ply_ascii(colored_ply_path, pts, colors_rgb)

    log = {
        "rgb_shape_hw": [int(h), int(w)],
        "valid_input_pixel_count": int(np.count_nonzero(valid_input)),
        "final_point_count": int(pts.shape[0]),
        "projected_point_count_in_image": int(np.count_nonzero(in_img)),
        "projected_unique_pixel_count": int(np.count_nonzero(proj_mask)),
        "projected_in_image_ratio": float(np.count_nonzero(in_img) / max(pts.shape[0], 1)),
        "files": {
            "valid_depth_region_mask": str(debug_dir / f"{stem}_final_valid_depth_region_mask.png"),
            "valid_depth_region_overlay": str(debug_dir / f"{stem}_final_valid_depth_region_overlay.png"),
            "valid_depth_rgb_masked": str(debug_dir / f"{stem}_final_valid_depth_rgb_masked.png"),
            "pointcloud_projection_mask": str(debug_dir / f"{stem}_final_pointcloud_projection_mask.png"),
            "pointcloud_projection_overlay": str(debug_dir / f"{stem}_final_pointcloud_projection_overlay.png"),
            "rgb_masked_by_pointcloud": str(debug_dir / f"{stem}_final_rgb_masked_by_pointcloud.png"),
            "colored_ply": str(colored_ply_path),
        },
    }
    save_json(debug_dir / f"{stem}_final_pointcloud_rgb_link_log.json", log)
    print(f"✔ Saved final pointcloud/RGB link debug files: {debug_dir}")
    return log

def _clip_point_to_image(pt: tuple[int, int], w: int, h: int) -> tuple[int, int]:
    x, y = pt
    x = max(-10000, min(10000, int(x)))
    y = max(-10000, min(10000, int(y)))
    return x, y


def _draw_text_safe(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.45):
    """
    OpenCVで安全にテキストを描画する．日本語は文字化けする可能性があるため，
    デバッグ用途では番号・score中心に描画する．
    """
    x, y = int(org[0]), int(org[1])
    h, w = img.shape[:2]
    x = max(0, min(w - 1, x))
    y = max(12, min(h - 1, y))
    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_polygon_on_image(
    img: np.ndarray,
    poly: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    fill_alpha: float = 0.0,
):
    """
    BGR画像上にpolygonを描画する．fill_alpha > 0 の場合は半透明塗りも行う．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return img
    pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    if fill_alpha > 0.0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        img[:] = cv2.addWeighted(img, 1.0 - float(fill_alpha), overlay, float(fill_alpha), 0)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return img


def _crop_polygon_perspective(
    image_bgr: np.ndarray,
    poly: np.ndarray,
    pad_px: int = 6,
):
    """
    OCR polygon周辺を透視変換で切り出す．
    polygon順序が多少崩れていても minAreaRect で安定化する．
    """
    poly = _poly_from_any(poly)
    if poly is None:
        return None

    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] < 4:
        return None

    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)

    w_rect = max(1.0, float(rect[1][0]))
    h_rect = max(1.0, float(rect[1][1]))
    out_w = int(round(w_rect))
    out_h = int(round(h_rect))

    # あまりに小さい場合はスキップ
    if out_w < 2 or out_h < 2:
        return None

    # boxPointsの順序を左上，右上，右下，左下に並べ替え
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]
    src = np.asarray([tl, tr, br, bl], dtype=np.float32)

    dst = np.asarray(
        [
            [pad_px, pad_px],
            [out_w + pad_px - 1, pad_px],
            [out_w + pad_px - 1, out_h + pad_px - 1],
            [pad_px, out_h + pad_px - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(
        image_bgr,
        M,
        (out_w + 2 * pad_px, out_h + 2 * pad_px),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop


def save_ocr_region_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    query: str | None,
    mask01: np.ndarray,
    merged: dict,
    refine_info: dict | None = None,
    stem: str = "ocr_region",
):
    """
    OCR文字領域そのものを確認するためのデバッグ保存．

    保存内容:
      - *_ocr_all_polygons_raw.png       : ocr_result.jsonの生polygonをそのまま描画
      - *_ocr_all_polygons_corrected.png : 回転補正後のpolygonを描画
      - *_ocr_all_polygons.json          : 各OCR候補のtext, raw_poly, corrected_poly, transform_mode
      - *_ocr_selected_polygon.png       : 補正に使ったOCR polygon
      - *_ocr_selected_crop.png          : 補正に使ったOCR領域の切り出し
      - *_ocr_merged_box.png             : OCR/SAM統合結果のaxis-aligned box
    """
    debug_dir = Path(shot_dir) / "debug_ocr_band"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask01 = (mask01 > 0).astype(np.uint8)
    info = refine_info or {}
    h, w = color_np.shape[:2]
    image_shape = (h, w)
    query = "" if query is None else str(query)

    selected = info.get("selected_ocr_polygon") if isinstance(info, dict) else None
    selected_transform_mode = None
    if isinstance(selected, dict):
        selected_transform_mode = selected.get("transform_mode")

    center_for_auto, _ = _extract_merged_box_center(merged, mask01)

    # ===== 1) ocr_result.json の全OCR polygonを描画 =====
    raw_img = color_np.copy()
    corrected_img = color_np.copy()
    candidates_log = []
    ocr_path = Path(shot_dir) / "ocr_result.json"

    if ocr_path.exists():
        try:
            obj = json.loads(ocr_path.read_text(encoding="utf-8"))
            for i, item in enumerate(_iter_ocr_items(obj)):
                text = str(item.get("text", ""))
                raw_poly = _poly_from_any(item.get("bbox"))
                if raw_poly is None:
                    continue

                # 生polygonを描画．これがRGBと合わない場合，OCR側の回転座標である可能性が高い．
                _draw_polygon_on_image(raw_img, raw_poly, (0, 165, 255), thickness=1, fill_alpha=0.03)
                raw_center, raw_axis, raw_short, raw_long = _polygon_center_and_axis(raw_poly)
                if raw_center is not None:
                    _draw_text_safe(raw_img, f"raw{i}", (int(raw_center[0]) + 4, int(raw_center[1]) - 4), scale=0.35)

                # 補正後polygonを描画．selectedと同じtransform_modeが分かる場合はそれを使う．
                # 分からない場合は，この候補ごとに最もマスク・merged中心に合う変換を選ぶ．
                mode_scores = []
                modes_to_try = [selected_transform_mode] if selected_transform_mode in _OCR_POLY_TRANSFORM_MODES else list(_OCR_POLY_TRANSFORM_MODES)
                for mode in modes_to_try:
                    poly_corr = _transform_ocr_poly_to_rgb(raw_poly, image_shape, mode)
                    if poly_corr is None:
                        continue
                    c, ax, short_len, long_len = _polygon_center_and_axis(poly_corr)
                    sim = _text_similarity(query, text) if query else 0.0
                    overlap = _polygon_overlap_ratio_with_mask(poly_corr, mask01)
                    center_score = _center_distance_score(c, center_for_auto, image_shape)
                    valid_score = _poly_image_valid_score(poly_corr, image_shape)
                    score = 2.0 * sim + 0.55 * overlap + 0.45 * center_score + 0.20 * valid_score
                    mode_scores.append((score, mode, poly_corr, c, ax, short_len, long_len, sim, overlap, center_score, valid_score))

                if not mode_scores:
                    continue
                mode_scores.sort(key=lambda x: x[0], reverse=True)
                score, mode, poly_corr, center, axis, short_len, long_len, sim, overlap, center_score, valid_score = mode_scores[0]

                _draw_polygon_on_image(corrected_img, poly_corr, (255, 255, 0), thickness=1, fill_alpha=0.04)
                if center is not None:
                    cx, cy = int(round(center[0])), int(round(center[1]))
                    cv2.circle(corrected_img, (cx, cy), 3, (255, 255, 0), -1)
                    _draw_text_safe(corrected_img, f"{i}:s{sim:.2f}/o{overlap:.2f}/{mode}", (cx + 4, cy - 4), scale=0.34)

                candidates_log.append({
                    "index": int(i),
                    "text": text,
                    "score": None if item.get("score") is None else float(item.get("score")),
                    "source": item.get("source", "unknown"),
                    "similarity_to_query": float(sim),
                    "overlap_with_selected_mask": float(overlap),
                    "center_distance_score": float(center_score),
                    "image_valid_score": float(valid_score),
                    "transform_mode": mode,
                    "center": None if center is None else [float(center[0]), float(center[1])],
                    "axis": None if axis is None else [float(axis[0]), float(axis[1])],
                    "short_len": None if short_len is None else float(short_len),
                    "long_len": None if long_len is None else float(long_len),
                    "raw_poly": np.asarray(raw_poly, dtype=np.float32).tolist(),
                    "corrected_poly": np.asarray(poly_corr, dtype=np.float32).tolist(),
                })
        except Exception:
            traceback.print_exc()
            candidates_log.append({"error": "failed to read or parse ocr_result.json"})
    else:
        candidates_log.append({"error": f"ocr_result.json not found: {ocr_path}"})

    cv2.imwrite(str(debug_dir / f"{stem}_ocr_all_polygons_raw.png"), raw_img)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_all_polygons_corrected.png"), corrected_img)
    # 互換用に，従来名も補正後polygonとして保存する．
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_all_polygons.png"), corrected_img)
    save_json(debug_dir / f"{stem}_ocr_all_polygons.json", {
        "query": query,
        "ocr_result_path": str(ocr_path),
        "note": "raw_poly is the coordinate in ocr_result.json; corrected_poly is transformed to after_init_rgb.png coordinates.",
        "selected_transform_mode": selected_transform_mode,
        "num_candidates": len([c for c in candidates_log if "corrected_poly" in c]),
        "candidates": candidates_log,
    })

    # ===== 2) merged結果のboxを描画 =====
    merged_img = color_np.copy()
    results = merged.get("results", []) if isinstance(merged, dict) else []
    if results:
        for i, r in enumerate(results):
            box = r.get("box") if isinstance(r, dict) else None
            poly = _poly_from_any(box)
            if poly is not None:
                _draw_polygon_on_image(merged_img, poly, (0, 255, 255), thickness=2, fill_alpha=0.08)
                c, _, _, _ = _polygon_center_and_axis(poly)
                if c is not None:
                    _draw_text_safe(merged_img, f"merged{i}: {r.get('name', '')} score={r.get('score', '')}", (int(c[0]) + 5, int(c[1]) - 5), scale=0.45)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_merged_box.png"), merged_img)

    # ===== 3) 補正に使ったselected OCR polygonを描画・切り出し =====
    selected_img = color_np.copy()
    selected_raw_img = color_np.copy()
    selected_poly = None
    selected_raw_poly = None
    if isinstance(selected, dict):
        selected_poly = _poly_from_any(selected.get("poly"))
        selected_raw_poly = _poly_from_any(selected.get("raw_poly"))

    if selected_raw_poly is not None:
        _draw_polygon_on_image(selected_raw_img, selected_raw_poly, (0, 165, 255), thickness=3, fill_alpha=0.18)
    else:
        _draw_text_safe(selected_raw_img, "No selected raw OCR polygon", (20, 40), scale=0.7)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_selected_polygon_raw.png"), selected_raw_img)

    if selected_poly is not None:
        _draw_polygon_on_image(selected_img, selected_poly, (0, 0, 255), thickness=3, fill_alpha=0.18)
        c, axis, short_len, long_len = _polygon_center_and_axis(selected_poly)
        if c is not None:
            cv2.circle(selected_img, (int(round(c[0])), int(round(c[1]))), 6, (255, 0, 0), -1)
            _draw_text_safe(
                selected_img,
                f"selected OCR: {selected.get('text', '')} sim={float(selected.get('sim', 0.0)):.2f} mode={selected.get('transform_mode', '')}",
                (int(round(c[0])) + 8, int(round(c[1])) - 8),
                scale=0.42,
            )
        if c is not None and axis is not None:
            ax, ay = float(axis[0]), float(axis[1])
            length = 350.0
            p1 = (int(round(c[0] - ax * length)), int(round(c[1] - ay * length)))
            p2 = (int(round(c[0] + ax * length)), int(round(c[1] + ay * length)))
            cv2.line(selected_img, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

        crop = _crop_polygon_perspective(color_np, selected_poly)
        if crop is not None:
            cv2.imwrite(str(debug_dir / f"{stem}_ocr_selected_crop.png"), crop)
    else:
        _draw_text_safe(selected_img, "No selected OCR polygon", (20, 40), scale=0.7)

    cv2.imwrite(str(debug_dir / f"{stem}_ocr_selected_polygon.png"), selected_img)

def save_mask_refine_debug(
    shot_dir: Path,
    color_np: np.ndarray,
    mask_before: np.ndarray,
    mask_after: np.ndarray,
    merged: dict,
    refine_info: dict | None = None,
    stem: str = "ocr_band_refine",
    query: str | None = None,
):
    """
    OCR帯補正のデバッグ用保存．
    入力画像，補正前マスク，補正後マスク，差分，OCR/SAM対応ログを保存する．
    """
    debug_dir = Path(shot_dir) / "debug_ocr_band"
    debug_dir.mkdir(parents=True, exist_ok=True)

    mask_before = (mask_before > 0).astype(np.uint8)
    mask_after = (mask_after > 0).astype(np.uint8)

    # 入力画像保存
    cv2.imwrite(str(debug_dir / f"{stem}_input_rgb.png"), color_np)

    # マスク保存
    cv2.imwrite(str(debug_dir / f"{stem}_mask_before.png"), mask_before * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_mask_after.png"), mask_after * 255)

    # 削除された領域・残った領域
    removed = ((mask_before == 1) & (mask_after == 0)).astype(np.uint8)
    kept = ((mask_before == 1) & (mask_after == 1)).astype(np.uint8)

    cv2.imwrite(str(debug_dir / f"{stem}_removed.png"), removed * 255)
    cv2.imwrite(str(debug_dir / f"{stem}_kept.png"), kept * 255)

    # overlay: 緑=残った領域，赤=削除された領域
    overlay = color_np.copy()
    overlay[kept == 1] = (0, 255, 0)
    overlay[removed == 1] = (0, 0, 255)

    blended = cv2.addWeighted(color_np, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(debug_dir / f"{stem}_overlay_kept_removed.png"), blended)

    # OCR中心線も描画
    line_img = color_np.copy()
    info = refine_info or {}
    center = info.get("ocr_center")
    axis = info.get("axis")
    half_width = info.get("half_width_px")
    if center is not None and axis is not None:
        cx, cy = float(center[0]), float(center[1])
        ax, ay = float(axis[0]), float(axis[1])
        length = 1000.0
        p1 = (int(round(cx - ax * length)), int(round(cy - ay * length)))
        p2 = (int(round(cx + ax * length)), int(round(cy + ay * length)))
        cv2.line(line_img, p1, p2, (0, 0, 255), 2)
        cv2.circle(line_img, (int(round(cx)), int(round(cy))), 6, (255, 0, 0), -1)

        # 帯の左右境界も描画
        if half_width is not None:
            nx = -ay
            ny = ax
            for sgn in (-1.0, 1.0):
                ox = nx * float(half_width) * sgn
                oy = ny * float(half_width) * sgn
                q1 = (int(round(cx + ox - ax * length)), int(round(cy + oy - ay * length)))
                q2 = (int(round(cx + ox + ax * length)), int(round(cy + oy + ay * length)))
                cv2.line(line_img, q1, q2, (255, 0, 255), 1)

    cv2.imwrite(str(debug_dir / f"{stem}_ocr_axis_band.png"), line_img)

    # 側面張り出し抑制の結果を見やすくするため，補正後マスクの輪郭も軸画像へ重ねる．
    mask_contour_img = line_img.copy()
    contours, _ = cv2.findContours(mask_after, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(mask_contour_img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / f"{stem}_ocr_axis_band_with_mask_after.png"), mask_contour_img)

    # OCR文字領域そのものの可視化も保存する．
    save_ocr_region_debug(
        shot_dir=shot_dir,
        color_np=color_np,
        query=query,
        mask01=mask_before,
        merged=merged,
        refine_info=info,
        stem=stem,
    )

    log = {
        "selected_mask_name": merged.get("book_name"),
        "selected_mask_index": int(merged.get("sel_idx")) if merged.get("sel_idx") is not None else None,
        "ocr_sam_results": merged.get("results"),
        "mask_before_area_px": int(mask_before.sum()),
        "mask_after_area_px": int(mask_after.sum()),
        "kept_ratio": float(mask_after.sum() / max(mask_before.sum(), 1)),
        "refine_info": info,
    }
    save_json(debug_dir / f"{stem}_log.json", log)

    print(f"✔ Saved OCR-band debug files: {debug_dir}")
def save_pointcloud_screenshot(
    pts_f: np.ndarray,
    target_point: np.ndarray,
    save_path: Path,
    show_window: bool = False,
) -> None:
    """
    点群とターゲット点を Open3D で描画し、そのスクリーンショットを保存する関数。
    既存の visualize_points_and_target_open3d には影響を与えない。

    Parameters
    ----------
    pts_f : (N, 3) np.ndarray
        点群 [m]
    target_point : (3,) np.ndarray
        ターゲット点 [m]
    save_path : Path
        画像の保存先パス (png など)
    show_window : bool
        True の場合はウィンドウを表示、False の場合は非表示でレンダリングのみ
    """
    try:
        pts = np.asarray(pts_f).reshape(-1, 3)
        tgt = np.asarray(target_point).reshape(3)

        # 点群
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # ターゲット点を小さな球で可視化
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(tgt)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 赤色

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show_window)
        vis.add_geometry(pcd)
        vis.add_geometry(sphere)

        # 一度レンダリングしてからキャプチャ
        vis.poll_events()
        vis.update_renderer()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(save_path), do_render=True)

        vis.destroy_window()
        print(f"✔ Saved pointcloud screenshot: {save_path}")

    except Exception:
        # ここで落ちてもメイン処理に影響が出ないようにする
        traceback.print_exc()
        print("⚠ 点群スクリーンショット保存に失敗しました（処理は続行します）")

def run_capture_and_pca(
    query: str,
    out_dir: str | Path = "captures",
    #1280x720 固定
    width: int = 1280,
    height: int = 720,
    fps: int = 6,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        shot_dir = out_dir / ts
        shot_dir.mkdir(parents=True, exist_ok=True)

        pipe2 = rs.pipeline()
        cfg2 = rs.config()
        cfg2.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg2.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        align2 = rs.align(rs.stream.color)

        # ===== 1) 撮影 =====
        capture_start = time.perf_counter()
        color_np, depth_np_u16, intr, depth_scale = capture_one_shot(
            pipe2, cfg2, align2, shot_dir, stem="after_init"
        )
        capture_end = time.perf_counter()
        print("Realsense captured.")
        print(f"[TIME] capture            : {capture_end - capture_start:.3f} sec")

        # ===== 1.5) カメラパラメータ保存 =====
        camera_json = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "depth_scale": float(depth_scale),
            "fps": int(fps),
        }

        camera_json_path = shot_dir / "camera_params.json"
        camera_json_path.write_text(
            json.dumps(camera_json, indent=2),
            encoding="utf-8"
        )
        print(f":heavy_check_mark: Saved camera params: {camera_json_path}")

        # ===== 2) OCR を先に非同期開始 =====
        ocr_start = time.perf_counter()
        ocr_proc = start_ocr_subprocess(shot_dir)
        print("[PARALLEL] OCR subprocess started.")

        # ===== 3) SAM 実行（OCR と並列） =====
        sam_start = time.perf_counter()

        sam_cfg = SamConfig(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            device=sam_device,
        )
        sam_runner = SamBatchInfer_storage(sam_cfg)

        # BGR（numpy） → RGB の PIL.Image に変換
        rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))

        # ステージ保存設定（after_nms / before_smooth など）
        stage_cfg = StageSaveCfg(out_dir=shot_dir)

        masks, sam_data = sam_runner.infer_masks(
            rgb_pil,
            stage_save=stage_cfg,
            stem_for_save="rgb",
        )

        sam_end = time.perf_counter()
        print(f"[TIME] SAM total          : {sam_end - sam_start:.3f} sec")

        # ===== 4) OCR 終了待ち =====
        ocr_stdout = wait_ocr_subprocess(ocr_proc, timeout=120.0)
        ocr_end = time.perf_counter()

        if ocr_stdout.strip():
            print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")

        print(f"[TIME] OCR wall           : {ocr_end - ocr_start:.3f} sec")
        print(f"[TIME] capture->SAM end   : {sam_end - capture_start:.3f} sec")
        print(f"[TIME] OCR start->end     : {ocr_end - ocr_start:.3f} sec")
        print(f"[TIME] SAM & OCR end           : {ocr_end - capture_start:.3f} sec")

        # ===== 5) OCR結果とSAM結果を統合 =====
        merge_start = time.perf_counter()

        merged = merge_ocr_and_masks(
            query=query,
            masks=masks,
            shot_dir=shot_dir,
            interactive=interactive,
            threshold=40,
        )

        sel_idx = merged["sel_idx"]
        sel_mask = merged["sel_mask"]
        mask01 = merged["mask01"]

        # ===== OCR bbox をアンカーにしたマスク補正 =====
        mask01_before_refine = mask01.copy()

        mask01, refine_info = refine_mask_by_ocr_axis_band(
            mask01=mask01,
            merged=merged,
            image_shape=color_np.shape[:2],
            shot_dir=shot_dir,
            query=query,
            mask_width_ratio=0.98,
            min_keep_ratio=0.60,
            use_ocr_short_width=False,
            ocr_short_to_half_width_scale=1.35,
            min_ocr_half_width_px=28.0,
            suppress_side_protrusions=False,
            profile_bin_size_px=8.0,
            profile_width_margin=1.15,
            min_profile_half_width_px=16.0,
            max_profile_shrink_ratio=0.35,
            return_info=True,
        )

        save_mask_refine_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_refine,
            mask_after=mask01,
            merged=merged,
            refine_info=refine_info,
            stem=f"mask{sel_idx}",
            query=query,
        )

        merge_end = time.perf_counter()
        print(f"[TIME] merge OCR+SAM      : {merge_end - merge_start:.3f} sec")
        print(f"[SAM] selected id = {sel_idx}, mask shape = {mask01.shape}")

        # ===== 6) 選択結果のオーバーレイ保存 =====
        _save_points_and_overlay(
            rgb_pil,
            [sel_mask],
            shot_dir,
            f"rgb_mask{sel_idx}_selected",
            draw_ids=False,
        )

        # ===== 7) 対象書籍のみの RGB/Depth を保存 =====
        # ここで従来の「Depth中央値±3cm」補正が行われる．
        depth_masked = save_masked_and_cropped(
            color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}"
        )

        # ===== 7.5) Depth補正後の点群列長さで側面を追加除去 =====
        # 文字領域は「対象書籍上のseed」として使い，幅を固定せず，
        # 背表紙方向に長く連続するt列だけを残す．
        mask01_before_column = mask01.copy()
        depth_masked_before_column = depth_masked.copy()

        mask01, depth_masked, column_info = refine_mask_by_spine_column_length_after_depth(
            mask01=mask01,
            depth_masked=depth_masked,
            refine_info=refine_info,
            image_shape=color_np.shape[:2],
            t_bin_size_px=4.0,
            s_bin_size_px=5.0,
            s_gap_allow_px=18.0,
            length_reference_percentile=85.0,
            min_length_ratio=0.65,
            relaxed_edge_length_ratio=0.55,
            min_points_per_t_bin=12,
            min_selected_t_bins=2,
            expand_selected_t_bins=0,
            s_margin_px=8.0,
            min_valid_keep_ratio=0.30,
            return_info=True,
        )

        save_spine_column_length_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask_before=mask01_before_column,
            mask_after=mask01,
            depth_before=depth_masked_before_column,
            depth_after=depth_masked,
            column_info=column_info,
            stem=f"mask{sel_idx}",
        )

        # ===== 8) マスク + depth → カメラ座標点群へ変換 =====
        _3D_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
        yaw = _3D_info["yaw"]
        pts_f = _3D_info["points"]  # (N,3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_f)

        # ===== 9) 点群を保存 =====
        save_ply_ascii(shot_dir / "pointcloud.ply", pts_f, None)

        # ===== 9.5) 最終点群とRGB画像の対応を保存 =====
        save_final_pointcloud_rgb_link_debug(
            shot_dir=shot_dir,
            color_np=color_np,
            mask01=mask01,
            depth_masked=depth_masked,
            pts_f=pts_f,
            intr=intr,
            stem=f"mask{sel_idx}",
        )

        # ===== 10) PCA =====
        mean, pc1, pc2 = pca_axes_fix_dir(pts_f)

        vx, vy = float(pc1[0]), float(pc1[1])
        norm_xy = float(np.hypot(vx, vy))
        if norm_xy < 1e-8:
            theta_rad = 0.0
        else:
            theta_rad = float(np.arctan2(vy, vx))  # 基準は +x 軸

        book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
        book_width = book_width_info.get("av_book_width_m")

        # ===== 11) 把持位置導出 =====
        target_point_info = find_target_point(pts_f)
        target_point = target_point_info.get("target_m")

        # ===== 12) 可視化 =====
        visualize_points_and_target_open3d(pts_f, target_point)

        # ===== 12.5) 点群ビューのスクリーンショット保存 =====
        vis_img_path = shot_dir / "pointcloud_view.png"
        save_pointcloud_screenshot(
            pts_f=pts_f,
            target_point=target_point,
            save_path=vis_img_path,
            show_window=False,
        )

        # ===== 13) PCA結果保存 =====
        print(":heavy_check_mark: PCA result:")
        print(f"  theta_rad = {theta_rad:.6f}")
        print(f"  p_min = {target_point}")
        print(":heavy_check_mark: Files saved under:", shot_dir)

        pca_json = {
            "theta_rad": float(theta_rad),
            "theta_deg": float(np.degrees(theta_rad)),
            "p_min_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
            "book_width_mm": float(book_width * 1000.0),  # book_width は [m] 想定
        }

        json_path = Path(shot_dir) / "pca_result.json"
        json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f":heavy_check_mark: Saved PCA JSON: {json_path}")

        return theta_rad, target_point, book_width * 1000.0, shot_dir  # m → mm

    except Exception:
        traceback.print_exc()
        print("認識失敗！！")
        return None, None, None, None
    
def run_capture_and_pca_offline(
    query: str,
    shot_dir: str | Path,
    sam_device: str = "gpu",
    encoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
    decoder_path: str = "/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
    interactive: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, Path]:
    """
    すでに撮影済みの画像・深度を用いて、
    run_capture_and_pca と同じ処理フロー（SAM→OCR→点群→PCA）を行うオフライン版。
    """
    shot_dir = Path(shot_dir)

    rgb_path = shot_dir / "after_init_rgb.png"
    depth_path = shot_dir / "after_init_depth.npy"

    if not rgb_path.exists():
        raise FileNotFoundError(f"{rgb_path} がありません")
    if not depth_path.exists():
        raise FileNotFoundError(f"{depth_path} がありません")

    color_np = cv2.imread(str(rgb_path))  # BGR
    depth_np_u16 = np.load(depth_path)    # (H,W) uint16

    intr = rs.intrinsics()
    intr.width = 1280
    intr.height = 720
    intr.fx = 908.1617431640625
    intr.fy = 906.4829711914062
    intr.ppx = 637.79833984375
    intr.ppy = 371.0213928222656

    depth_scale = 0.0010000000474974513

    # ===== 1) OCR を先に非同期開始 =====
    ocr_start = time.perf_counter()
    ocr_proc = start_ocr_subprocess(shot_dir)
    print("[PARALLEL][OFFLINE] OCR subprocess started.")

    # ===== 2) SAM 実行（OCR と並列） =====
    sam_start = time.perf_counter()

    sam_cfg = SamConfig(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        device=sam_device,
    )
    sam_runner = SamBatchInfer_storage(sam_cfg)

    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    stage_cfg = StageSaveCfg(out_dir=shot_dir)

    masks, sam_data = sam_runner.infer_masks(
        rgb_pil,
        stage_save=stage_cfg,
        stem_for_save="rgb",
    )

    sam_end = time.perf_counter()
    print(f"[TIME][OFFLINE] SAM total     : {sam_end - sam_start:.3f} sec")

    # ===== 3) OCR 終了待ち =====
    ocr_stdout = wait_ocr_subprocess(ocr_proc, timeout=120.0)
    ocr_end = time.perf_counter()

    if ocr_stdout.strip():
        print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")

    print(f"[TIME][OFFLINE] OCR wall      : {ocr_end - ocr_start:.3f} sec")

    # ===== 4) 統合 =====
    merge_start = time.perf_counter()

    merged = merge_ocr_and_masks(
        query=query,
        masks=masks,
        shot_dir=shot_dir,
        interactive=interactive,
        threshold=40,
    )

    sel_idx = merged["sel_idx"]
    sel_mask = merged["sel_mask"]
    mask01 = merged["mask01"]

    # ===== OCR bbox をアンカーにしたマスク補正 =====
    mask01_before_refine = mask01.copy()

    mask01, refine_info = refine_mask_by_ocr_axis_band(
        mask01=mask01,
        merged=merged,
        image_shape=color_np.shape[:2],
        shot_dir=shot_dir,
        query=query,
        mask_width_ratio=0.98,
        min_keep_ratio=0.60,
        use_ocr_short_width=False,
        ocr_short_to_half_width_scale=1.35,
        min_ocr_half_width_px=28.0,
        suppress_side_protrusions=False,
        profile_bin_size_px=8.0,
        profile_width_margin=1.15,
        min_profile_half_width_px=16.0,
        max_profile_shrink_ratio=0.35,
        return_info=True,
    )

    save_mask_refine_debug(
        shot_dir=shot_dir,
        color_np=color_np,
        mask_before=mask01_before_refine,
        mask_after=mask01,
        merged=merged,
        refine_info=refine_info,
        stem=f"mask{sel_idx}_offline",
        query=query,
    )

    merge_end = time.perf_counter()
    print(f"[TIME][OFFLINE] merge OCR+SAM: {merge_end - merge_start:.3f} sec")
    print(f"[SAM OFFLINE] selected id = {sel_idx}, mask shape = {mask01.shape}")

    # ===== 5) 選択結果保存 =====
    _save_points_and_overlay(
        rgb_pil,
        [sel_mask],
        shot_dir,
        f"rgb_mask{sel_idx}_selected_offline",
        draw_ids=False,
    )

    # ===== 6) マスクDepth保存 =====
    # ここで従来の「Depth中央値±3cm」補正が行われる．
    depth_masked = save_masked_and_cropped(
        color_np, depth_np_u16, mask01, shot_dir, f"mask{sel_idx}_offline"
    )

    # ===== 6.5) Depth補正後の点群列長さで側面を追加除去 =====
    # 文字領域は「対象書籍上のseed」として使い，幅を固定せず，
    # 背表紙方向に長く連続するt列だけを残す．
    mask01_before_column = mask01.copy()
    depth_masked_before_column = depth_masked.copy()

    mask01, depth_masked, column_info = refine_mask_by_spine_column_length_after_depth(
        mask01=mask01,
        depth_masked=depth_masked,
        refine_info=refine_info,
        image_shape=color_np.shape[:2],
        t_bin_size_px=4.0,
        s_bin_size_px=5.0,
        s_gap_allow_px=18.0,
        length_reference_percentile=85.0,
        min_length_ratio=0.65,
        relaxed_edge_length_ratio=0.55,
        min_points_per_t_bin=12,
        min_selected_t_bins=2,
        expand_selected_t_bins=0,
        s_margin_px=8.0,
        min_valid_keep_ratio=0.30,
        return_info=True,
    )

    save_spine_column_length_debug(
        shot_dir=shot_dir,
        color_np=color_np,
        mask_before=mask01_before_column,
        mask_after=mask01,
        depth_before=depth_masked_before_column,
        depth_after=depth_masked,
        column_info=column_info,
        stem=f"mask{sel_idx}_offline",
    )

    # ===== 7) 点群化 =====
    _3D_info = calculate_yaw(mask01, depth_masked, intr, depth_scale)
    yaw = _3D_info["yaw"]
    pts_f = _3D_info["points"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_f)

    save_ply_ascii(shot_dir / "pointcloud_offline.ply", pts_f, None)

    # ===== 7.5) 最終点群とRGB画像の対応を保存 =====
    save_final_pointcloud_rgb_link_debug(
        shot_dir=shot_dir,
        color_np=color_np,
        mask01=mask01,
        depth_masked=depth_masked,
        pts_f=pts_f,
        intr=intr,
        stem=f"mask{sel_idx}_offline",
    )

    # ===== 8) PCA =====
    mean, pc1, pc2 = pca_axes_fix_dir(pts_f)
    vx, vy = float(pc1[0]), float(pc1[1])
    norm_xy = float(np.hypot(vx, vy))
    if norm_xy < 1e-8:
        theta_rad = 0.0
    else:
        theta_rad = float(np.arctan2(vy, vx))

    book_width_info = estimate_book_width(pts_f, mean, pc1, pc2)
    book_width = book_width_info.get("av_book_width_m")

    # ===== 9) 把持位置 =====
    target_point_info = find_target_point(pts_f)
    target_point = target_point_info.get("target_m")

    # ===== 10) 可視化 =====
    visualize_points_and_target_open3d(pts_f, target_point)

    vis_img_path = shot_dir / "pointcloud_view_offline.png"
    save_pointcloud_screenshot(
        pts_f=pts_f,
        target_point=target_point,
        save_path=vis_img_path,
        show_window=False,
    )

    print(":heavy_check_mark: [OFFLINE] PCA result:")
    print(f"  theta_rad = {theta_rad:.6f}")
    print(f"  p_min = {target_point}")
    print(":heavy_check_mark: Files saved under:", shot_dir)

    pca_json = {
        "theta_rad": float(theta_rad),
        "theta_deg": float(np.degrees(theta_rad)),
        "p_min_m": [float(x) for x in np.asarray(target_point).reshape(-1)],
        "book_width_mm": float(book_width * 1000.0),
    }
    json_path = shot_dir / "pca_result_offline.json"
    json_path.write_text(json.dumps(pca_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f":heavy_check_mark: Saved PCA JSON (offline): {json_path}")

    return theta_rad, target_point, book_width * 1000.0, shot_dir


    
def main():
    import pyrealsense2 as rs
    import cv2
    from pathlib import Path
    from datetime import datetime

    # ========= 設定 =========
    width = 1280
    height = 720
    fps = 1

    # 保存先: captures/YYYY-MM-DD
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_dir = Path("./captures") / today_str
    save_dir.mkdir(parents=True, exist_ok=True)

    # ========= RealSense 初期化 =========
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipe.start(cfg)

    try:
        print("カメラ起動中...")
        print("s: 保存")
        print("q or ESC: 終了")

        # 最初の数フレームは捨てる（露出安定用）
        for _ in range(30):
            pipe.wait_for_frames()

        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = cv2.cvtColor(
                cv2.imread("/dev/null") if False else
                __import__("numpy").asanyarray(color_frame.get_data()),
                cv2.COLOR_RGB2BGR
            ) if False else __import__("numpy").asanyarray(color_frame.get_data())

            # 表示用
            preview = color_image.copy()
            cv2.putText(
                preview,
                f"{width}x{height}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                preview,
                "Press 's' to save / 'q' or ESC to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("RealSense Capture", preview)
            key = cv2.waitKey(1) & 0xFF

            # sキーで保存
            if key == ord("s"):
                now = datetime.now().strftime("%H-%M-%S")
                filename = f"{width}x{height}_{now}.png"
                filepath = save_dir / filename

                cv2.imwrite(str(filepath), color_image)
                print(f"保存: {filepath}")

            # q or ESCで終了
            elif key == ord("q") or key == 27:
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--out", type=str, default="captures", help="撮影と結果保存のベースフォルダ")
#     ap.add_argument("--w", type=int, default=1280)
#     ap.add_argument("--h", type=int, default=720)
#     ap.add_argument("--fps", type=int, default=6)
#     ap.add_argument("--sam_device", choices=["gpu", "cpu", "auto"], default="gpu")
#     ap.add_argument(
#         "--encoder",
#         default="/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx",
#     )
#     ap.add_argument(
#         "--decoder",
#         default="/home/book/pro_book/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx",
#     )
#     args = ap.parse_args()

#     theta_rad, p_min, book_width, yaw = run_capture_and_pca(
#         query="独立行政法人",
#         out_dir=args.out,
#         width=args.w,
#         height=args.h,
#         fps=args.fps,
#         sam_device=args.sam_device,
#         encoder_path=args.encoder,
#         decoder_path=args.decoder,
#         interactive=True,
#     )

#     print("\n=== Summary ===")
#     print(f"book width = {book_width}")
#     print(f"roll (deg) = {np.degrees(theta_rad):.6f}")
#     print(f"p_min = {p_min}")

#     #print("yaw (deg) = {:.3f}".format(np.degrees(yaw)))
#     print("===============")
    

if __name__ == "__main__":
    main()
