#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import re

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    _HAS_TESS = False


@dataclass
class OcrResult:
    ok: bool
    label_raw: str
    label_digits6: Optional[str]
    label_digits8: Optional[str]


def rotate180(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.rotate(img_bgr, cv2.ROTATE_180)


def bbox_rotate180_xyxy(bbox_xyxy: Tuple[float, float, float, float], W: int, H: int):
    """元画像のbbox(x1,y1,x2,y2)を180度回転後座標へ変換"""
    x1, y1, x2, y2 = bbox_xyxy
    rx1 = W - x2
    rx2 = W - x1
    ry1 = H - y2
    ry2 = H - y1
    return (rx1, ry1, rx2, ry2)


def crop_right_of_barcode(
    img_bgr: np.ndarray,
    barcode_bbox_xyxy: Tuple[float, float, float, float],
    x_margin_ratio: float = 0.05,
    w_ratio: float = 0.9,
    y_margin_ratio: float = 0.25,
) -> np.ndarray:
    """
    180度回転後の画像に対して、
    バーコードbboxの「右側」にある番号領域を切り出す。

    - x方向: bbox右端 + margin 〜 bbox幅*w_ratio ぶん右へ（＋クリップ）
    - y方向: bbox上下にmarginを付けて切り出し
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = barcode_bbox_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    x_start = int(np.clip(x2 + bw * x_margin_ratio, 0, W - 1))
    x_end   = int(np.clip(x2 + bw * (x_margin_ratio + w_ratio), 0, W))

    y_start = int(np.clip(y1 - bh * y_margin_ratio, 0, H - 1))
    y_end   = int(np.clip(y2 + bh * y_margin_ratio, 0, H))

    if x_end <= x_start + 5 or y_end <= y_start + 5:
        # 変なROIになったら保険で右半分
        x_start = int(W * 0.55)
        x_end = W
        y_start = 0
        y_end = H

    return img_bgr[y_start:y_end, x_start:x_end]


def preprocess_digits_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """
    数字OCR向け前処理（“細線を潰しすぎない”寄り）
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 軽くノイズ低減（強すぎると細い線が消える）
    gray = cv2.bilateralFilter(gray, 7, 35, 35)

    # コントラスト
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 拡大（Tesseractは少し大きい方が安定）
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # 2値化
    bin_img = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )

    # 小さな穴埋め（数字の欠け対策、やりすぎ注意）
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)

    return bin_img


def parse_label(text: str) -> OcrResult:
    """
    期待: '00-00-00' もしくは '000000'
    """
    s = text.strip()
    s = s.replace(" ", "").replace("\n", "").replace("\t", "")

    # まず 00-00-00 を優先
    m = re.search(r"(\d{2}-\d{2}-\d{2})", s)
    if m:
        raw = m.group(1)
        d6 = raw.replace("-", "")
        d8 = "02" + d6
        return OcrResult(True, raw, d6, d8)

    # 次に 6桁連続
    m = re.search(r"(\d{6})", s)
    if m:
        d6 = m.group(1)
        d8 = "02" + d6
        return OcrResult(True, d6, d6, d8)

    return OcrResult(False, s, None, None)


class DigitsOcrEngine:
    """
    数字専用OCR（Tesseract推奨）
    """
    def __init__(self):
        if not _HAS_TESS:
            raise RuntimeError("pytesseract が import できません。先に pytesseract + tesseract-ocr を入れてください。")

        # digits + hyphen のみ
        self._tess_cfg = r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789-'

    def infer_from_roi(self, roi_bgr: np.ndarray, debug_dir: Optional[Path] = None, prefix: str = "digits") -> OcrResult:
        proc = preprocess_digits_roi(roi_bgr)

        if debug_dir is not None:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"{prefix}_roi.png"), roi_bgr)
            cv2.imwrite(str(debug_dir / f"{prefix}_proc.png"), proc)

        text = pytesseract.image_to_string(proc, config=self._tess_cfg)
        return parse_label(text)

    def infer_from_frame(
        self,
        frame_bgr: np.ndarray,
        barcode_bbox_xyxy: Optional[Tuple[float, float, float, float]] = None,
        rotate_180: bool = True,
        debug_dir: Optional[Path] = None,
    ) -> OcrResult:
        img = rotate180(frame_bgr) if rotate_180 else frame_bgr

        if barcode_bbox_xyxy is not None:
            H, W = frame_bgr.shape[:2]
            bbox_rot = bbox_rotate180_xyxy(barcode_bbox_xyxy, W=W, H=H)
            roi = crop_right_of_barcode(img, bbox_rot)
        else:
            # bboxが無い場合は右側を広めに見る（保険）
            H, W = img.shape[:2]
            roi = img[:, int(W * 0.55):]

        return self.infer_from_roi(roi, debug_dir=debug_dir, prefix="digits")