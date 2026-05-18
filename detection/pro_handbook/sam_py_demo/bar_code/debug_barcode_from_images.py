#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import argparse
from pathlib import Path
import numpy as np
import re
import easyocr

# ===== 既存コードからimport =====
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic_ros2_editing import (
    detect_barcode_bbox,
)

# =========================
# OCR初期化（1回だけ）
# =========================
reader = easyocr.Reader(['en'], gpu=False)


# =========================
# OCR
# =========================
def run_ocr_easy(roi_bgr):
    results = reader.readtext(
        roi_bgr,
        detail=1,
        paragraph=False,
        allowlist='0123456789-'
    )

    texts = []
    for bbox, txt, conf in results:
        texts.append((txt, float(conf)))

    return texts


# =========================
# ラベル抽出
# =========================
def extract_best_label(results):
    best = None
    best_score = -1

    for txt, conf in results:
        digits = re.sub(r"\D", "", txt)

        if len(digits) == 6:
            score = conf

            if score > best_score:
                best_score = score
                best = f"{digits[:2]}-{digits[2:4]}-{digits[4:]}"

    return best


# =========================
# ROI（高さ固定）
# =========================
def crop_band(frame_rot):
    H, W = frame_rot.shape[:2]

    y1 = int(H * 0.42)
    y2 = int(H * 0.58)

    return frame_rot[y1:y2, :]


def make_label_roi_strict(frame_rot, bbox):
    band = crop_band(frame_rot)

    x1, y1, x2, y2 = bbox
    bw = x2 - x1

    # 👉 あなたが調整した広めROI
    X1 = int(x2 + bw * 0.00)
    X2 = int(x2 + bw * 4.00)

    H_band, W_band = band.shape[:2]

    X1 = max(0, min(W_band - 1, X1))
    X2 = max(0, min(W_band, X2))

    if X2 <= X1:
        return band

    return band[:, X1:X2]


# =========================
# 描画
# =========================
def draw_bbox(img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


# =========================================
# main
# =========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ===== 画像読み込み =====
    img = cv2.imread(args.image)
    if img is None:
        print("❌ Failed to load image")
        return

    # ===== 180°回転 =====
    img_rot = cv2.rotate(img, cv2.ROTATE_180)
    cv2.imwrite(str(outdir / "debug_rotated.png"), img_rot)

    # ===== バーコード検出 =====
    bbox = detect_barcode_bbox(img_rot)

    if bbox is None:
        print("❌ No barcode detected")
        return

    print(f"[OK] bbox = {bbox}")

    # ===== 可視化 =====
    vis = draw_bbox(img_rot, bbox)
    cv2.imwrite(str(outdir / "debug_bbox.png"), vis)

    # ===== ROI =====
    roi = make_label_roi_strict(img_rot, bbox)
    cv2.imwrite(str(outdir / "dbg_ocr_roi_strict.png"), roi)

    # ===== OCR =====
    results = run_ocr_easy(roi)

    print("\n========== OCR RAW ==========")
    for txt, conf in results:
        print(f"{txt} ({conf:.3f})")

    # ===== ラベル抽出 =====
    label = extract_best_label(results)

    print("\n========== PARSED ==========")
    print(label)

    # ===== 保存 =====
    with open(outdir / "result.txt", "w") as f:
        f.write("RAW:\n")
        for txt, conf in results:
            f.write(f"{txt} ({conf:.3f})\n")
        f.write(f"\nlabel: {label}\n")

    print(f"\n[OK] saved to {outdir}")


if __name__ == "__main__":
    main()