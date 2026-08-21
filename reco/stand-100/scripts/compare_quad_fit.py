#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
既存の catheter-100/annotations/instances_default.json のポリゴンアノテーションを
「SAM3が生成したマスク」の代わりの入力として使い、
- 方式A: min_area_rect（最小外接矩形）
- 方式B: approx_poly_quad（凸包を4点近似）
の2方式でマスクを4角形に整形し、元マスクとのIoU・ギザギザ度を比較する。

入力（読み取り専用、一切書き換えない）:
    pro_hand_book_python/catheter-100/annotations/instances_default.json
    pro_hand_book_python/catheter-100/images/default/*.png

出力（新規作成のみ）:
    pro_hand_book_python/catheter/outputs/summary.csv
    pro_hand_book_python/catheter/outputs/overlays/<image_id>_<ann_id>.png   (1件ずつの拡大比較)
    pro_hand_book_python/catheter/outputs/full_image/<image_id>.png         (画像全体の3パネル比較)

実行:
    python3 pro_hand_book_python/catheter/scripts/compare_quad_fit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from quad_fit import fit_quad_approx_poly, fit_quad_min_area_rect, quad_to_mask

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../pro_hand_book_python
SRC_DIR = REPO_ROOT / "catheter-100"
ANNOTATIONS_JSON = SRC_DIR / "annotations" / "instances_default.json"
IMAGES_DIR = SRC_DIR / "images" / "default"

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
OVERLAY_DIR = OUT_DIR / "overlays"
FULL_IMAGE_DIR = OUT_DIR / "full_image"

CROP_MARGIN_PX = 25

COLOR_ORIGINAL = (0, 255, 0)     # green: 元マスクの輪郭
COLOR_MIN_AREA_RECT = (0, 0, 255)  # red:   方式A
COLOR_APPROX_POLY = (255, 128, 0)  # blue-orange: 方式B


def decode_uncompressed_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    """
    COCOの非圧縮RLE（counts が int のリスト）を0/1マスクにデコードする。
    列優先（Fortran順）で 0/1 が交互に続くランレングス形式。
    pycocotools が無い環境向けの最小実装。
    """
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if c:
            flat[idx: idx + c] = val
        idx += c
        val = 1 - val
    return flat.reshape((height, width), order="F")


def segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """COCOセグメンテーション（ポリゴン形式 or 非圧縮RLE形式）を0/1マスクにラスタライズする。"""
    if isinstance(segmentation, dict):
        counts = segmentation["counts"]
        if isinstance(counts, str):
            raise NotImplementedError(
                "圧縮RLE（counts が文字列）はこのスクリプトでは未対応です（pycocotools未導入のため）。"
            )
        return decode_uncompressed_rle(counts, height, width)

    mask = np.zeros((height, width), dtype=np.uint8)
    for part in segmentation:
        pts = np.array(part, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def jaggedness_ratio(mask: np.ndarray) -> float | None:
    """元輪郭長 / 凸包輪郭長。1.0に近いほど滑らか、大きいほどギザギザ。"""
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri_cnt = cv2.arcLength(cnt, True)
    peri_hull = cv2.arcLength(hull, True)
    if peri_hull < 1e-6:
        return None
    return float(peri_cnt / peri_hull)


def draw_quad(img: np.ndarray, quad: np.ndarray | None, color, thickness=2) -> None:
    if quad is None:
        return
    pts = np.round(quad).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_mask_contour(img: np.ndarray, mask: np.ndarray, color, thickness=2) -> None:
    m = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness, lineType=cv2.LINE_AA)


def save_crop_comparison(
    bgr_image: np.ndarray,
    mask: np.ndarray,
    quad_mar: np.ndarray | None,
    quad_approx: np.ndarray | None,
    bbox_xywh: list,
    out_path: Path,
) -> None:
    h, w = mask.shape
    x, y, bw, bh = bbox_xywh
    x0 = max(0, int(x - CROP_MARGIN_PX))
    y0 = max(0, int(y - CROP_MARGIN_PX))
    x1 = min(w, int(x + bw + CROP_MARGIN_PX))
    y1 = min(h, int(y + bh + CROP_MARGIN_PX))

    canvas = bgr_image.copy()
    draw_mask_contour(canvas, mask, COLOR_ORIGINAL, thickness=2)
    draw_quad(canvas, quad_mar, COLOR_MIN_AREA_RECT, thickness=2)
    draw_quad(canvas, quad_approx, COLOR_APPROX_POLY, thickness=2)

    crop = canvas[y0:y1, x0:x1]
    if crop.size == 0:
        return

    scale = 3 if max(crop.shape[:2]) < 200 else 1
    if scale != 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def build_full_image_panels(
    bgr_image: np.ndarray,
    entries: list[dict],
) -> np.ndarray:
    """
    1枚の画像につき、
      左パネル : 元マスク輪郭のみ
      中央パネル: 方式A（min_area_rect）
      右パネル  : 方式B（approx_poly_quad）
    を横に並べたパネル画像を作る。
    """
    panel_original = bgr_image.copy()
    panel_mar = bgr_image.copy()
    panel_approx = bgr_image.copy()

    for e in entries:
        draw_mask_contour(panel_original, e["mask"], COLOR_ORIGINAL, thickness=2)
        draw_quad(panel_mar, e["quad_mar"], COLOR_MIN_AREA_RECT, thickness=2)
        draw_quad(panel_approx, e["quad_approx"], COLOR_APPROX_POLY, thickness=2)

    def label(panel, text):
        cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1, cv2.LINE_AA)
        return panel

    label(panel_original, "original mask (jagged)")
    label(panel_mar, "A: min_area_rect")
    label(panel_approx, "B: approx_poly_quad")

    sep = np.full((bgr_image.shape[0], 4, 3), 255, dtype=np.uint8)
    return np.hstack([panel_original, sep, panel_mar, sep, panel_approx])


def main() -> None:
    with open(ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {im["id"]: im for im in data["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    FULL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for image_id, image_info in sorted(images_by_id.items()):
        img_path = IMAGES_DIR / image_info["file_name"]
        pil_bgr = cv2.imread(str(img_path))
        if pil_bgr is None:
            print(f"⚠ 画像を読み込めません: {img_path}")
            continue

        height, width = image_info["height"], image_info["width"]
        entries = []

        for ann in anns_by_image.get(image_id, []):
            mask = segmentation_to_mask(ann["segmentation"], height, width)

            quad_mar = fit_quad_min_area_rect(mask)
            quad_approx = fit_quad_approx_poly(mask)

            mask_area = int(mask.sum())
            jagged = jaggedness_ratio(mask)

            iou_mar = iou(mask, quad_to_mask(quad_mar, (height, width))) if quad_mar is not None else None
            iou_approx = (
                iou(mask, quad_to_mask(quad_approx, (height, width))) if quad_approx is not None else None
            )

            rows.append(
                {
                    "image_id": image_id,
                    "ann_id": ann["id"],
                    "file_name": image_info["file_name"],
                    "mask_area_px": mask_area,
                    "jaggedness_ratio": jagged,
                    "min_area_rect_ok": quad_mar is not None,
                    "min_area_rect_iou": iou_mar,
                    "approx_poly_quad_ok": quad_approx is not None,
                    "approx_poly_quad_iou": iou_approx,
                }
            )

            entries.append(
                {
                    "mask": mask,
                    "quad_mar": quad_mar,
                    "quad_approx": quad_approx,
                }
            )

            crop_out = OVERLAY_DIR / f"{image_id}_{ann['id']}.png"
            save_crop_comparison(pil_bgr, mask, quad_mar, quad_approx, ann["bbox"], crop_out)

        panel_img = build_full_image_panels(pil_bgr, entries)
        cv2.imwrite(str(FULL_IMAGE_DIR / f"{image_id}.png"), panel_img)
        print(f"✔ image_id={image_id} ({image_info['file_name']}): {len(entries)} instances -> {FULL_IMAGE_DIR / f'{image_id}.png'}")

    # summary.csv
    csv_path = OUT_DIR / "summary.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 集計
    n = len(rows)
    mar_ious = [r["min_area_rect_iou"] for r in rows if r["min_area_rect_iou"] is not None]
    approx_ious = [r["approx_poly_quad_iou"] for r in rows if r["approx_poly_quad_iou"] is not None]
    approx_fail = sum(1 for r in rows if not r["approx_poly_quad_ok"])

    print("\n===== SUMMARY =====")
    print(f"total instances            : {n}")
    print(f"A) min_area_rect  mean IoU : {np.mean(mar_ious):.4f}  (n={len(mar_ious)})")
    print(f"B) approx_poly_quad mean IoU: {np.mean(approx_ious):.4f}  (n={len(approx_ious)})")
    print(f"B) approx_poly_quad 失敗数  : {approx_fail} / {n}  (4点にきれいに収束しなかった件数)")
    print(f"\nCSV     : {csv_path}")
    print(f"crops   : {OVERLAY_DIR}")
    print(f"panels  : {FULL_IMAGE_DIR}")


if __name__ == "__main__":
    main()
