#!/usr/bin/env python3
"""一時調査用(2026-10-13)。processorのconfidence_threshold(既定0.05)自体を
0にして、通常は出力段階で完全に捨てられる候補まで見えるようにする。
1280x720 shot3のSHOURYU2(SS2-40-070H)周辺(x=700〜950)を対象に、
一番それらしい(隙間に近い)候補を画像化する。本番コードは変更しない。"""
from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_DIR.parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from detection.pro_handbook.sam3_runtime.service.model_loader import load_model  # noqa: E402

RGB_PATH = (
    RUNTIME_DIR.parents[2]
    / "captures/resolution_test/4/ocr_test_reco_result_1280x720_20261007_173131/work/3/after_init_rgb.png"
)
OUT_DIR = RUNTIME_DIR.parents[2] / "captures/resolution_test/4/sam3_resize_check"

# SHOURYU2の隙間はSpiderFX(mask_21, x=597-749)とAXS Vecta(x=754-948)の間
GAP_X1, GAP_X2 = 749, 754


def main() -> None:
    print("モデル読み込み中...")
    loaded = load_model()
    adapter = loaded.adapter
    print(f"読み込み完了 ({loaded.model_load_seconds:.1f}秒)")

    adapter.processor.confidence_threshold = 0.0

    instances = adapter.predict(str(RGB_PATH), prompt="book spine", score_threshold=0.0, min_area=0)
    boxes = instances.bboxes.astype(np.float32)
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1
    scores = instances.scores
    masks = instances.masks

    # 隙間(x=749-754)に最も近い中心を持つ候補を、スコア降順で探す
    candidates = []
    for i in range(len(scores)):
        x0, y0, x1, y1 = boxes[i]
        cx = (x0 + x1) / 2
        if 650 <= cx <= 850:
            candidates.append((i, scores[i], x0, y0, x1, y1))
    candidates.sort(key=lambda c: -c[1])

    print(f"x=650〜850の全候補({len(candidates)}件):")
    for i, s, x0, y0, x1, y1 in candidates:
        print(f"  idx={i} score={s:.5f} box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")

    below_th = [c for c in candidates if c[1] < 0.3]
    print(f"\n足切り(score<0.3)された候補: {len(below_th)}件 → 全て画像化します")

    rgb = cv2.imread(str(RGB_PATH))
    h, w = rgb.shape[:2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n, (i, s, x0, y0, x1, y1) in enumerate(sorted(below_th, key=lambda c: -c[1]), start=1):
        margin = 100
        cx0, cy0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
        cx1, cy1 = min(w, int(x1 + margin)), min(h, int(y1 + margin))
        crop = rgb[cy0:cy1, cx0:cx1].copy()
        mask_crop = masks[i][cy0:cy1, cx0:cx1]
        overlay = crop.copy()
        overlay[mask_crop] = (0.4 * overlay[mask_crop] + 0.6 * np.array([0, 0, 255])).astype("uint8")
        cv2.rectangle(overlay, (int(x0 - cx0), int(y0 - cy0)), (int(x1 - cx0), int(y1 - cy0)), (0, 255, 255), 3)
        cv2.putText(overlay, f"score={s:.4f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
        out_path = OUT_DIR / f"1280_shot3_gap_rejected_{n}_score{s:.4f}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"画像保存: {out_path}")


if __name__ == "__main__":
    main()
