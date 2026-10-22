#!/usr/bin/env python3
"""一時調査用(2026-10-13)。1920x1080 shot3で、スコア閾値0.3未満で除外された
候補が無いか確認する(SHOURYU2/SS2-40-070H探索用)。本番コードは変更しない。"""
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
    / "captures/resolution_test/4/ocr_test_reco_result_1920x1080_20261006_202703/work/3/after_init_rgb.png"
)
OUT_DIR = RUNTIME_DIR.parents[2] / "captures/resolution_test/4/sam3_resize_check"


def main() -> None:
    print("モデル読み込み中...")
    loaded = load_model()
    adapter = loaded.adapter
    print(f"読み込み完了 ({loaded.model_load_seconds:.1f}秒)")

    instances = adapter.predict(str(RGB_PATH), prompt="book spine", score_threshold=0.0, min_area=0)
    boxes = instances.bboxes.astype(np.float32)
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1
    scores = instances.scores
    masks = instances.masks

    order = np.argsort(-scores)
    print(f"\n全候補数(confidence>0.05): {len(scores)}")
    below = []
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        flag = "" if scores[i] >= 0.3 else "  <-- 閾値0.3未満(通常は除外される)"
        print(f"  score={scores[i]:.4f}  box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}){flag}")
        if scores[i] < 0.3:
            below.append(i)

    rgb = cv2.imread(str(RGB_PATH))
    h, w = rgb.shape[:2]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(below, start=1):
        x0, y0, x1, y1 = boxes[i]
        margin = 80
        cx0, cy0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
        cx1, cy1 = min(w, int(x1 + margin)), min(h, int(y1 + margin))
        crop = rgb[cy0:cy1, cx0:cx1].copy()
        mask_crop = masks[i][cy0:cy1, cx0:cx1]
        overlay = crop.copy()
        overlay[mask_crop] = (0.4 * overlay[mask_crop] + 0.6 * np.array([0, 0, 255])).astype("uint8")
        cv2.rectangle(overlay, (int(x0 - cx0), int(y0 - cy0)), (int(x1 - cx0), int(y1 - cy0)), (0, 255, 255), 3)
        cv2.putText(overlay, f"score={scores[i]:.3f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
        out_path = OUT_DIR / f"1920_shot3_below_threshold_{n}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"画像保存: {out_path}")

    if not below:
        print("\n閾値未満の候補はありませんでした。")


if __name__ == "__main__":
    main()
