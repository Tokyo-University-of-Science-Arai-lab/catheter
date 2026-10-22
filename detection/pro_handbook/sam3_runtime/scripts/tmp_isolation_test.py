#!/usr/bin/env python3
"""一時調査用(2026-10-12)。SHOURYU2の箱だけを切り出して単独でSAM3に通し、
スコアが変わるか確認する。本番コードは変更しない。"""
from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_DIR.parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from detection.pro_handbook.sam3_runtime.service.model_loader import load_model  # noqa: E402

FULL_RGB = (
    RUNTIME_DIR.parents[2]
    / "captures/resolution_test/4/ocr_test_reco_result_1920x1080_20261006_202703/work/1/after_init_rgb.png"
)


def main() -> None:
    print("モデル読み込み中...")
    loaded = load_model()
    adapter = loaded.adapter
    print(f"読み込み完了 ({loaded.model_load_seconds:.1f}秒)")

    full = cv2.imread(str(FULL_RGB))
    crop = full[:, 760:910]
    crop_path = "/tmp/isolated_shouryu2.png"
    cv2.imwrite(crop_path, crop)

    instances = adapter.predict(crop_path, prompt="book spine", score_threshold=0.0, min_area=0)
    scores = instances.scores
    boxes = instances.bboxes.astype(np.float32)
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1

    order = np.argsort(-scores)
    print(f"\n単独切り出し後の全候補数: {len(scores)}")
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        flag = "" if scores[i] >= 0.3 else "  <-- 閾値0.3未満"
        print(f"  score={scores[i]:.4f}  box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}){flag}")

    print("\n(元の画像全体の中では score=0.1309 でした)")


if __name__ == "__main__":
    main()
