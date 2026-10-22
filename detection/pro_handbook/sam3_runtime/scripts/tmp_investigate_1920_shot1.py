#!/usr/bin/env python3
"""
一時調査用(2026-10-12)。1920x1080 shot1で、スコア閾値0.3未満で除外された
候補が無いか、画像全体を対象に確認する。本番コードは変更しない。
"""
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
    / "captures/resolution_test/4/ocr_test_reco_result_1920x1080_20261006_202703/work/1/after_init_rgb.png"
)


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
    for i in order:
        x0, y0, x1, y1 = boxes[i]
        flag = "" if scores[i] >= 0.3 else "  <-- 閾値0.3未満(通常は除外される)"
        print(f"  score={scores[i]:.4f}  box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}){flag}")

    # score=0.1309の候補(x=781,102,890,955)を画像化
    rgb = cv2.imread(str(RGB_PATH))
    h, w = rgb.shape[:2]
    for target_score, out_name in [(0.1309, "1920_shot1_gap_candidate_score013.png")]:
        idx = int(np.argmin(np.abs(scores - target_score)))
        x0, y0, x1, y1 = boxes[idx]
        margin = 100
        cx0, cy0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
        cx1, cy1 = min(w, int(x1 + margin)), min(h, int(y1 + margin))
        crop = rgb[cy0:cy1, cx0:cx1].copy()
        mask_crop = masks[idx][cy0:cy1, cx0:cx1]
        overlay = crop.copy()
        overlay[mask_crop] = (0.4 * overlay[mask_crop] + 0.6 * np.array([0, 0, 255])).astype("uint8")
        cv2.rectangle(overlay, (int(x0 - cx0), int(y0 - cy0)), (int(x1 - cx0), int(y1 - cy0)), (0, 255, 255), 3)
        cv2.putText(overlay, f"score={scores[idx]:.3f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA)
        out_path = RUNTIME_DIR.parents[2] / "captures/resolution_test/4/sam3_resize_check" / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)
        print(f"画像保存: {out_path}")


if __name__ == "__main__":
    main()
