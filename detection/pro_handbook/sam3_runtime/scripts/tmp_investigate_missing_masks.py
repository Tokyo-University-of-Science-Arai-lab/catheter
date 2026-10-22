#!/usr/bin/env python3
"""
SAM3の通常のスコア閾値(0.3)でフィルタされる前の、全候補(confidence_threshold=0.05
より上のもの全て)を確認するための調査用スクリプト(2026-10-12、一時的な調査用。
本番コード(service/predictor.py, vendor/owner_repo/core/sam3_adapter.py)は変更しない)。

4032x2268で検出漏れがあった箱について、x座標付近に「惜しい」低スコア候補が
実際には存在していたのかどうかを確認する。

使い方:
    python tmp_investigate_missing_masks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_DIR.parents[2]))  # repo root

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from detection.pro_handbook.sam3_runtime.service.model_loader import load_model  # noqa: E402

TARGETS = {
    "1": 2200,
    "2": 2400,
    "4": 1600,
    "5": 2400,
}

BASE = Path(
    "captures/resolution_test/4/no_depth_reco_4032x2268_20261007_194756/work"
)

WINDOW = 200  # 対象x座標から±200pxの範囲にある候補を表示


def main() -> None:
    print("モデル読み込み中...")
    loaded = load_model()
    adapter = loaded.adapter
    print(f"読み込み完了 ({loaded.model_load_seconds:.1f}秒)")

    import numpy as np

    for shot, target_x in TARGETS.items():
        rgb_path = RUNTIME_DIR.parents[2] / BASE / shot / "after_init_rgb.png"
        # adapter.predict()を直接呼ぶ(score_threshold=0.0にして、通常は捨てられる
        # confidence_threshold(0.05)〜0.3の候補も全部残す)。autocast等の
        # dtype処理はadapter.predict()内部でやってくれるので、ここでは
        # 直接呼ばずにpredict()経由にする。
        instances = adapter.predict(str(rgb_path), prompt="book spine", score_threshold=0.0, min_area=0)
        boxes = instances.bboxes.astype(np.float32)  # xywh
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1
        scores = instances.scores
        masks = instances.masks

        print(f"\n=== shot {shot} (対象x={target_x}, 全候補数={len(scores)}) ===")
        hits = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x0, y0, x1, y1 = box
            cx = (x0 + x1) / 2
            if abs(cx - target_x) <= WINDOW:
                hits.append((float(score), float(x0), float(y0), float(x1), float(y1), i))
        hits.sort(key=lambda h: -h[0])
        if not hits:
            print(f"  x={target_x}±{WINDOW}の範囲に候補は1つもありません(confidence>0.05でも0件)")
            continue

        widest = max(hits, key=lambda h: h[3] - h[1])
        for score, x0, y0, x1, y1, i in hits:
            flag = "" if score >= 0.3 else "  <-- 通常の閾値0.3未満(通常は捨てられる)"
            mark = "  [最も幅が広い候補 -> 画像化]" if i == widest[5] else ""
            print(f"  score={score:.4f}  box=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}){flag}{mark}")

        # 最も幅が広い候補を、元画像に重ねて保存(margin付きでクロップ)
        score, x0, y0, x1, y1, i = widest
        rgb = cv2.imread(str(rgb_path))
        h, w = rgb.shape[:2]
        margin = 150
        cx0, cy0 = max(0, int(x0 - margin)), max(0, int(y0 - margin))
        cx1, cy1 = min(w, int(x1 + margin)), min(h, int(y1 + margin))
        crop = rgb[cy0:cy1, cx0:cx1].copy()
        mask_crop = masks[i][cy0:cy1, cx0:cx1]
        overlay = crop.copy()
        overlay[mask_crop] = (0.4 * overlay[mask_crop] + 0.6 * np.array([0, 0, 255])).astype("uint8")
        cv2.rectangle(
            overlay,
            (int(x0 - cx0), int(y0 - cy0)),
            (int(x1 - cx0), int(y1 - cy0)),
            (0, 255, 255),
            3,
        )
        cv2.putText(
            overlay, f"score={score:.3f} width={int(x1-x0)}px",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3, cv2.LINE_AA,
        )
        out_path = RUNTIME_DIR.parents[2] / "captures/resolution_test/4/sam3_resize_check" / f"shot{shot}_wide_low_score_candidate.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"  画像保存: {out_path}")


if __name__ == "__main__":
    main()
