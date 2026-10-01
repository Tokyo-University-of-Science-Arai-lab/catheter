#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_image_match_test.py の結果フォルダ(work/<画像番号>/)にある
sam3_service_masks.npz(SAM3が検出した生のマスク配列)から、マスク番号
(mask_1, mask_2, ...)ごとに形だけを切り出した画像を work/<画像番号>/mask_images/
にまとめる(2026-10-01追加、ユーザー要望:「各マスクの形を確認したい」)。

マスク番号の振り方(SAM3の検出スコア降順。画像内の左右位置とは無関係。
詳細は mask_nms.py の apply_mask_nms() 参照)は multikey_matcher.py や
sam3_all_masks_overlay.png の番号と揃えてある(masks配列のインデックスに
そのまま mask_{i+1} と命名)。

出力画像は、そのマスクのbounding boxに余白を付けて切り取り、マスク外の
部分は背景色(薄い灰色)に置き換えたもの(マスクの形だけが分かるように)。

2026-10-01時点、single_image_match_test.py実行時にこの処理は自動で呼ばれるように
なっている(_export_mask_shapes())。このスクリプトを手動で実行する必要があるのは、
それより前に作られた結果フォルダに後から追加したい場合のみ。

実行:
    python3 reco/scripts/export_mask_shape_images.py \
        --result-dir reco/0911_crop10/single_image_match_reco_result_20261001_103957
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

PAD = 12
BG_COLOR = (235, 235, 235)


def export_shot(shot_dir: Path) -> int:
    npz_path = shot_dir / "sam3_service_masks.npz"
    rgb_path = shot_dir / "after_init_rgb.png"
    if not npz_path.exists() or not rgb_path.exists():
        return 0

    data = np.load(npz_path)
    masks = data["masks"]
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    h, w = rgb.shape[:2]

    out_dir = shot_dir / "mask_images"
    out_dir.mkdir(exist_ok=True)

    n = 0
    for i, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        x0, x1 = max(int(xs.min()) - PAD, 0), min(int(xs.max()) + PAD + 1, w)
        y0, y1 = max(int(ys.min()) - PAD, 0), min(int(ys.max()) + PAD + 1, h)

        crop_rgb = rgb[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        bg = np.full_like(crop_rgb, BG_COLOR)
        out = np.where(crop_mask[..., None], crop_rgb, bg)

        Image.fromarray(out, mode="RGB").save(out_dir / f"mask_{i + 1}.png")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", required=True,
                    help="single_image_match_test.py --dataset で作られた reco_result フォルダ")
    args = ap.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    work_root = result_dir / "work"
    if not work_root.is_dir():
        raise FileNotFoundError(f"{work_root} がありません")

    total = 0
    for shot_dir in sorted(work_root.iterdir(), key=lambda p: p.name):
        if not shot_dir.is_dir():
            continue
        n = export_shot(shot_dir)
        if n:
            print(f"✔ {shot_dir.name}: {n}枚 -> {shot_dir / 'mask_images'}")
            total += n
    print(f"合計 {total}枚")


if __name__ == "__main__":
    main()
