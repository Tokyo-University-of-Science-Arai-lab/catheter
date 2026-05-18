#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
input/ 内の画像を infer_for_retrival.SamBatchInfer_retrieval で推論し，
GTs/ 内のカラーラベル GT と比較して輪郭誤差ベースの認識成功率を評価する。

評価条件:
  ・GTマスク M_G と推論マスク M の輪郭の Hausdorff 距離 d_H(M_G, M) を計算
  ・d_H <= tol_px (デフォルト 10px) を満たす推論マスクが1つでもあれば，
    その GT インスタンスは「認識成功」とみなす。
"""

import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2

# infer_for_retrival.py からクラスを import
from .infer_for_retrival import SamConfig, SamBatchInfer_retrieval  # ファイル名に合わせてください


# ============================================================
#  GT: カラーラベル画像 → 各色をインスタンスマスクに分解
# ============================================================

def masks_from_color_instances(path: str) -> List[np.ndarray]:
    """
    カラーラベル画像から、非黒の各“色(=RGB)”を1インスタンスとして抽出。
    返り値: List[(H,W) uint8 0/1]
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    # BGR -> 1値にパック
    b, g, r = cv2.split(img)
    packed = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)

    uniq = np.unique(packed)
    uniq = uniq[uniq != 0]  # 黒(0,0,0)は背景

    masks: List[np.ndarray] = []
    for code in uniq:
        m = (packed == code).astype(np.uint8)
        if m.sum() > 0:
            masks.append(m)
    if not masks:
        raise ValueError(f"No non-black color instances found in {path}")
    return masks


# ============================================================
#  輪郭抽出 & Hausdorff 距離
# ============================================================

def mask_to_boundary(mask: np.ndarray) -> np.ndarray:
    """
    2値マスクから 1px 幅の輪郭画像を作る。
    出力は 0/1 の uint8 (H,W)。
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    kernel = np.ones((3, 3), np.uint8)
    # MORPH_GRADIENT = dilate - erode → 周囲1px分が輪郭になる
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
    grad[grad != 0] = 1
    return grad


def hausdorff_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    2つのマスクの輪郭同士の Hausdorff 距離 d_H を返す。
      d_H(A,B) = max( sup_{a∈∂A} inf_{b∈∂B} ||a-b||,
                      sup_{b∈∂B} inf_{a∈∂A} ||a-b|| )
    を距離変換を用いて高速に近似。
    """
    ba = mask_to_boundary(mask_a)
    bb = mask_to_boundary(mask_b)

    if ba.sum() == 0 or bb.sum() == 0:
        return float("inf")

    # distanceTransform は「0ピクセルまでの距離」を返すので、
    # 輪郭を 0、それ以外を 1 にして距離変換をとる
    da = cv2.distanceTransform(1 - ba, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(1 - bb, cv2.DIST_L2, 3)

    ya, xa = np.where(ba > 0)
    yb, xb = np.where(bb > 0)

    # Aの輪郭各点から B への最短距離の最大値
    d_ab = float(db[ya, xa].max())
    # Bの輪郭各点から A への最短距離の最大値
    d_ba = float(da[yb, xb].max())

    return max(d_ab, d_ba)


def pass_contour_threshold(gt_mask: np.ndarray, pred_mask: np.ndarray, tol_px: float) -> bool:
    """
    GTと推論マスクの輪郭 Hausdorff 距離が tol_px 以下かどうかを判定。
    """
    d = hausdorff_distance(gt_mask, pred_mask)
    print(f"[DEBUG] pass_contour_threshold: Hausdorff distance = {d:.3f} px")
    return d <= tol_px


# ============================================================
#  1枚の画像に対する評価
# ============================================================

def eval_one_image(
    gt_path: Path,
    pred_masks: List[np.ndarray],
    tol_px: float
) -> Tuple[int, int]:
    """
    1画像について、GTカラーラベルPNGと推論マスク群を比較して
    「認識成功した GT インスタンス数」と「GT インスタンス総数」を返す。
    """
    gt_masks = masks_from_color_instances(str(gt_path))
    H, W = gt_masks[0].shape

    # 推論マスクを GT と同じサイズ＆0/1に統一
    preds: List[np.ndarray] = []
    # for m in pred_masks:
    #     m = np.asarray(m)
    #     if m.shape != (H, W):
    #         m = cv2.resize((m != 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    #     preds.append((m != 0).astype(np.uint8))

    success = 0
    total = len(gt_masks)
    i = 0
    for g in gt_masks:
        print(f"[DEBUG] Evaluating GT{i}")
        g_bin = (g != 0).astype(np.uint8)
        ok = False
        for p in preds:
            if pass_contour_threshold(g_bin, p, tol_px):
                ok = True
                break
        if ok:
            success += 1
        i += 1
    return success, total


# ============================================================
#  データセット全体のループ
# ============================================================

def run_eval(
    input_dir: Path,
    gt_dir: Path,
    tol_px: float,
    runner: SamBatchInfer_retrieval,
    csv_path: Path | None = None,
) -> float:
    """
    input_dir の画像それぞれに対して infer → GT と比較 → 成功率を集計。
    """
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    img_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXT])
    if not img_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    rows: List[Dict[str, Any]] = []
    total_success = 0
    total_gt = 0

    for img_path in img_paths:
        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.png"
        if not gt_path.exists():
            print(f"[WARN] GT not found for {img_path.name} → skip")
            continue

        print(f"==> {img_path.name}")

        # SAM 推論（skip_selection=True で対話なし）
        res = runner.run_on_image_path(
            img_path,
            stage_save_dir=None,   # infer_for_retrival.DEFAULT_OUT_DIR が使われる
            skip_selection=True,
        )
        print(res.keys())
        print("Number of predicted masks:", len(res.get("masks", [])))
        pred_masks = res.get("masks", [])
        if not pred_masks:
            print("   NO PREDICTED MASKS → all GT fail")
            suc, tot = 0, len(masks_from_color_instances(str(gt_path)))
        else:
            suc, tot = eval_one_image(gt_path, pred_masks, tol_px)

        rate = suc / tot if tot > 0 else 0.0
        print(f"   success={suc} / {tot}  rate={rate:.6f}")

        total_success += suc
        total_gt += tot
        rows.append({
            "basename": stem,
            "success": suc,
            "gt_total": tot,
            "rate": rate,
        })

    overall = total_success / total_gt if total_gt > 0 else 0.0
    print("\n==============================")
    print(f"OVERALL  success={total_success} / {total_gt}  rate={overall:.6f}")
    print("==============================")

    if csv_path is not None:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["basename", "success", "gt_total", "rate"])
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {csv_path}")

    return overall


# ============================================================
#  CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="書籍マスクの輪郭誤差 (Hausdorff距離) による認識成功率評価スクリプト"
    )
    ap.add_argument("--input_dir", help="RGB画像フォルダ (例: ./input)", default = "/home/tai/Downloads/input")
    ap.add_argument("--gt_dir", help="GTカラーラベルPNGフォルダ (例: ./GTs)", default = "/home/tai/Downloads/GTs") 
    ap.add_argument("--encoder", default="/home/tai/pro_handbook/pro_hand_book_python/models/sam_vit_h_4b8939.encoder.onnx")
    ap.add_argument("--decoder", default="/home/tai/pro_handbook/pro_hand_book_python/models/sam_vit_h_4b8939.decoder.onnx")
    ap.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")
    ap.add_argument("--tol", type=float, default=10.0, help="輪郭誤差の許容 [px]")
    ap.add_argument("--csv", type=str, default=None, help="結果CSVの保存先 (任意)")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    gt_dir = Path(args.gt_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    if not gt_dir.is_dir():
        raise NotADirectoryError(gt_dir)

    # SAM 設定（必要ならここに min_area などを追加で指定）
    cfg = SamConfig(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        device=args.device,
    )
    runner = SamBatchInfer_retrieval(cfg)

    csv_path = Path(args.csv) if args.csv else None
    run_eval(input_dir, gt_dir, tol_px=args.tol, runner=runner, csv_path=csv_path)


if __name__ == "__main__":
    main()
