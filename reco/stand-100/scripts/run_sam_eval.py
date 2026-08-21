#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM (infer_for_storage.SamBatchInfer_storage) を catheter-100 の5棚画像に実際に
走らせ、次の3種類の精度を算出する。

  1. IoU精度      : 予測マスク vs 正解COCOポリゴンマスクのIoU
  2. 幅の精度(px)  : mask_rectify.min_area_rect_box() で矩形化した短辺長の誤差
  3. 識別精度       : OCR照合結果をmaster_catheter_20260216.jsonに突き合わせ、
                      正解マスクベースの識別結果(catheter/outputs/match_multi/assignment.csv)
                      と一致するか

pro_book配下のコード・catheter-100配下のデータ・catheter/outputs配下の既存出力は
一切変更しない（読み取り専用）。新規出力は
catheter/outputs/sam_recognition_eval/ にのみ書き込む。

実行:
    cd ~/pro_book/pro_hand_book_python
    .pro_hand_book_fixed/bin/python3.10 \
        ~/ダウンロード/catheter_test80/catheter/scripts/run_sam_eval.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# onnxruntime-gpu が cuDNN/cuBLAS 等を見つけられるよう、venv内 nvidia-* パッケージの
# lib ディレクトリを LD_LIBRARY_PATH に積んでから自分自身を再起動する。
# (recognition_accuracy_test.py の手法を、このマシンの実パスに合わせて再実装)
# ---------------------------------------------------------------------------
def _ensure_cuda_ld_library_path() -> None:
    site_packages = Path(sys.executable).resolve().parents[1] / "lib" / "python3.10" / "site-packages"
    nvidia_dir = site_packages / "nvidia"
    if not nvidia_dir.is_dir():
        return
    lib_dirs = [str(p) for p in sorted(nvidia_dir.glob("*/lib")) if p.is_dir()]
    if not lib_dirs:
        return
    joined = ":".join(lib_dirs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d not in cur for d in lib_dirs):
        os.environ["LD_LIBRARY_PATH"] = joined + (":" + cur if cur else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_ld_library_path()

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# パス設定
# ---------------------------------------------------------------------------
# pro_book のコード(SAM推論本体)は catheter_test80 とは別のディレクトリツリーにある。
REPO_ROOT = Path.home() / "pro_book" / "pro_hand_book_python"
CATHETER_ROOT = Path(__file__).resolve().parents[1]  # .../catheter_test80/catheter
SRC = CATHETER_ROOT.parent / "catheter-100"
IMAGES_DIR = SRC / "images" / "default"
ANNOTATIONS_JSON = SRC / "annotations" / "instances_default.json"
MASTER_JSON = SRC / "master_catheter_20260216.json"

OCR_DIR = CATHETER_ROOT / "outputs" / "ocr"  # 既存出力(単一角度OCR)、読み取り専用
REF_ASSIGNMENT_CSV = CATHETER_ROOT / "outputs" / "match_multi" / "assignment.csv"  # 読み取り専用

OUT_DIR = CATHETER_ROOT / "outputs" / "sam_recognition_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR = OUT_DIR / "pred_masks_cache"
MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR = OUT_DIR / "overlays"
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

SAM_ENCODER = REPO_ROOT / "models" / "sam_vit_h_4b8939.encoder.onnx"
SAM_DECODER = REPO_ROOT / "models" / "sam_vit_h_4b8939.decoder.onnx"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CATHETER_ROOT / "scripts"))

IOU_MATCH_THRESHOLD = 0.5

# 4枚目の画像はユーザー指示により評価対象から除外する(4枚 x 20クエリ = 80件で評価)。
EXCLUDED_IMAGE_IDS = {4}

# SAM推論パラメータ。1280x720の棚全体画像で箱が多数並ぶため、デフォルトの
# pts_side=(48,12) をやや密にして拾い漏れを減らす。
SAM_PTS_SIDE = (64, 16)
SAM_MIN_AREA = 500

# infer_masks() 内部のdedup(box-IoU 0.7 / mask-IoU 0.97)を通っても、この
# データセットでは20個のGTに対し予測33〜38件が残ることが分かっている(HANDOFF参照)。
# infer_for_storage.py本体(実機の把持動作にも使われる)は変更せず、この評価スクリプト
# 側だけで追加のmask-IoU NMSをかけて間引く。
DEDUP_IOU_THRESHOLD = 0.5


def dedup_masks_by_iou(masks: list[np.ndarray], iou_thr: float = DEDUP_IOU_THRESHOLD):
    """予測マスク同士のIoUが高いものを間引く貪欲NMS(評価スクリプト側のみの後処理)。

    infer_masks()はscoreを返さない(内部で保持しているが最終的に捨てている)ため、
    面積が大きい方(=より完全な検出とみなす)を優先して残す。
    """
    n = len(masks)
    if n <= 1:
        return masks, 0
    areas = [int(m.sum()) for m in masks]
    order = sorted(range(n), key=lambda i: areas[i], reverse=True)
    kept: list[int] = []
    for i in order:
        mi = masks[i] > 0
        is_dup = False
        for j in kept:
            mj = masks[j] > 0
            inter = int(np.logical_and(mi, mj).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(mi, mj).sum())
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_thr:
                is_dup = True
                break
        if not is_dup:
            kept.append(i)
    kept_sorted = sorted(kept)
    return [masks[i] for i in kept_sorted], n - len(kept_sorted)


def main() -> None:
    from compare_quad_fit import segmentation_to_mask, iou  # noqa: E402
    from mask_rectify import min_area_rect_box  # noqa: E402
    from match_eval import normalize, digits_only, key_score  # noqa: E402
    from scipy.optimize import linear_sum_assignment  # noqa: E402
    from detection.pro_handbook.sam_py_demo.infer_for_storage import (  # noqa: E402
        SamConfig,
        SamBatchInfer_storage,
        mask_to_bbox2pt,
    )
    from PIL import Image  # noqa: E402

    print("=" * 70)
    print("SAM recognition eval on catheter-100 shelf images")
    print("=" * 70)

    # ---- 入力読み込み ----
    with open(ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)
    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        master = json.load(f)

    # 正解マスクベースの識別結果（ann_id -> book_name(ref)）。既存出力を読み取るのみ。
    gt_ref_by_ann: dict[int, str] = {}
    if REF_ASSIGNMENT_CSV.exists():
        with open(REF_ASSIGNMENT_CSV, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                hm = (row.get("hungarian_mask") or "").strip()
                if hm:
                    gt_ref_by_ann[int(hm)] = row["ref"]
        print(f"reference identification loaded: {len(gt_ref_by_ann)} / 85 anns from {REF_ASSIGNMENT_CSV}")
    else:
        print(f"⚠ reference assignment.csv が見つかりません: {REF_ASSIGNMENT_CSV} (識別精度は算出しません)")

    # ---- SAMモデルロード ----
    cfg = SamConfig(
        encoder_path=str(SAM_ENCODER),
        decoder_path=str(SAM_DECODER),
        device="gpu",
        pts_side=SAM_PTS_SIDE,
        min_area=SAM_MIN_AREA,
    )
    t0 = time.time()
    runner = SamBatchInfer_storage(cfg)
    print(f"[TIMER] model load: {time.time()-t0:.2f}s")

    per_instance_rows: list[dict] = []
    pred_extra_rows: list[dict] = []  # GTと対応しない予測(誤検出)の記録用

    for image_id in sorted(images_by_id.keys()):
        if image_id in EXCLUDED_IMAGE_IDS:
            print(f"\n--- image_id={image_id} : ユーザー指示により除外 ---")
            continue
        info = images_by_id[image_id]
        img_path = IMAGES_DIR / info["file_name"]
        height, width = info["height"], info["width"]
        anns = anns_by_image.get(image_id, [])

        print(f"\n--- image_id={image_id} ({info['file_name']}) : GT {len(anns)}件 ---")

        # ---- SAM推論（キャッシュがあれば再利用） ----
        cache_path = MASK_CACHE_DIR / f"{image_id}.npz"
        if cache_path.exists():
            npz = np.load(cache_path)
            pred_masks = [npz[k] for k in sorted(npz.files, key=lambda s: int(s.split("_")[1]))]
            print(f"  予測マスク: キャッシュから読込 {len(pred_masks)}件")
        else:
            pil_img = Image.open(img_path).convert("RGB")
            t0 = time.time()
            pred_masks = runner.infer_masks(
                pil_img,
                min_fold_deg=70.0,
                min_aspect=1.3,
                axis_angle_tol_deg=10.0,
                offset_width_factor=0.25,
                gap_len_factor=0.50,
                z_lower=-4.0,
                min_len_px=0.0,
                stage_save=None,
                stem_for_save=str(image_id),
                return_sam_data=False,
            )
            print(f"  [TIMER] infer: {time.time()-t0:.2f}s -> 予測マスク {len(pred_masks)}件")
            savez = {f"mask_{i}": np.asarray(m, dtype=np.uint8) for i, m in enumerate(pred_masks)}
            np.savez_compressed(cache_path, **savez)

        pred_masks = [np.asarray(m, dtype=np.uint8) for m in pred_masks]
        pred_masks, n_deduped = dedup_masks_by_iou(pred_masks)
        if n_deduped:
            print(f"  重複マスク間引き(評価側NMS, IoU>={DEDUP_IOU_THRESHOLD}): {n_deduped}件除去")
        n_pred = len(pred_masks)
        n_gt = len(anns)

        # ---- GTマスクをラスタライズ ----
        gt_masks = [segmentation_to_mask(a["segmentation"], height, width) for a in anns]

        # ---- IoU行列 (n_pred x n_gt) ----
        iou_mat = np.zeros((n_pred, n_gt), dtype=np.float64)
        for i in range(n_pred):
            for j in range(n_gt):
                iou_mat[i, j] = iou(pred_masks[i], gt_masks[j])

        # ---- ハンガリー法でマッチング(コスト=-IoU) ----
        matched_pred_of_gt: dict[int, int] = {}
        if n_pred > 0 and n_gt > 0:
            rows_i, cols_j = linear_sum_assignment(-iou_mat)
            for i, j in zip(rows_i, cols_j):
                if iou_mat[i, j] > 0.0:
                    matched_pred_of_gt[j] = i

        # ---- 識別: OCRをロードし予測マスクへ帰属、マスタと照合 ----
        pred_identified: dict[int, str] = {}
        ocr_path = OCR_DIR / f"{image_id}.json"
        if ocr_path.exists() and n_pred > 0:
            with open(ocr_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

            pred_boxes = []
            for m in pred_masks:
                bb = mask_to_bbox2pt(m)
                pred_boxes.append(bb)  # (x0,y0,x1,y1) or None

            MIN_REC_SCORE = 0.5
            buckets: list[list[tuple[float, str]]] = [[] for _ in range(n_pred)]
            for item in ocr_data.get("items", []):
                text = (item.get("text") or "").strip()
                if not text or float(item.get("rec_score", 1.0)) < MIN_REC_SCORE:
                    continue
                poly = np.asarray(item["poly_original"], dtype=np.float32)
                xs, ys = poly[:, 0], poly[:, 1]
                mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
                area = (mx2 - mx1) * (my2 - my1)
                if area <= 0:
                    continue
                best_i, best_ratio = None, 0.0
                for i, bb in enumerate(pred_boxes):
                    if bb is None:
                        continue
                    bx1, by1, bx2, by2 = bb
                    ox = max(0.0, min(mx2, bx2) - max(mx1, bx1))
                    oy = max(0.0, min(my2, by2) - max(my1, by1))
                    ratio = (ox * oy) / area
                    if ratio > best_ratio:
                        best_ratio, best_i = ratio, i
                if best_i is not None and best_ratio > 0:
                    buckets[best_i].append((float((my1 + my2) / 2.0), text))

            combined = [" ".join(t for _, t in sorted(b)) for b in buckets]

            n_master = len(master)
            S_ref = np.zeros((n_pred, n_master))
            S_dn = np.zeros((n_pred, n_master))
            S_date = np.zeros((n_pred, n_master))
            for i in range(n_pred):
                c = combined[i]
                for j, mst in enumerate(master):
                    S_ref[i, j] = key_score(mst.get("book_name", ""), c)
                    S_dn[i, j] = key_score(mst.get("display_name", ""), c)
                    S_date[i, j] = key_score(mst.get("expiration date", ""), c, is_date=True)
            S_max = np.maximum(np.maximum(S_ref, S_dn), S_date)

            if n_pred > 0 and n_master > 0:
                rows_i2, cols_j2 = linear_sum_assignment(-S_max)
                for i, j in zip(rows_i2, cols_j2):
                    pred_identified[i] = master[j].get("book_name", "")
        elif not ocr_path.exists():
            print(f"  ⚠ OCR結果が見つかりません: {ocr_path} (識別精度は算出しません)")

        # ---- per-instance 行を作成 ----
        matched_gt_count = 0
        for j, ann in enumerate(anns):
            ann_id = ann["id"]
            gt_mask = gt_masks[j]
            gt_box = min_area_rect_box(gt_mask)
            gt_width_px = None
            if gt_box is not None:
                w1 = float(np.linalg.norm(gt_box[0] - gt_box[1]))
                w2 = float(np.linalg.norm(gt_box[1] - gt_box[2]))
                gt_width_px = min(w1, w2)

            i = matched_pred_of_gt.get(j)
            pred_iou = float(iou_mat[i, j]) if i is not None else 0.0
            matched = pred_iou >= IOU_MATCH_THRESHOLD

            pred_width_px = None
            width_error_px = None
            width_error_pct = None
            pred_book_name = ""
            identified_match = ""

            if matched:
                matched_gt_count += 1
                pred_box = min_area_rect_box(pred_masks[i])
                if pred_box is not None:
                    p1 = float(np.linalg.norm(pred_box[0] - pred_box[1]))
                    p2 = float(np.linalg.norm(pred_box[1] - pred_box[2]))
                    pred_width_px = min(p1, p2)
                if pred_width_px is not None and gt_width_px is not None:
                    width_error_px = pred_width_px - gt_width_px
                    if gt_width_px > 0:
                        width_error_pct = 100.0 * width_error_px / gt_width_px

                pred_book_name = pred_identified.get(i, "")
                gt_ref = gt_ref_by_ann.get(ann_id, "")
                if gt_ref_by_ann:
                    if gt_ref and pred_book_name:
                        identified_match = str(pred_book_name == gt_ref)
                    else:
                        identified_match = "N/A"
            else:
                gt_ref = gt_ref_by_ann.get(ann_id, "")

            per_instance_rows.append({
                "image_id": image_id,
                "gt_ann_id": ann_id,
                "matched": matched,
                "iou": round(pred_iou, 4),
                "gt_width_px": round(gt_width_px, 2) if gt_width_px is not None else "",
                "pred_width_px": round(pred_width_px, 2) if pred_width_px is not None else "",
                "width_error_px": round(width_error_px, 2) if width_error_px is not None else "",
                "width_error_pct": round(width_error_pct, 2) if width_error_pct is not None else "",
                "gt_book_name_ref": gt_ref,
                "pred_book_name_identified": pred_book_name,
                "identified_match": identified_match,
            })

        n_extra = n_pred - len(set(matched_pred_of_gt.values()))
        print(f"  予測{n_pred}件 / GT{n_gt}件  ->  IoU>=0.5マッチ {matched_gt_count}/{n_gt}  誤検出(未対応の予測) {n_extra}")

        # ---- QC用オーバーレイ画像 ----
        bgr = cv2.imread(str(img_path))
        if bgr is not None:
            vis = bgr.copy()
            for j, ann in enumerate(anns):
                m = gt_masks[j] > 0
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
            for i in range(n_pred):
                m = pred_masks[i] > 0
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
            cv2.imwrite(str(OVERLAY_DIR / f"{image_id}.png"), vis)

    # ---- per_instance.csv 出力 ----
    csv_path = OUT_DIR / "per_instance.csv"
    fieldnames = list(per_instance_rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_instance_rows)
    print(f"\n✔ per_instance.csv -> {csv_path}")

    # ---- 集計 ----
    n_total = len(per_instance_rows)
    matched_rows = [r for r in per_instance_rows if r["matched"]]
    n_matched = len(matched_rows)
    ious_matched = [r["iou"] for r in matched_rows]
    ious_all = [r["iou"] for r in per_instance_rows]

    width_errs_px = [r["width_error_px"] for r in matched_rows if r["width_error_px"] != ""]
    width_errs_pct = [r["width_error_pct"] for r in matched_rows if r["width_error_pct"] != ""]

    ident_rows = [r for r in matched_rows if r["identified_match"] in ("True", "False")]
    n_ident_match = sum(1 for r in ident_rows if r["identified_match"] == "True")

    used_image_ids = sorted(iid for iid in images_by_id.keys() if iid not in EXCLUDED_IMAGE_IDS)
    summary_lines = []
    summary_lines.append(
        f"# SAM認識評価サマリ (catheter-100, 画像{used_image_ids}・GT{n_total}件)"
    )
    if EXCLUDED_IMAGE_IDS:
        summary_lines.append(
            f"※画像{sorted(EXCLUDED_IMAGE_IDS)}はユーザー指示により評価対象から除外"
        )
    summary_lines.append("")
    summary_lines.append(f"SAMパラメータ: pts_side={SAM_PTS_SIDE}, min_area={SAM_MIN_AREA}, "
                          f"encoder/decoder=sam_vit_h_4b8939 (ONNX, GPU)")
    summary_lines.append("")
    summary_lines.append("## 1. IoU精度")
    summary_lines.append(f"- GT総数: {n_total}")
    summary_lines.append(f"- IoU>=0.5でマッチした件数: {n_matched} / {n_total} "
                          f"({100.0*n_matched/n_total:.1f}%)")
    summary_lines.append(f"- 未検出(マッチなし): {n_total - n_matched} / {n_total}")
    if ious_matched:
        arr = np.array(ious_matched)
        summary_lines.append(f"- マッチしたペアのIoU: 平均={arr.mean():.4f}  中央値={np.median(arr):.4f}  "
                              f"最小={arr.min():.4f}  最大={arr.max():.4f}")
    arr_all = np.array(ious_all)
    summary_lines.append(f"- 全GTに対するIoU(未検出は0扱い): 平均={arr_all.mean():.4f}  中央値={np.median(arr_all):.4f}")
    summary_lines.append("")

    summary_lines.append("## 2. 書籍幅(px)の精度  ※マッチしたペアのみ、min_area_rect短辺")
    if width_errs_px:
        e = np.array(width_errs_px, dtype=float)
        ep = np.array(width_errs_pct, dtype=float)
        summary_lines.append(f"- サンプル数: {len(e)}")
        summary_lines.append(f"- MAE: {np.mean(np.abs(e)):.2f} px")
        summary_lines.append(f"- RMSE: {np.sqrt(np.mean(e**2)):.2f} px")
        summary_lines.append(f"- 平均誤差(符号あり, バイアス): {np.mean(e):+.2f} px")
        summary_lines.append(f"- 平均絶対%誤差: {np.mean(np.abs(ep)):.2f} %")
    else:
        summary_lines.append("- 算出不可（マッチしたペアで矩形化が成功した件がありませんでした）")
    summary_lines.append("")

    summary_lines.append("## 3. 識別精度 (OCR + マスタ照合)")
    if ident_rows:
        summary_lines.append(f"- 比較可能件数(IoUマッチ済み かつ 予測/参照とも識別結果あり): {len(ident_rows)}")
        summary_lines.append(f"- 参照識別結果(正解マスクベース, match_multi/assignment.csv)と一致: "
                              f"{n_ident_match} / {len(ident_rows)} ({100.0*n_ident_match/len(ident_rows):.1f}%)")
    else:
        summary_lines.append("- 比較可能件数が0件でした（OCR結果 or 参照識別結果が不足）")
    summary_lines.append("")

    summary_lines.append("## 備考")
    summary_lines.append("- depth無しのため書籍幅はmm実測できず、px比較で代替。")
    summary_lines.append("- catheter/outputs/ocr_multi/ が存在しなかったため、識別精度の計算には既存の "
                          "catheter/outputs/ocr/ (単一角度OCR) を用いた。")
    summary_lines.append("- SAM3(本番SAM ONNXモデル)による実推論を初めて本データセットに適用した結果である。")

    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"✔ summary.md -> {summary_path}")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
