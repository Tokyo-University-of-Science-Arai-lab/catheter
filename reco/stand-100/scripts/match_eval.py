#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
照合方式の比較評価。

検証したいこと:
  1. REF(book_name) は何箱で読めていて、何箱で読めていないのか
  2. display_name / 有効期限を第2・第3のキーにすると何箱救えるのか
  3. 独立照合（現行）と 全体最適割当（ハンガリー法）で結果がどう変わるか

入力（読み取り専用）:
    catheter/outputs/ocr/<image_id>.json           （run_ocr_catheter.py の出力）
    catheter-100/annotations/instances_default.json
    catheter-100/master_catheter_20260216.json

出力（新規作成のみ）:
    catheter/outputs/match/key_detectability.csv   キーごとの読み取り可否
    catheter/outputs/match/assignment.csv          割当結果（独立 vs ハンガリー）
    catheter/outputs/match/overlay_<image_id>.png  割当結果の可視化

実行:
    .pro_hand_book_fixed/bin/python catheter/scripts/match_eval.py
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "catheter-100"
ANNOTATIONS_JSON = SRC / "annotations" / "instances_default.json"
MASTER_JSON = SRC / "master_catheter_20260216.json"
IMAGES_DIR = SRC / "images" / "default"

OUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"

# 既定は多角度OCR（run_ocr_multiangle.py）の結果を使う。
# 単一角度版と比較したいときは --ocr-dir ocr / --out-dir match_single を指定する。
OCR_DIR = OUT_ROOT / "ocr_multi"
OUT_DIR = OUT_ROOT / "match_multi"

# 本番 only_one.py と同じ閾値（score > THRESHOLD で採用）
THRESHOLD = 40.0


def normalize(s: str) -> str:
    """全角半角・記号ゆれを吸収して比較しやすくする。"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("®", "").replace("™", "").replace("©", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip().upper()


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def load_master() -> list[dict]:
    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# 低信頼の検出はゴミ文字（'n' '福' 等）になりやすいので落とす
MIN_REC_SCORE = 0.5


def assign_ocr_to_masks(ocr_items: list[dict], anns: list[dict]) -> dict[int, list[str]]:
    """
    本番 extract_book_texts と同じ方針:
    OCR文字boxの面積のうち、どのマスクbboxに最も多く重なるかで帰属を決める。

    多角度OCRの結果をそのまま連結すると精度が落ちる問題への対策を2つ入れている:
      1. 並び順の復元
         多角度版の重複排除は rec_score 降順で並べるため、文字の空間的な順序が壊れる。
         fuzz.partial_ratio は連続した部分列を見るので、'pNOVUS17' と '150' が
         離れて並ぶだけでスコアが落ちる（実測 80.0 -> 66.7）。
         そこで各マスク内を読み順に並べ直す。
         base座標では base_x = orig_y なので、orig_y 昇順が左→右の読み順にあたる。
      2. 低信頼検出の除去
         傾き画像から出る rec_score の低い断片はほぼゴミなので落とす。
    """
    buckets: dict[int, list[tuple[float, str]]] = {a["id"]: [] for a in anns}

    for item in ocr_items:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if float(item.get("rec_score", 1.0)) < MIN_REC_SCORE:
            continue

        # poly_orig: 多角度版 / poly_original: 単一角度版
        poly_key = "poly_orig" if "poly_orig" in item else "poly_original"
        poly = np.asarray(item[poly_key], dtype=np.float32)
        xs, ys = poly[:, 0], poly[:, 1]
        mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
        area = (mx2 - mx1) * (my2 - my1)
        if area <= 0:
            continue

        best_id, best_ratio = None, 0.0
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            ox = max(0.0, min(mx2, bx + bw) - max(mx1, bx))
            oy = max(0.0, min(my2, by + bh) - max(my1, by))
            ratio = (ox * oy) / area
            if ratio > best_ratio:
                best_ratio, best_id = ratio, a["id"]

        if best_id is not None and best_ratio > 0:
            # orig_y の中心を読み順のキーにする（base座標の左→右に対応）
            buckets[best_id].append((float((my1 + my2) / 2.0), text))

    return {mid: [t for _, t in sorted(v)] for mid, v in buckets.items()}


def key_score(key: str, combined: str, *, is_date: bool = False) -> float:
    """
    1つのキー文字列に対するスコア。

    日付について:
      当初 digits_only + partial_ratio で比較したところ全100件が閾値を超え、
      明らかな偽陽性だった。OCR結果には型番・寸法・LOTなど数字が大量にあり、
      数字だけを連結した文字列に対する部分一致は、ほぼ何にでも当たってしまう。
      そこで、combined 側は「日付らしいトークン」だけを候補にし、
      その中で最も近いものとのスコアを返すようにする。
    """
    if not key:
        return 0.0

    if is_date:
        k = digits_only(key)
        if len(k) < 6:  # '2029-02' のような不完全な日付はキーとして使わない
            return 0.0
        cands = [digits_only(t) for t in re.findall(r"\d[\d\-/.]{5,}", combined)]
        cands = [c for c in cands if len(c) >= 6]
        if not cands:
            return 0.0
        return float(max(fuzz.ratio(k, c) for c in cands))

    return float(fuzz.partial_ratio(normalize(key), normalize(combined)))


def best_and_margin(col: np.ndarray) -> tuple[float, float]:
    """
    最高スコアと、2位との差（margin）を返す。
    margin が小さいほど「どの箱とも同じくらい似ている」＝信用できない。
    """
    if col.size == 0:
        return 0.0, 0.0
    s = np.sort(col)[::-1]
    top = float(s[0])
    second = float(s[1]) if s.size > 1 else 0.0
    return top, top - second


def main() -> None:
    global OCR_DIR, OUT_DIR
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ocr-dir", default=None, help="outputs/ 配下のOCR結果ディレクトリ名")
    ap.add_argument("--out-dir", default=None, help="outputs/ 配下の出力先ディレクトリ名")
    args = ap.parse_args()
    if args.ocr_dir:
        OCR_DIR = OUT_ROOT / args.ocr_dir
    if args.out_dir:
        OUT_DIR = OUT_ROOT / args.out_dir
    print(f"OCR: {OCR_DIR}\n出力: {OUT_DIR}\n")

    with open(ANNOTATIONS_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    master = load_master()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    detect_rows = []
    assign_rows = []

    for image_id in sorted(images):
        ocr_path = OCR_DIR / f"{image_id}.json"
        if not ocr_path.exists():
            print(f"⚠ OCR結果がありません: {ocr_path}")
            continue

        with open(ocr_path, "r", encoding="utf-8") as f:
            ocr = json.load(f)

        anns = anns_by_image.get(image_id, [])
        mask_texts = assign_ocr_to_masks(ocr["items"], anns)
        mask_ids = [a["id"] for a in anns]
        combined = {mid: " ".join(mask_texts[mid]) for mid in mask_ids}

        n_mask, n_master = len(mask_ids), len(master)

        # ---- スコア行列（キー別） ----
        S_ref = np.zeros((n_mask, n_master))
        S_dn = np.zeros((n_mask, n_master))
        S_date = np.zeros((n_mask, n_master))

        for i, mid in enumerate(mask_ids):
            c = combined[mid]
            for j, m in enumerate(master):
                S_ref[i, j] = key_score(m.get("book_name", ""), c)
                S_dn[i, j] = key_score(m.get("display_name", ""), c)
                S_date[i, j] = key_score(m.get("expiration date", ""), c, is_date=True)

        S_max = np.maximum(np.maximum(S_ref, S_dn), S_date)

        # ---- 1) キーごとの読み取り可否 ----
        for j, m in enumerate(master):
            ref_top, ref_mg = best_and_margin(S_ref[:, j])
            dn_top, dn_mg = best_and_margin(S_dn[:, j])
            date_top, date_mg = best_and_margin(S_date[:, j])
            detect_rows.append(
                {
                    "image_id": image_id,
                    "ref": m.get("book_name", ""),
                    "display_name": m.get("display_name", ""),
                    "expiration": m.get("expiration date", ""),
                    "best_ref_score": round(ref_top, 1),
                    "ref_margin": round(ref_mg, 1),
                    "best_dn_score": round(dn_top, 1),
                    "dn_margin": round(dn_mg, 1),
                    "best_date_score": round(date_top, 1),
                    "date_margin": round(date_mg, 1),
                    "ref_ok": bool(S_ref[:, j].max() > THRESHOLD),
                    "dn_ok": bool(S_dn[:, j].max() > THRESHOLD),
                    "date_ok": bool(S_date[:, j].max() > THRESHOLD),
                    "ref_pick_mask": mask_ids[int(S_ref[:, j].argmax())],
                    "dn_pick_mask": mask_ids[int(S_dn[:, j].argmax())],
                    "maxkey_pick_mask": mask_ids[int(S_max[:, j].argmax())],
                }
            )

        # ---- 2) 独立照合（現行）: 各queryが独自にargmaxを取る ----
        indep_pick = {j: int(S_max[:, j].argmax()) for j in range(n_master)}
        # 衝突 = 同じマスクを複数のqueryが取り合っている
        from collections import Counter

        collisions = sum(v - 1 for v in Counter(indep_pick.values()).values() if v > 1)

        # ---- 3) 全体最適割当（ハンガリー法） ----
        rows_i, cols_j = linear_sum_assignment(-S_max)
        hung = {int(j): int(i) for i, j in zip(rows_i, cols_j)}

        agree = sum(1 for j in range(n_master) if hung.get(j) == indep_pick.get(j))

        for j, m in enumerate(master):
            i_ind = indep_pick[j]
            i_hun = hung.get(j)
            assign_rows.append(
                {
                    "image_id": image_id,
                    "ref": m.get("book_name", ""),
                    "display_name": m.get("display_name", ""),
                    "independent_mask": mask_ids[i_ind],
                    "independent_score": round(float(S_max[i_ind, j]), 1),
                    "hungarian_mask": mask_ids[i_hun] if i_hun is not None else "",
                    "hungarian_score": round(float(S_max[i_hun, j]), 1) if i_hun is not None else "",
                    "same": bool(i_hun == i_ind),
                }
            )

        print(
            f"image_id={image_id}: マスク{n_mask} / マスタ{n_master}  "
            f"独立照合の衝突={collisions}  ハンガリーと一致={agree}/{n_master}"
        )

        # ---- 4) 可視化 ----
        img = cv2.imread(str(IMAGES_DIR / images[image_id]["file_name"]))
        if img is not None:
            vis = img.copy()
            for j, m in enumerate(master):
                i = hung.get(j)
                if i is None:
                    continue
                a = anns[i]
                bx, by, bw, bh = [int(v) for v in a["bbox"]]
                sc = S_max[i, j]
                color = (0, 200, 0) if sc > THRESHOLD else (0, 0, 255)
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 2)
                label = f"{m.get('display_name','')[:14]} {sc:.0f}"
                cv2.putText(vis, label, (bx, max(12, by - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(vis, label, (bx, max(12, by - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            cv2.imwrite(str(OUT_DIR / f"overlay_{image_id}.png"), vis)

    # ---- CSV 出力 ----
    if detect_rows:
        with open(OUT_DIR / "key_detectability.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(detect_rows[0].keys()))
            w.writeheader()
            w.writerows(detect_rows)
    if assign_rows:
        with open(OUT_DIR / "assignment.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(assign_rows[0].keys()))
            w.writeheader()
            w.writerows(assign_rows)

    # ---- サマリ ----
    print("\n===== キー別の読み取り可否（全画像・全マスタ件数を母数） =====")
    n = len(detect_rows)
    if n:
        ref_ok = sum(r["ref_ok"] for r in detect_rows)
        dn_ok = sum(r["dn_ok"] for r in detect_rows)
        date_ok = sum(r["date_ok"] for r in detect_rows)
        any_ok = sum(r["ref_ok"] or r["dn_ok"] or r["date_ok"] for r in detect_rows)
        saved = sum((not r["ref_ok"]) and (r["dn_ok"] or r["date_ok"]) for r in detect_rows)
        print(f"母数                       : {n}")
        print(f"REF だけで合格             : {ref_ok} ({ref_ok/n*100:.1f}%)")
        print(f"display_name だけで合格    : {dn_ok} ({dn_ok/n*100:.1f}%)")
        print(f"有効期限だけで合格         : {date_ok} ({date_ok/n*100:.1f}%)")
        print(f"いずれかで合格             : {any_ok} ({any_ok/n*100:.1f}%)")
        print(f"→ REF失敗を第2/第3キーが救済: {saved}")

    print(f"\nCSV: {OUT_DIR}")


if __name__ == "__main__":
    main()
