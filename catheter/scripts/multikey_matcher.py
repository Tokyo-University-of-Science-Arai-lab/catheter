#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多段query + 全体最適割当（ハンガリー法）による対象マスク選択。

本番 OCR/only_one_tilted.match_text_to_mask_main のドロップイン置換。
既存ファイルは一切変更せず、呼び出し側の import を差し替えるだけで使える。

    # 変更前
    from .OCR.only_one_tilted import match_text_to_mask_main
    # 変更後
    from catheter.scripts.multikey_matcher import match_text_to_mask_main

■ 何を変えるのか
  現行: query（REF）1本で、各マスクを独立にスコアリングして argmax を取る。
        → REFの印字が小さすぎて読めない品目（例 MC1715000）は選定に失敗する。
        → 似た品目（Excelsior XT-17 / XT-27 / SL-10）で取り違えが起きる。

  本実装:
    1. 多段query
       マスタから query(REF) に対応する display_name と有効期限を引き、
       REF / display_name / 期限 の3キーで採点して最大値を採用する。
       実測: 確信ありが 56/80 → 78/80 に改善（catheter-100・4画像・20品目）。
    2. 全体最適割当
       「全マスク × マスタ全品目」のスコア行列を作りハンガリー法で1対1に割り当てる。
       「他の品目がより強く欲しがっているマスク」は取られるため、
       似た品目どうしの取り違えが構造的に減る。
       棚とマスタが1対1に閉じている場合は、読めない品目も消去法で確定できる。

■ 返り値
  本番と同一形式（score降順）:
      [{"name": "mask_3", "score": 87, "box": {"x1":..,"y1":..,"x2":..,"y2":..},
        "forced_angle": 90}, ...]
  name の末尾数字が1始まりのマスク番号。呼び出し側 merge_ocr_and_masks は
  re.search(r"(\\d+)$", name) でこれを取り出すため、命名規則を厳守している。

■ 保存されるデバッグ出力（shot_dir配下）
  multikey_match_debug.json : キー別スコア・割当・採用理由
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rapidfuzz import fuzz

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # scipy が無い環境では貪欲法にフォールバック
    _HAS_SCIPY = False


# ===== 設定 =====

# マスタJSON。呼び出し時に master_json= で上書きできる。
DEFAULT_MASTER_JSON = (
    Path(__file__).resolve().parents[2] / "catheter-100" / "master_catheter_20260216.json"
)

# 低信頼のOCR断片はゴミ文字になりやすいので落とす
MIN_REC_SCORE = 0.5

# 「確信あり」の判定。margin = 最高スコアと2位の差。
# margin が小さいものは「どのマスクとも同程度に似ている」＝信用できない。
CONFIDENT_SCORE = 70.0
CONFIDENT_MARGIN = 20.0

# combined(マスクに帰属したOCRテキスト)がこれより短い場合はスコアを信用しない。
# fuzz.partial_ratio は短い文字列ほど「たまたま部分一致」しやすく、2文字の
# OCR断片("00"等)が長いREFコード("M00345100950"等)に含まれるだけで満点(100)に
# なってしまう実例が確認された(2026-08-21)。過去に軸検出側で見つかった同種の
# バグ(HANDOFF_20260731.md、1文字断片が満点になっていた件)と同根の問題。
MIN_KEY_TEXT_LEN_FOR_MATCH = 4

FORCED_ANGLE = 90

# 色補助キー(2026-08-21試験導入)。無地・OCR手がかりの薄い品目(オレンジ箱等)向けの
# 補助シグナル。マスタ側にcolor_rgbが無い品目は0点になり、他の3キー(ref/display_name/
# 日付)だけで採点した場合と同じ挙動になる(後方互換、マスタ未対応でも壊れない)。
COLOR_MAX_DIST = 150.0

# 色キーの中央値センタリング後の値に掛ける減衰係数。多くの品目は白〜グレー系の
# 似た色なので、素の色スコアだけでも僅かな差でwinning_key/marginの計算を乗っ取り、
# 本来ref/display_nameで確信度が高いはずのマスクの信頼度表示を壊す実例が確認された
# (2026-08-21、MC1715000等で確認: 同じマスクが選ばれ続けるのにcolorキーが僅差で
# argmaxを奪いmargin=-3.1等になり見かけ上confident=Falseになっていた)。
# 減衰させることで、色がオレンジ箱のように明確に違う場合だけ実際に勝てるようにする。
COLOR_KEY_WEIGHT = 0.35


# ===== 文字列正規化 =====

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    for ch in ("®", "™", "©"):
        s = s.replace(ch, "")
    return re.sub(r"\s+", " ", s).strip().upper()


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


# 選択マスクのOCRテキストが「識別の根拠として意味を持ちそうか」の簡易判定。
# 2026-08-21、他セッションとの共同分析(88件の目視レビュー)で、選択マスクの
# OCRテキスト長がT(正解)/F(誤り)を分ける最も強いシグナルだと判明した
# (T群 text_len 中央値12、F群中央値4〜7.5)。ただし文字数だけでは
# sensitivity/specificityが頭打ちになる(F群にも誤読で長くなったガーベジ文字列が
# 混じる)ため、長さに加えて「REFコードや日付らしい形式か」も見る2段構成にする。
MIN_TEXT_LEN_FOR_PLAUSIBLE = 7


def _looks_like_plausible_identifier(text: str) -> bool:
    """selected_maskのcombinedテキストが、REF/日付として意味を持ちそうかを判定する。

    根拠にならない短い断片・でたらめなOCR誤読を、スコアが高くても弾くための
    2段目のガード(1段目は_key_scoreの最小文字数ガード)。
    """
    t = (text or "").strip()
    if len(_normalize(t)) < MIN_TEXT_LEN_FOR_PLAUSIBLE:
        return False
    digits = _digits_only(t)
    has_long_digit_run = len(digits) >= 4
    has_alnum_mix = bool(re.search(r"[A-Za-z]", t)) and bool(re.search(r"\d", t))
    looks_like_date = bool(re.search(r"\d[\d\-/.]{5,}", t))
    return has_long_digit_run or has_alnum_mix or looks_like_date


def _key_score(key: str, combined: str, *, is_date: bool = False) -> float:
    """
    1キーに対するスコア（0〜100）。

    日付は素朴に数字だけ比較すると、型番・寸法・LOTの数字に当たって
    ほぼ何にでも一致してしまう（検証時に全件が閾値を超えた）。
    そのため combined 側を「日付らしいトークン」に絞ってから比較する。
    """
    if not key:
        return 0.0

    if is_date:
        k = _digits_only(key)
        if len(k) < 6:  # '2029-02' のような不完全な日付はキーにしない
            return 0.0
        cands = [_digits_only(t) for t in re.findall(r"\d[\d\-/.]{5,}", combined)]
        cands = [c for c in cands if len(c) >= 6]
        if not cands:
            return 0.0
        return float(max(fuzz.ratio(k, c) for c in cands))

    key_norm = _normalize(key)
    combined_norm = _normalize(combined)
    # partial_ratioは短い方の文字列が長い方のどこかに偶然含まれるだけで満点になり
    # うるため、key/combinedどちらが短くても最低文字数を満たさなければ信用しない。
    if min(len(key_norm), len(combined_norm)) < MIN_KEY_TEXT_LEN_FOR_MATCH:
        return 0.0
    return float(fuzz.partial_ratio(key_norm, combined_norm))


def _color_score(mask_rgb, master_rgb, *, max_dist: float = COLOR_MAX_DIST) -> float:
    """マスクの平均色とマスタの参照色(color_rgb)のユークリッド距離を0〜100に変換する。

    他の3キー(ref/display_name/date)と同じ0〜100スケールに合わせ、後段の
    列ごと中央値センタリングと同じ仕組みに乗せる。ほとんどの品目は白系パッケージで
    互いに似た色になるため中央値センタリング後はほぼ0になり、オレンジ箱のように
    明確に色が違う品目だけが浮き上がる設計(色が万能の識別キーになるわけではない)。
    """
    if mask_rgb is None or not master_rgb:
        return 0.0
    a = np.asarray(mask_rgb, dtype=np.float64)
    b = np.asarray(master_rgb, dtype=np.float64)
    if a.shape != (3,) or b.shape != (3,):
        return 0.0
    dist = float(np.linalg.norm(a - b))
    return max(0.0, 100.0 * (1.0 - dist / max_dist))


# ===== マスク・OCRの前処理 =====

def _mask_to_binary(mask, h: int, w: int) -> np.ndarray:
    """本番 only_one_tilted._mask_to_binary と同じ意図の正規化。"""
    b = (np.asarray(mask) > 0).astype(np.uint8)
    if b.ndim > 2:
        b = np.squeeze(b)
    if b.ndim != 2:
        raise ValueError(f"mask の次元が不正です: {b.shape}")
    if b.shape != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_NEAREST)
    return b


def _mask_bbox(mask_bin: np.ndarray) -> dict[str, float] | None:
    ys, xs = np.where(mask_bin > 0)
    if xs.size == 0:
        return None
    return {
        "x1": float(xs.min()), "y1": float(ys.min()),
        "x2": float(xs.max()), "y2": float(ys.max()),
    }


def _mask_mean_rgb(mask_bin: np.ndarray, rgb_img: np.ndarray) -> list[float] | None:
    """マスク内画素の平均色(R,G,B)。rgb_imgはcv2.imread直後のBGR画像を想定。"""
    if mask_bin.sum() < 1:
        return None
    mean_bgr = rgb_img[mask_bin > 0].mean(axis=0)
    return [round(float(mean_bgr[2]), 1), round(float(mean_bgr[1]), 1), round(float(mean_bgr[0]), 1)]


def _poly_mask_overlap_ratio(poly_pts: np.ndarray, mask_bin: np.ndarray, h: int, w: int) -> float:
    """OCR文字ポリゴンの面積のうち、実際のマスク輪郭(ピクセル単位)と重なる割合。

    2026-08-21: 従来は矩形バウンディングボックス同士の重なりで判定していたが、
    棚の箱は微妙に傾いて立っているため隣接マスクの矩形が重なり合い、隣の箱の
    文字が誤って割り当てられる実例が確認された(query=ESC0305で隣のAXS Vecta 46
    DACのテキストバケットに"ESC0305"が混入し、confident=Trueで誤選択)。矩形では
    なく実際のマスク輪郭との重なりで判定することで、傾いた隣接マスク同士が
    矩形上は重なっていても実体(輪郭)は重ならない場合に正しく区別できる。
    """
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts_int = np.round(poly_pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(canvas, [pts_int], 1)
    poly_area = int(canvas.sum())
    if poly_area <= 0:
        return 0.0
    inter = int(np.count_nonzero((canvas > 0) & (mask_bin > 0)))
    return inter / poly_area


def _unrotate_poly(poly, angle: int, w: int, h: int) -> np.ndarray:
    """本番 unrotate_poly_to_original と同一の変換。"""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    a = int(angle) % 360
    if a == 90:
        xo, yo = y, (h - 1) - x
    elif a == 180:
        xo, yo = (w - 1) - x, (h - 1) - y
    elif a == 270:
        xo, yo = (w - 1) - y, x
    else:
        xo, yo = x, y
    return np.stack([xo, yo], axis=1)


def _collect_mask_texts(
    ocr_json_path: Path,
    masks: list,
    rgb_path: Path,
    forced_angle: int = FORCED_ANGLE,
) -> tuple[list[str], list[dict | None], list[dict]]:
    """
    OCR文字を各マスクへ割り当て、マスクごとの結合テキストを作る。

    重要: 文字は「読み順」に並べて結合する。
    fuzz.partial_ratio は連続部分列を見るため、並びが崩れると
    'pNOVUS17' と '150' が離れただけでスコアが落ちる（実測 80.0 → 66.7）。
    """
    with open(ocr_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    polys = data.get("dt_polys", [])
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [1.0] * len(texts))

    img = cv2.imread(str(rgb_path))
    if img is None:
        raise FileNotFoundError(f"画像が読めませんでした: {rgb_path}")
    h, w = img.shape[:2]

    mask_bins = [_mask_to_binary(m, h, w) for m in masks]
    boxes = [_mask_bbox(mb) for mb in mask_bins]

    buckets: list[list[tuple[float, str]]] = [[] for _ in masks]
    debug: list[dict] = []

    for idx, (poly, text, sc) in enumerate(zip(polys, texts, scores), start=1):
        text = (text or "").strip()
        if not text or float(sc) < MIN_REC_SCORE:
            continue

        p = _unrotate_poly(poly, forced_angle, w=w, h=h)
        x1, y1 = float(p[:, 0].min()), float(p[:, 1].min())
        x2, y2 = float(p[:, 0].max()), float(p[:, 1].max())
        area = (x2 - x1) * (y2 - y1)
        if area <= 0:
            continue

        # 文字ポリゴンの面積のうち、どのマスクの実際の輪郭(ピクセル単位)に
        # 最も多く重なるかで帰属を決める(矩形バウンディングボックスではない。
        # 理由は_poly_mask_overlap_ratioのdocstring参照)。
        best_i, best_ratio = None, 0.0
        for i, mb in enumerate(mask_bins):
            ratio = _poly_mask_overlap_ratio(p, mb, h, w)
            if ratio > best_ratio:
                best_ratio, best_i = ratio, i

        if best_i is not None and best_ratio > 0:
            # 読み順キー。
            # 本番のOCR入力は rotate90CW(after_init_rgb) なので base_x = h-1-y。
            # つまり base 上の左→右は、本番画像座標では y の降順にあたる。
            # 昇順にすると 'pNOVUS17 ... 150' が '150 ... pNOVUS17' と反転し、
            # 連続部分列を見る partial_ratio のスコアが落ちる（実測 76.9 → 66.7）。
            buckets[best_i].append((-(y1 + y2) / 2.0, text))
            debug.append({
                "ocr_index": idx, "text": text,
                "matched": f"mask_{best_i + 1}", "overlap_ratio": round(best_ratio, 3),
            })
        else:
            debug.append({"ocr_index": idx, "text": text, "matched": None, "overlap_ratio": 0.0})

    combined = [" ".join(t for _, t in sorted(b)) for b in buckets]
    return combined, boxes, debug


# ===== マスタ =====

def _load_master(master_json: Path) -> list[dict]:
    with open(master_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"マスタはリスト形式である必要があります: {master_json}")
    return data


def _keys_of(entry: dict) -> list[tuple[str, str, bool]]:
    """(キー種別, 文字列, 日付か) の一覧。"""
    return [
        ("ref", entry.get("book_name", "") or "", False),
        ("display_name", entry.get("display_name", "") or "", False),
        ("date", entry.get("expiration date", "") or "", True),
    ]


# ===== 割当 =====

def _greedy_assign(S: np.ndarray) -> dict[int, int]:
    """scipy が無い場合のフォールバック（スコアの高い順に確定させる）。"""
    n_mask, n_master = S.shape
    order = np.argsort(-S, axis=None)
    used_m, used_j, out = set(), set(), {}
    for flat in order:
        i, j = divmod(int(flat), n_master)
        if i in used_m or j in used_j:
            continue
        out[j] = i
        used_m.add(i)
        used_j.add(j)
        if len(out) >= min(n_mask, n_master):
            break
    return out


# ===== 本番互換の入口 =====

def match_text_to_mask_main(
    query: str,
    masks,
    shot_dir,
    threshold: int = 40,
    *,
    master_json: str | Path | None = None,
    use_hungarian: bool = True,
    use_multikey: bool = True,
) -> list[dict[str, Any]]:
    """
    本番 only_one_tilted.match_text_to_mask_main と同じ入出力。

    query はマスタの book_name（REF）を想定する。
    マスタに query が無い場合は、多段queryが使えないため
    query 1本のみで採点する（現行と同じ挙動）。
    """
    shot_dir = Path(shot_dir)
    ocr_json_path = shot_dir / "ocr_result.json"
    rgb_path = shot_dir / "after_init_rgb.png"

    combined, boxes, ocr_debug = _collect_mask_texts(ocr_json_path, masks, rgb_path)
    n_mask = len(masks)

    # 色補助キー用: 各マスクの平均色を取っておく(2026-08-21試験導入)。
    rgb_img_for_color = cv2.imread(str(rgb_path))
    mask_mean_rgb: list[list[float] | None] = [None] * n_mask
    if rgb_img_for_color is not None:
        h_c, w_c = rgb_img_for_color.shape[:2]
        for i, m in enumerate(masks):
            mb = _mask_to_binary(m, h_c, w_c)
            mask_mean_rgb[i] = _mask_mean_rgb(mb, rgb_img_for_color)

    master_path = Path(master_json) if master_json else DEFAULT_MASTER_JSON
    if not use_multikey:
        # 現行方式の再現用: query(REF) 1本だけで採点する
        master_path = Path("(disabled)")
        master = []
    else:
        try:
            master = _load_master(master_path)
        except Exception as e:
            print(f"[multikey] マスタを読めませんでした（query単独にフォールバック）: {e}")
            master = []

    # query に対応するマスタ行を探す
    q_norm = _normalize(query)
    target_j = next(
        (j for j, m in enumerate(master) if _normalize(m.get("book_name", "")) == q_norm),
        None,
    )
    if target_j is None:
        master = [{"book_name": query, "display_name": "", "expiration date": ""}]
        target_j = 0
        print(f"[multikey] query '{query}' はマスタに無いため query 単独で採点します")

    n_master = len(master)

    # ===== スコア行列 =====
    # 素朴に max(REF, display_name, 期限) を取ると、識別力の無いキーが勝ってしまう。
    # 実例: query=MC1715000（期限2028-11-27）に対し、別の箱の '2028-11-15' が
    #       fuzz.ratio=75 を出し、正解の箱の display_name=66.7 を上回った。
    # そこで各キーの列から中央値を引いた「その品目らしさの突出度」で比較する。
    # 期限のように全マスクへ一様に高い値を出すキーは、中央値を引くとほぼ0になり、
    # 本当に効いているキーだけが残る。
    n_keys = 4  # ref, display_name, date, color(2026-08-21試験導入)
    per_key = np.zeros((n_mask, n_master, n_keys), dtype=np.float64)
    for i in range(n_mask):
        c = combined[i]
        for j, m in enumerate(master):
            for k, (_, key, is_date) in enumerate(_keys_of(m)):
                per_key[i, j, k] = _key_score(key, c, is_date=is_date)
            per_key[i, j, 3] = _color_score(mask_mean_rgb[i], m.get("color_rgb"))

    centered = np.zeros_like(per_key)
    for j in range(n_master):
        for k in range(n_keys):
            col = per_key[:, j, k]
            centered[:, j, k] = col - (np.median(col) if col.size else 0.0)
    centered[:, :, 3] *= COLOR_KEY_WEIGHT  # color キーの影響を減衰(理由は定数定義部を参照)

    S = centered.max(axis=2)

    # ===== 割当 =====
    if use_hungarian and n_mask > 0 and n_master > 0:
        if _HAS_SCIPY:
            rows, cols = linear_sum_assignment(-S)
            assign = {int(j): int(i) for i, j in zip(rows, cols)}
        else:
            assign = _greedy_assign(S)
        method = "hungarian" if _HAS_SCIPY else "greedy"
    else:
        assign = {target_j: int(S[:, target_j].argmax())} if n_mask else {}
        method = "independent"

    sel_i = assign.get(target_j)
    if sel_i is None and n_mask:
        sel_i = int(S[:, target_j].argmax())

    # ===== 信頼度 =====
    # 報告するスコア/marginは「実際に効いたキー」の生スコアで出す。
    # 中央値を引いた値は比較用の内部量で、そのまま出すと意味が読み取れないため。
    key_names = ["ref", "display_name", "date", "color"]
    col = S[:, target_j]
    if sel_i is not None:
        win_k = int(np.argmax(centered[sel_i, target_j]))
        raw_col = per_key[:, target_j, win_k]
        srt = np.sort(raw_col)[::-1]
        top = float(srt[0])
        second = float(srt[1]) if srt.size > 1 else 0.0
        # 採用マスクが、そのキーで実際に最上位かどうかも見る
        top = float(raw_col[sel_i])
        others = np.delete(raw_col, sel_i)
        margin = top - (float(others.max()) if others.size else 0.0)
        win_key = key_names[win_k]
    else:
        top, margin, win_key = 0.0, 0.0, "none"
    selected_text = combined[sel_i] if sel_i is not None else ""
    text_plausible = _looks_like_plausible_identifier(selected_text)
    confident = (top >= CONFIDENT_SCORE) and (margin >= CONFIDENT_MARGIN) and text_plausible

    # ===== 本番形式へ =====
    results: list[dict[str, Any]] = []
    if sel_i is not None:
        order = [sel_i] + [i for i in np.argsort(-raw_col).tolist() if i != sel_i]
        for i in order:
            sc = float(raw_col[i])
            if i != sel_i and sc <= threshold:
                continue
            results.append({
                "name": f"mask_{i + 1}",
                "score": int(round(sc)),
                "box": boxes[i],
                "forced_angle": FORCED_ANGLE,
            })

    # ===== デバッグ保存 =====
    try:
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "multikey_match_debug.json").write_text(
            json.dumps({
                "query": query,
                "master_json": str(master_path),
                "master_row": master[target_j],
                "assign_method": method,
                "threshold": threshold,
                "selected_mask": None if sel_i is None else f"mask_{sel_i + 1}",
                "selected_score": round(top, 1),
                "winning_key": win_key,
                "margin": round(margin, 1),
                "selected_text_len": len(_normalize(selected_text)),
                "text_plausible": bool(text_plausible),
                "confident": bool(confident),
                "confident_rule": (
                    f"score>={CONFIDENT_SCORE} and margin>={CONFIDENT_MARGIN} "
                    f"and text_plausible(len>={MIN_TEXT_LEN_FOR_PLAUSIBLE} and looks like REF/date)"
                ),
                "per_mask": [
                    {
                        "mask": f"mask_{i + 1}",
                        "score": round(float(raw_col[i]), 1),
                        "by_key": {
                            key_names[k]: round(float(per_key[i, target_j, k]), 1)
                            for k in range(n_keys)
                        },
                        "text": combined[i][:200],
                    }
                    for i in range(n_mask)
                ],
                "ocr_assignments": ocr_debug,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[multikey] デバッグ保存に失敗（処理は継続）: {e}")

    if not confident:
        print(f"[multikey] 警告: query='{query}' は確信度が低いです "
              f"(score={top:.1f}, margin={margin:.1f})")

    return results


# 呼び出し側が find_similar_books を使っている場合のための別名
find_similar_books = match_text_to_mask_main
