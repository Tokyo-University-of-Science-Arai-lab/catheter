import json
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz


def unrotate_poly_to_original(poly, angle: int, w: int, h: int):
    """
    poly: [[x,y]x4]  (OCRが返す座標: 回転後画像座標)
    angle: doc_preprocessor_res["angle"] (0/90/180/270)
           ※ここでは「時計回り(angle)だけ回転された画像上の座標」を元に戻す前提
    w,h: 元画像サイズ（SAMと同じ画像の width/height）
    return: 元画像座標系に戻したpoly
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]

    a = int(angle) % 360
    if a == 0:
        xo, yo = x, y
    elif a == 90:
        # 90° clockwise の逆変換
        xo = y
        yo = (h - 1) - x
    elif a == 180:
        xo = (w - 1) - x
        yo = (h - 1) - y
    elif a == 270:
        # 270° clockwise (= 90° CCW) の逆変換
        xo = (w - 1) - y
        yo = x
    else:
        xo, yo = x, y

    return np.stack([xo, yo], axis=1).tolist()


def compute_min_arearect_list(masks, min_area_px=50):
    """
    互換性のため残している関数。
    今回の IoU ベース対応付けでは未使用。
    """
    min_arearect = []
    for m in masks:
        b = (np.asarray(m) > 0).astype(np.uint8)
        if b.sum() < min_area_px:
            min_arearect.append(None)
            continue

        cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            min_arearect.append(None)
            continue

        rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        min_arearect.append(rect if rect[1][0] * rect[1][1] > 1e-6 else None)
    return min_arearect


def rects_to_pts(rects):
    """
    互換性のため残している関数。
    今回の IoU ベース対応付けでは未使用。
    """
    quads = []
    for r in rects:
        if r is None:
            continue
        quads.append(cv2.boxPoints(r).astype(np.float32))
    return quads


def load_text_data(shot_dir):
    with open(shot_dir, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_min_arearect = data.get("dt_polys", [])
    all_mojis = data.get("rec_texts", [])
    return all_min_arearect, all_mojis


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _mask_to_binary(mask, h: int, w: int) -> np.ndarray:
    """
    任意の mask を 0/1 の uint8 2次元配列にそろえる
    """
    b = (np.asarray(mask) > 0).astype(np.uint8)

    if b.ndim > 2:
        b = np.squeeze(b)

    if b.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, but got shape={b.shape}")

    if b.shape != (h, w):
        raise ValueError(f"mask shape mismatch: expected {(h, w)}, got {b.shape}")

    return b


def _poly_to_mask(poly, h: int, w: int) -> np.ndarray:
    """
    OCR polygon を塗りつぶした 0/1 mask に変換
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

    if pts.shape[0] < 3:
        return np.zeros((h, w), dtype=np.uint8)

    pts = np.round(pts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(canvas, [pts], 1)
    return canvas


def _mask_bbox(mask_bin: np.ndarray):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "x1": float(xs.min()),
        "y1": float(ys.min()),
        "x2": float(xs.max()),
        "y2": float(ys.max()),
    }


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    """
    0/1 mask 同士の IoU
    """
    inter = int(np.logical_and(mask_a > 0, mask_b > 0).sum())
    if inter == 0:
        return 0.0, 0, int(np.logical_or(mask_a > 0, mask_b > 0).sum())

    union = int(np.logical_or(mask_a > 0, mask_b > 0).sum())
    if union <= 0:
        return 0.0, inter, 0

    return float(inter / union), inter, union


def match_text_to_mask(
    all_boxes,
    all_mojis,
    sam_quads,   # ← 互換性のため名前はそのまま。中身は「SAM masks」を渡す
    rgb_path,
    forced_angle=None,
    return_debug=False,
):
    """
    OCR polygon と SAM マスクの IoU に基づいて文字列を本マスクへ割り当てる。

    Parameters
    ----------
    all_boxes : list
        OCR の polygon 一覧（dt_polys）
    all_mojis : list
        OCR の認識文字列一覧（rec_texts）
    sam_quads : list
        実際には SAM masks の list を渡す
    rgb_path : Path or str
        元画像パス
    forced_angle : int or None
        OCR polygon を元画像座標へ戻す回転角
    return_debug : bool
        True のとき debug 情報も返す

    Returns
    -------
    book_texts, book_boxes
    または
    book_texts, book_boxes, debug_rows
    """
    img = cv2.imread(str(rgb_path))
    if img is None:
        raise FileNotFoundError(f"画像が読めませんでした: {rgb_path}")

    h, w = img.shape[:2]
    if forced_angle in (90, 180, 270):
        all_boxes = [unrotate_poly_to_original(p, forced_angle, w=w, h=h) for p in all_boxes]

    # sam_quads という名前だが、中身は SAM の元マスクを受ける
    sam_masks = [_mask_to_binary(m, h, w) for m in sam_quads]

    names = [f"mask_{i+1}" for i in range(len(sam_masks))]
    book_texts = {n: [] for n in names}

    book_boxes = {}
    for n, m in zip(names, sam_masks):
        book_boxes[n] = _mask_bbox(m)

    debug_rows = []

    for idx, (moji, quad) in enumerate(zip(all_mojis, all_boxes), start=1):
        text_mask = _poly_to_mask(quad, h, w)
        text_area = int(text_mask.sum())

        if text_area <= 0:
            debug_rows.append({
                "ocr_index": idx,
                "text": moji,
                "matched": None,
                "best_iou": 0.0,
                "text_area": 0,
                "note": "text polygon area is zero",
            })
            continue

        best_name = None
        best_iou = 0.0
        best_inter = 0
        best_union = 0
        best_text_cover = 0.0

        per_mask_scores = []

        for name, sam_mask in zip(names, sam_masks):
            iou, inter, union = _compute_iou(text_mask, sam_mask)
            text_cover = float(inter / text_area) if text_area > 0 else 0.0

            per_mask_scores.append({
                "name": name,
                "iou": float(iou),
                "intersection": int(inter),
                "union": int(union),
                "text_cover": float(text_cover),
            })

            # まず IoU 最大を優先
            # 同点なら「OCR領域の何割がその本に含まれるか」で比較
            if (
                iou > best_iou
                or (
                    np.isclose(iou, best_iou)
                    and text_cover > best_text_cover
                )
            ):
                best_name = name
                best_iou = float(iou)
                best_inter = int(inter)
                best_union = int(union)
                best_text_cover = float(text_cover)

        # 少なくとも 1px でも重なっているものだけ採用
        matched = best_name if best_inter > 0 else None

        if matched is not None:
            book_texts[matched].append(moji)

        debug_rows.append({
            "ocr_index": idx,
            "text": moji,
            "matched": matched,
            "best_iou": float(best_iou),
            "best_intersection": int(best_inter),
            "best_union": int(best_union),
            "best_text_cover": float(best_text_cover),
            "text_area": int(text_area),
            "per_mask_scores": per_mask_scores,
        })

    if return_debug:
        return book_texts, book_boxes, debug_rows
    return book_texts, book_boxes


def match_book_name(query, book_texts, shot_dir, threshold=40.0):
    matches = []
    all_scores = []

    for name, text in book_texts.items():
        combined = " ".join(text) if isinstance(text, (list, tuple)) else str(text)
        score = int(fuzz.partial_ratio(query, combined))
        all_scores.append({"name": name, "score": score, "text": combined})

        if score > threshold:
            matches.append((name, score))

    matches.sort(key=lambda x: x[1], reverse=True)
    all_scores.sort(key=lambda d: d["score"], reverse=True)

    save_json(
        Path(shot_dir) / "similarity_scores.json",
        {
            "query": query,
            "threshold": threshold,
            "scores": all_scores,
        },
    )

    return matches, all_scores


def match_text_to_mask_main(query, masks, shot_dir, threshold=40):
    """
    既存コードからそのまま呼べる入口関数。
    run_capture_and_pca 側の修正は不要。
    """
    ocr_json_path = Path(shot_dir) / "ocr_result.json"
    rgb_path = Path(shot_dir) / "after_init_rgb.png"

    all_text_polys, all_texts = load_text_data(ocr_json_path)

    # ここで masks をそのまま渡し、IoU ベースで割り当てる
    book_texts, book_boxes, debug_rows = match_text_to_mask(
        all_text_polys,
        all_texts,
        masks,
        rgb_path,
        forced_angle=90,
        return_debug=True,
    )

    save_json(
        Path(shot_dir) / "text_mask_iou_debug.json",
        {
            "forced_angle": 90,
            "num_masks": len(masks),
            "num_ocr_boxes": len(all_texts),
            "assignments": debug_rows,
        },
    )

    matches, all_scores = match_book_name(query, book_texts, shot_dir, threshold)

    results = []
    for name, score in matches:
        results.append(
            {
                "name": name,
                "score": score,
                "box": book_boxes.get(name),
                "forced_angle": 90,
            }
        )
    return results