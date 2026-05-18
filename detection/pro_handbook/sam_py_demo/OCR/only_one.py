import json
from rapidfuzz import fuzz

from pathlib import Path
import json
from datetime import datetime
import numpy as np
import cv2

def unrotate_poly_to_original(poly, angle: int, w: int, h: int):
    """
    poly: [[x,y]x4]  (OCRが返す座標: 回転後画像座標)
    angle: doc_preprocessor_res["angle"] (0/90/180/270)
           ※ここでは「時計回り(angle)だけ回転された画像上の座標」を元に戻す前提
    w,h: 元画像サイズ（SAMと同じ画像の width/height）
    return: 元画像座標系に戻したpoly
    """
    pts = np.asarray(poly, dtype=np.float32)  # (4,2)
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
        # 想定外はそのまま返す
        xo, yo = x, y

    return np.stack([xo, yo], axis=1).tolist()

def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def intersection_area(a, b):
    x_overlap = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    y_overlap = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    return x_overlap * y_overlap

def extract_book_texts(ocr_json_path, sam_data, shot_dir=None, img_name="after_init_rgb.png", forced_angle: int | None = None):
    data = load_json(ocr_json_path)

    all_boxes = data.get("dt_polys", [])
    all_mojis = data.get("rec_texts", [])

    # ★ここ：手動回転したなら forced_angle を使う
    angle_json = int(data.get("doc_preprocessor_res", {}).get("angle", 0))
    angle = int(forced_angle) % 360 if forced_angle is not None else angle_json

    if shot_dir is None:
        raise ValueError("shot_dir が必要です（元画像サイズ取得のため）")
    img_path = Path(shot_dir) / img_name
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"image not found: {img_path}")
    h, w = img.shape[:2]

    if angle in (90, 180, 270):
        all_boxes = [unrotate_poly_to_original(poly, angle, w=w, h=h) for poly in all_boxes]

    # ★重要：dict(zip()) は同じ文字があると上書きされるのでやめる（空が増える原因にもなる）
    book_texts = {entry["name"]: [] for entry in sam_data}

    for moji, poly in zip(all_mojis, all_boxes):
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        moji_rect = {"x1": min_x, "y1": min_y, "x2": max_x, "y2": max_y}
        moji_area = (max_x - min_x) * (max_y - min_y)
        if moji_area <= 0:
            continue

        best_match = None
        best_ratio = 0.0

        for entry in sam_data:
            book_box = entry["box"]
            inter_area = intersection_area(moji_rect, book_box)
            ratio = inter_area / moji_area
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = entry["name"]

        if best_match:
            book_texts[best_match].append(moji)

    return book_texts, {entry["name"]: entry["box"] for entry in sam_data}

    

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
    # 全スコアを保存（可視化=見れる形）
    save_json(Path(shot_dir) / "similarity_scores.json", {
        "query": query,
        "threshold": threshold,
        "scores": all_scores, # [{"name","score","text"}, ...]
    })


    return matches, all_scores


def find_similar_books(query, sam_data, shot_dir, threshold=40):
    ocr_json_path = shot_dir / "ocr_result.json"
    book_texts, book_boxes = extract_book_texts(ocr_json_path, sam_data, shot_dir=shot_dir, forced_angle=90)  # ★ここ
    matches, all_scores = match_book_name(query, book_texts, shot_dir, threshold)  # ★ここ

    results = []
    for name, score in matches:
        results.append({
            "name": name,
            "score": score,
            "box": book_boxes.get(name),
            "forced_angle": 90
        })
    return results

# ✅ 使用你已经加载的 sam_data 来调用
#query = "熱力学"
#results = find_similar_books(query, sam_data)

#for r in results:
#    print(f"題名: {r['name']}, 類似度: {r['score']:.2f}, 座標: {r['box']}")