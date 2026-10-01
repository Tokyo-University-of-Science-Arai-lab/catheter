#!/usr/bin/env python3
"""reco/<元データセット>/ の画像の左右両端を切り取った、新しいデータセットを作る。

目的(2026-09-26、ユーザー要望):
    「リストにあるが画像に映っていない品目」の状況を、実際の写真から作る。
    棚の写真の左右の端を切り取ると、端にあった箱が写らなくなる(または見切れる)。
    箱と箱の境目できれいには切らない。ロボットの撮影は毎回良い条件とは限らないため、
    見切れた箱が残ってもよい(見切れ具合による認識の変化は今後の課題)。

使い方(リポジトリルートから):
    python3 reco/scripts/make_cropped_dataset.py --src 0911 --dst 0911_crop10 --ratio 0.10

切り取るもの(元データセットは一切変更しない):
    depth_shots/<N>/after_init_rgb.png, after_init_depth.npy  同じ範囲で切る
    depth_shots/<N>/camera_params.json                        width と ppx を更新(fx,fy,ppyは不変)
    images/<N>.png                                            アノテーション用の画像(depth_shots
                                                              に対して180度回転しているが、左右対称に
                                                              切るので同じ処理でよい)
    annotations/*.json (COCO)                                 ポリゴンを平行移動し、切り取り範囲でクリップ。
                                                              範囲外に出たものは削除。bbox/areaは再計算
    crop_info.json                                            切り取りの条件と、画像ごとの残った/消えたアノテーション数

切り取り範囲は、左右とも round(幅 * ratio) px。1280px・ratio=0.10なら x0=128, x1=1152(幅1024)。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon, box as shapely_box

REPO_ROOT = Path(__file__).resolve().parents[2]
RECO_ROOT = REPO_ROOT / "reco"


def crop_polygon(flat: list[float], x0: float, x1: float, height: float) -> list[list[float]]:
    """1つのポリゴン(flat=[x1,y1,x2,y2,...])を[x0,x1]の範囲でクリップし、x0だけ左へずらす。
    範囲外なら空リスト。クリップ後に複数の塊に分かれた場合は、それぞれ別ポリゴンで返す。"""
    pts = np.asarray(flat, dtype=float).reshape(-1, 2)
    if len(pts) < 3:
        return []
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    clipped = poly.intersection(shapely_box(x0, 0.0, x1, height))
    if clipped.is_empty:
        return []
    parts = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
    out = []
    for part in parts:
        if part.geom_type != "Polygon" or part.area < 1.0:
            continue
        coords = np.asarray(part.exterior.coords)[:-1]
        coords[:, 0] -= x0
        out.append(coords.reshape(-1).round(2).tolist())
    return out


def crop_coco(src_json: dict, x0: int, x1: int, new_w: int) -> tuple[dict, dict]:
    dst = json.loads(json.dumps(src_json))  # deep copy
    heights = {im["id"]: im["height"] for im in dst["images"]}
    for im in dst["images"]:
        im["width"] = new_w
    new_anns, stats = [], {}
    next_id = 1
    for ann in src_json["annotations"]:
        seg = ann.get("segmentation")
        st = stats.setdefault(ann["image_id"], {"元": 0, "残った": 0, "範囲外で消えた": 0})
        st["元"] += 1
        if not isinstance(seg, list):  # RLE(dict)は対象外。0911には無い
            raise ValueError(f"ポリゴン以外のsegmentationは未対応です(ann id={ann['id']})")
        polys = []
        for flat in seg:
            polys += crop_polygon(flat, x0, x1, heights[ann["image_id"]])
        if not polys:
            st["範囲外で消えた"] += 1
            continue
        st["残った"] += 1
        xs = [p[0::2] for p in polys]
        ys = [p[1::2] for p in polys]
        x_min, x_max = min(min(v) for v in xs), max(max(v) for v in xs)
        y_min, y_max = min(min(v) for v in ys), max(max(v) for v in ys)
        area = sum(Polygon(np.asarray(p).reshape(-1, 2)).area for p in polys)
        new = dict(ann)
        new.update(id=next_id, segmentation=polys,
                   bbox=[round(x_min, 2), round(y_min, 2), round(x_max - x_min, 2), round(y_max - y_min, 2)],
                   area=round(area, 2))
        new_anns.append(new)
        next_id += 1
    dst["annotations"] = new_anns
    return dst, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="元のデータセット名(reco/<名前>/)")
    ap.add_argument("--dst", required=True, help="作るデータセット名(reco/<名前>/、既にあると中止)")
    ap.add_argument("--ratio", type=float, default=0.10, help="左右それぞれ切り取る割合(既定0.10)")
    args = ap.parse_args()

    src, dst = RECO_ROOT / args.src, RECO_ROOT / args.dst
    if not src.is_dir():
        raise SystemExit(f"元データセットがありません: {src}")
    if dst.exists():
        raise SystemExit(f"{dst} は既にあります。上書きしないので、別の名前にするか、先に手で移動してください")

    crop_info = {"src": args.src, "ratio": args.ratio, "shots": {}, "annotations": {}}

    # --- depth_shots ---
    for shot in sorted((src / "depth_shots").iterdir()):
        if not shot.is_dir():
            continue
        out = dst / "depth_shots" / shot.name
        out.mkdir(parents=True)
        rgb = cv2.imread(str(shot / "after_init_rgb.png"))
        depth = np.load(shot / "after_init_depth.npy")
        h, w = rgb.shape[:2]
        assert depth.shape[:2] == (h, w), f"RGBとdepthの大きさが違います: {shot}"
        x0 = int(round(w * args.ratio))
        x1 = w - x0
        cv2.imwrite(str(out / "after_init_rgb.png"), rgb[:, x0:x1])
        np.save(out / "after_init_depth.npy", depth[:, x0:x1])
        cam = json.loads((shot / "camera_params.json").read_text())
        cam["width"] = x1 - x0
        cam["ppx"] = cam["ppx"] - x0
        (out / "camera_params.json").write_text(json.dumps(cam, indent=2, ensure_ascii=False))
        for f in shot.iterdir():  # 上の3つ以外(あれば)はそのままコピー
            if f.name not in ("after_init_rgb.png", "after_init_depth.npy", "camera_params.json"):
                shutil.copy2(f, out / f.name)
        crop_info["shots"][shot.name] = {"元の幅": w, "x0": x0, "x1": x1, "新しい幅": x1 - x0, "高さ": h}

    # --- images(アノテーション用) ---
    if (src / "images").is_dir():
        (dst / "images").mkdir(parents=True)
        for f in sorted((src / "images").glob("*.png")):
            im = cv2.imread(str(f))
            h, w = im.shape[:2]
            x0 = int(round(w * args.ratio))
            cv2.imwrite(str(dst / "images" / f.name), im[:, x0:w - x0])

    # --- annotations ---
    if (src / "annotations").is_dir():
        (dst / "annotations").mkdir(parents=True)
        for f in sorted((src / "annotations").glob("*.json")):
            data = json.loads(f.read_text())
            if not data.get("images"):  # 空ファイル(0911のinstances_default.json)はそのままコピー
                shutil.copy2(f, dst / "annotations" / f.name)
                continue
            w = data["images"][0]["width"]
            x0 = int(round(w * args.ratio))
            new, stats = crop_coco(data, x0, w - x0, w - 2 * x0)
            (dst / "annotations" / f.name).write_text(json.dumps(new, ensure_ascii=False, indent=2))
            crop_info["annotations"][f.name] = {
                str(k): v for k, v in sorted(stats.items())} | {"合計": {
                    "元": sum(v["元"] for v in stats.values()),
                    "残った": sum(v["残った"] for v in stats.values()),
                    "範囲外で消えた": sum(v["範囲外で消えた"] for v in stats.values())}}

    (dst / "crop_info.json").write_text(json.dumps(crop_info, ensure_ascii=False, indent=2))
    print(f"作成しました: {dst}")
    print(json.dumps(crop_info["annotations"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
