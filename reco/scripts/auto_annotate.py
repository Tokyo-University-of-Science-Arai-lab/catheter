#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
これまでCVATで手動で描いていたポリゴンアノテーション(instances_default.json)を、
新しく差し替えたSAM3(カテーテル学習済み)モデルで自動生成する。

reco/<dataset>/images/*.png (無ければ depth_shots/*/after_init_rgb.png で代用) を
1枚ずつSAM3サービスへ投げ、クエリ(book_name)指定なしで画像内の全マスクを検出する
(multikey_matcherによる品目照合はかけない。マスク検出そのものの自動化が目的のため)。
検出結果を、既存のinstances_default.jsonと同じCOCO 1.0互換のポリゴン形式で書き出す。

ポリゴン簡略化のロジックは vendor/owner_repo/core/coco_export.py の
simplify_contour_to_target/mask_to_polygon と同じもの(CVAT編集しやすい8〜12点に間引く)。
このファイルはpycocotools依存(coco_export.py側はRLE出力用に読み込む)を避けるため、
ポリゴン出力に必要な部分だけをこの中に複製している。

instances_default.json が既存の場合(手動アノテーション等)だけ、誤って上書きしないよう
instances_default_auto.json に書き出す(--overwriteを付ければ直接上書きも可)。
まだ instances_default.json が無い新規フォルダ(anno-catheter/<N>/等)では、リネームの
手間を省くため最初から instances_default.json に直接書き込む(2026-08-25、ユーザー要望)。

reco/<dataset>/images/ に、instances_default.jsonのfile_nameと同じ名前でRGBを必ず
コピーしておく(depth_shots代用時も含む。2026-08-25、ユーザー要望: 「CVATにインポート
しやすくするため」)。これが無いと、JSON側のfile_nameとCVATにアップロードした実ファイル名が
一致せず`Could not match item id`エラーになる。

前提: SAM3サービスが起動していること
    (detection/pro_handbook/sam3_runtime/scripts/check_service.sh で確認できる)。

実行:
    python3 reco/scripts/auto_annotate.py --dataset OPTIMA-No1
    python3 reco/scripts/auto_annotate.py --dataset-dir anno-catheter/4
    python3 reco/scripts/auto_annotate.py --dataset OPTIMA-No1 --only 2
    python3 reco/scripts/auto_annotate.py --dataset OPTIMA-No1 --overwrite
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

RECO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RECO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from detection.pro_handbook.sam3_runtime.service.client import Sam3BatchInfer, Sam3ServiceError  # noqa: E402

CATEGORY_NAME = "book_spine"
POLYGON_TARGET_POINTS = 8
POLYGON_MAX_POINTS = 12
MIN_MASK_AREA_PX = 1


def mask_bbox_xywh(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def simplify_contour_to_target(contour: np.ndarray, target_points: int = POLYGON_TARGET_POINTS,
                                max_points: int = POLYGON_MAX_POINTS) -> np.ndarray:
    """core/coco_export.pyのsimplify_contour_to_targetと同じロジック。"""
    if len(contour) < 3:
        return contour.reshape(-1, 2)
    peri = cv2.arcLength(contour, True)
    if peri <= 0:
        return contour.reshape(-1, 2)
    lo, hi = 0.0, peri
    best = cv2.approxPolyDP(contour, peri * 0.02, True)
    for _ in range(40):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(contour, mid, True)
        if len(approx) > target_points:
            lo = mid
        else:
            best = approx
            hi = mid
    points = best.reshape(-1, 2)
    if len(points) < 3:
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect).astype(np.int32)
    if len(points) > max_points:
        rect = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rect).astype(np.int32)
    return points


def mask_to_polygon(mask: np.ndarray):
    binary = mask.astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if float(cv2.contourArea(contour)) < MIN_MASK_AREA_PX or len(contour) < 3:
        return None
    points = simplify_contour_to_target(contour).astype(float)
    if len(points) < 3:
        return None
    bbox = [float(v) for v in mask_bbox_xywh(mask)]
    return points.reshape(-1).tolist(), bbox, float(mask.sum())


def resolve_images(dataset_dir: Path, images_dir_arg: str | None) -> tuple[list[Path], str]:
    """アノテーション対象の画像一覧を返す。images/があればそちら優先、
    無ければdepth_shots/*/after_init_rgb.pngで代用する。
    """
    if images_dir_arg:
        d = Path(images_dir_arg)
        return sorted(d.glob("*.png")), f"images-dir指定: {d}"

    images_dir = dataset_dir / "images"
    if images_dir.is_dir() and any(images_dir.glob("*.png")):
        return sorted(images_dir.glob("*.png")), f"{images_dir}"

    depth_shots_dir = dataset_dir / "depth_shots"
    paths = []
    if depth_shots_dir.is_dir():
        for shot_dir in sorted(depth_shots_dir.iterdir(), key=lambda p: (len(p.name), p.name)):
            rgb = shot_dir / "after_init_rgb.png"
            if shot_dir.is_dir() and rgb.exists():
                paths.append(rgb)
    return paths, (f"images/が無いため depth_shots/*/after_init_rgb.png で代用 "
                    f"({depth_shots_dir})。images/を用意した場合と回転が異なる可能性があるので、"
                    "1枚目の結果を必ず目視確認すること。")


def image_output_name(image_path: Path, dataset_dir: Path) -> str:
    """depth_shots由来の場合はshotフォルダ名を、images/由来の場合は元のファイル名を使う。"""
    if image_path.parent.name != "images" and image_path.parent.parent.name == "depth_shots":
        return f"{image_path.parent.name}.png"
    return image_path.name


OVERLAY_BOX_COLOR = (0, 0, 255)
OVERLAY_TEXT_COLOR = (0, 255, 255)


def draw_overlay(image_path: Path, anns: list[dict]) -> np.ndarray:
    """検出したポリゴンを画像に重ね描きする(目視確認用)。"""
    img = cv2.imread(str(image_path))
    for a in anns:
        pts = np.array(a["segmentation"], dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=OVERLAY_BOX_COLOR, thickness=2)
        score = a.get("score")
        if score is not None:
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(img, f"{score:.2f}", (x, max(15, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, OVERLAY_TEXT_COLOR, 1, cv2.LINE_AA)
    return img


def auto_annotate_image(runner: Sam3BatchInfer, image_path: Path):
    pil = Image.open(image_path).convert("RGB")
    masks, sam_data = runner.infer_masks(pil)
    anns = []
    for mask, meta in zip(masks, sam_data):
        result = mask_to_polygon(mask)
        if result is None:
            continue
        polygon, bbox, area = result
        anns.append({"segmentation": polygon, "bbox": bbox, "area": area, "score": meta.get("score")})
    return pil.size, anns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="reco/<dataset>/ 配下のデータセット名(reco/配下専用の簡易指定)。"
                         "reco/以外(例: anno-catheter/3)を対象にする場合は --dataset-dir を使うこと。")
    ap.add_argument("--dataset-dir", default=None,
                    help="depth_shots/・images/を含むフォルダを直接指定する(絶対パス、または"
                         "リポジトリルートからの相対パス)。例: anno-catheter/3 。"
                         "--datasetと排他、こちらを指定した場合はreco/配下に限定されない"
                         "(2026-08-25、ユーザー要望: anno-catheter/配下でも使えるようにしてほしい)。")
    ap.add_argument("--images-dir", default=None,
                    help="画像フォルダを明示指定(省略時は <dataset_dir>/images/、"
                         "無ければdepth_shots/*/after_init_rgb.pngで自動代用)")
    ap.add_argument("--only", default=None, help="このファイル名(拡張子省略可)だけ処理する")
    ap.add_argument("--output", default=None,
                    help="出力パス(省略時は <dataset_dir>/annotations/instances_default_auto.json)")
    ap.add_argument("--overwrite", action="store_true",
                    help="<dataset_dir>/annotations/instances_default.json に直接上書きする"
                         "(既存の手動アノテーションを消す可能性があるため、明示指定時のみ)")
    ap.add_argument("--prompt", default="book spine")
    args = ap.parse_args()

    if bool(args.dataset) == bool(args.dataset_dir):
        raise SystemExit("--dataset と --dataset-dir のどちらか一方だけを指定してください")
    dataset_dir = (RECO_ROOT / args.dataset) if args.dataset else Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"データセットフォルダが見つかりません: {dataset_dir}")

    image_paths, source_note = resolve_images(dataset_dir, args.images_dir)
    print(f"画像ソース: {source_note}")

    if args.only:
        stem = Path(args.only).stem
        image_paths = [p for p in image_paths if image_output_name(p, dataset_dir).rsplit(".", 1)[0] == stem]
    if not image_paths:
        raise SystemExit("対象画像が見つかりません")

    default_path = dataset_dir / "annotations" / "instances_default.json"
    if args.overwrite:
        output_path = default_path
    elif args.output:
        output_path = Path(args.output)
    elif default_path.exists():
        # 既存の(手動アノテーション等の)instances_default.jsonがある場合だけ安全側に倒す。
        # 無ければリネームの手間を省くため最初からinstances_default.jsonに直接書く
        # (2026-08-25、ユーザー要望: 確認後に毎回手でリネームするのが面倒)。
        output_path = dataset_dir / "annotations" / "instances_default_auto.json"
        print(f"  ※ 既存の {default_path.name} があるため、{output_path.name} に書き出します"
              f"(上書きするには --overwrite を付けてください)")
    else:
        output_path = default_path

    print(f"SAM3サービスへ接続します(prompt={args.prompt!r})...")
    runner = Sam3BatchInfer(prompt=args.prompt)

    coco: dict = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "SAM3 auto-annotation",
                  "url": "", "version": "", "year": ""},
        "categories": [{"id": 1, "name": CATEGORY_NAME, "supercategory": ""}],
        "images": [],
        "annotations": [],
    }
    overlays_dir = output_path.parent / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # CVATへのインポート用に、instances_default.jsonのfile_nameと同じ名前で
    # reco/<dataset>/images/ にもRGBを置いておく(2026-08-25、ユーザー要望: 「CVATに
    # インポートしやすくするため」。images/がもともと無い場合(depth_shots代用時)、
    # これが無いとJSONのfile_nameとCVATにアップロードした実ファイル名が一致せず
    # `Could not match item id` エラーになる、という実例が過去にあったため)。
    images_out_dir = dataset_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    ann_id = 1
    for image_id, image_path in enumerate(image_paths, start=1):
        out_name = image_output_name(image_path, dataset_dir)
        dst = images_out_dir / out_name
        if image_path.resolve() != dst.resolve():
            shutil.copyfile(image_path, dst)
        print(f"[{image_id}/{len(image_paths)}] {out_name} ...", flush=True)
        try:
            (width, height), anns = auto_annotate_image(runner, image_path)
        except Sam3ServiceError as e:
            print(f"  ✗ SAM3サービスに失敗: {e}")
            continue
        coco["images"].append({
            "id": image_id, "width": width, "height": height, "file_name": out_name,
            "license": 0, "flickr_url": "", "coco_url": "", "date_captured": 0,
        })
        for a in anns:
            coco["annotations"].append({
                "id": ann_id, "image_id": image_id, "category_id": 1,
                "segmentation": [a["segmentation"]], "area": a["area"], "bbox": a["bbox"],
                "iscrowd": 0, "attributes": {"occluded": False}, "score": a["score"],
            })
            ann_id += 1
        overlay_img = draw_overlay(image_path, anns)
        cv2.imwrite(str(overlays_dir / out_name), overlay_img)
        print(f"  -> {len(anns)}件検出 (確認用画像: {overlays_dir / out_name})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✔ {len(coco['images'])}枚・{len(coco['annotations'])}件のアノテーション -> {output_path}")
    print(f"✔ CVATインポート用画像 -> {images_out_dir}")
    print(f"✔ 目視確認用オーバーレイ画像 -> {overlays_dir}")
    if output_path != default_path:
        print(f"  ※ 既存の {default_path.name} を保護するため {output_path.name} に書き出しました。")
        print("     内容を確認し、問題なければ --overwrite を付けて再実行するか、")
        print(f"     ファイル名を {default_path.name} にリネームしてください。")


if __name__ == "__main__":
    main()
