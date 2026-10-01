#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVATの代わりに、PC内のアノテーション(COCO 1.0互換のinstances_default*.json)を
ブラウザで直接編集するためのローカルサーバー(2026-09-26、ユーザー要望:
「毎回CVATへ画像をアップロードするのが面倒なのでPC内で完結させたい」)。

想定の手順: auto_annotate.py で instances_default_auto.json を作る
    → このアプリで _auto.json を開いて修正 → instances_default.json として保存。

・依存はPython標準ライブラリ + cv2/numpy のみ(FastAPI等は未導入のため使わない)。
・reco/<dataset>/ と anno-catheter/<N>/ のうち annotations/*.json を持つものを一覧表示する。
・画像は images/ 以下(サブフォルダ含む。例: reco/diagonal-40/images/default/)から
  file_name で探す。images/ 側がアノテーション座標の基準の向き。
  depth_shots/ 側は古いデータセット(stand-100, diagonal-40)では180度回転しているため、
  images/ に無い場合だけ代用し、画面上で回転させて合わせられるようにしている。
・RLE(segmentationがdict)は表示用にポリゴンへ変換して送るが、編集しなかったものは
  保存時に元のRLEのまま書き戻す(編集したものだけポリゴンになる。ユーザー判断(b))。
・保存時、保存先が既にあれば annotations/backup/<名前>.<日時>.json に退避してから書く。
  bbox/area はポリゴンから再計算する。score/attributes等の他のキーは保持する。

実行(リポジトリルートから):
    python anno_editor/server.py            # http://127.0.0.1:8780 をブラウザで開く
    (8765はSAM3サービスが使っているので避けている)
    python anno_editor/server.py --port 8800
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sys
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATASET_PARENTS = ("reco", "anno-catheter")
IMAGE_EXTS = (".png", ".jpg", ".jpeg")

# ポリゴン簡略化は auto_annotate.py と同じもの(8〜12点)を使う
sys.path.insert(0, str(REPO_ROOT / "reco" / "scripts"))
from auto_annotate import simplify_contour_to_target  # noqa: E402


# ---------------------------------------------------------------- データセット

def list_datasets() -> list[dict]:
    out = []
    for parent in DATASET_PARENTS:
        base = REPO_ROOT / parent
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            ann_dir = d / "annotations"
            if not ann_dir.is_dir():
                continue
            files = sorted(p.name for p in ann_dir.glob("*.json"))
            if files:
                out.append({"id": f"{parent}/{d.name}", "files": files})
    return out


def dataset_dir(ds_id: str) -> Path:
    """ds_id(例: reco/0911)を検証してパスにする。一覧に無いものは拒否する。"""
    parent, _, name = ds_id.partition("/")
    if parent not in DATASET_PARENTS or not name or "/" in name or name in (".", ".."):
        raise ValueError(f"不正なデータセット指定: {ds_id}")
    d = REPO_ROOT / parent / name
    if not (d / "annotations").is_dir():
        raise ValueError(f"annotations/ がありません: {ds_id}")
    return d


def safe_json_name(name: str) -> str:
    if "/" in name or "\\" in name or not name.endswith(".json") or name.startswith("."):
        raise ValueError(f"不正なファイル名: {name}")
    return name


def find_image(ds: Path, file_name: str) -> tuple[Path | None, str]:
    """(パス, 由来)。由来は "images" か "depth_shots"(向きが違う可能性あり)。"""
    images = ds / "images"
    if images.is_dir():
        direct = images / file_name
        if direct.is_file():
            return direct, "images"
        for p in sorted(images.rglob(Path(file_name).name)):
            if p.is_file():
                return p, "images"
    stem = Path(file_name).stem
    shot = ds / "depth_shots" / stem
    if shot.is_dir():
        for cand in (shot / "after_init_rgb.png", shot / file_name):
            if cand.is_file():
                return cand, "depth_shots"
        pngs = sorted(p for p in shot.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if pngs:
            return pngs[0], "depth_shots"
    return None, ""


# ---------------------------------------------------------------- RLE / ポリゴン

def rle_decode(seg: dict) -> np.ndarray:
    """非圧縮RLE(countsがlist、CVATのCOCO出力形式)をマスクにする。列優先(Fortran順)。"""
    h, w = seg["size"]
    counts = seg["counts"]
    if isinstance(counts, str):
        raise ValueError("圧縮RLE(counts文字列)は未対応")
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for n in counts:
        if val:
            flat[pos:pos + n] = 1
        pos += n
        val ^= 1
    return np.ascontiguousarray(flat.reshape((w, h)).T)


def rle_to_polygon(seg: dict) -> list[float] | None:
    mask = rle_decode(seg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    pts = simplify_contour_to_target(contour).astype(float)
    return pts.reshape(-1).tolist() if len(pts) >= 3 else None


def polygon_area(flat: list[float]) -> float:
    xs, ys = np.array(flat[0::2]), np.array(flat[1::2])
    return float(abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2.0)


def polygons_bbox(polys: list[list[float]]) -> list[float]:
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    x0, y0 = min(xs), min(ys)
    return [round(x0, 2), round(y0, 2), round(max(xs) - x0, 2), round(max(ys) - y0, 2)]


# ---------------------------------------------------------------- 読み込み / 保存

def load_for_editor(ds_id: str, name: str) -> dict:
    ds = dataset_dir(ds_id)
    data = json.loads((ds / "annotations" / safe_json_name(name)).read_text(encoding="utf-8"))
    images = []
    for im in data.get("images", []):
        path, origin = find_image(ds, im["file_name"])
        images.append({**im, "found": path is not None, "origin": origin})
    anns = []
    for a in data.get("annotations", []):
        seg = a.get("segmentation")
        a = dict(a)
        if isinstance(seg, dict):
            poly = rle_to_polygon(seg)
            a["_rle"] = True
            a["_polys"] = [poly] if poly else []
        else:
            a["_rle"] = False
            a["_polys"] = [list(map(float, p)) for p in (seg or []) if len(p) >= 6]
        anns.append(a)
    return {"raw": data, "images": images, "annotations": anns}


def build_output(raw: dict, anns: list[dict]) -> dict:
    """エディタから返ってきた annotations を COCO 形式に戻す。
    _edited が無いRLEは元のsegmentationのまま、それ以外はポリゴンとして bbox/area を再計算。"""
    out_anns = []
    next_id = 1
    for a in anns:
        polys = [p for p in a.get("_polys", []) if len(p) >= 6]
        keep_rle = a.get("_rle") and not a.get("_edited")
        if not keep_rle and not polys:
            continue  # 頂点が3点未満になったものは捨てる
        b = {k: v for k, v in a.items() if not k.startswith("_")}
        b["id"] = next_id
        next_id += 1
        if not keep_rle:
            polys = [[round(float(v), 2) for v in p] for p in polys]
            b["segmentation"] = polys
            b["bbox"] = polygons_bbox(polys)
            b["area"] = round(sum(polygon_area(p) for p in polys), 2)
            b["iscrowd"] = 0
        b.setdefault("attributes", {}).setdefault("occluded", False)
        out_anns.append(b)
    out = dict(raw)
    out["annotations"] = out_anns
    return out


def save(ds_id: str, name: str, raw: dict, anns: list[dict]) -> dict:
    ds = dataset_dir(ds_id)
    target = ds / "annotations" / safe_json_name(name)
    backup = None
    if target.exists():
        bdir = ds / "annotations" / "backup"
        bdir.mkdir(exist_ok=True)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        backup = bdir / f"{target.stem}.{stamp}.json"
        n = 1
        while backup.exists():  # 同じ秒に複数回保存しても上書きしない
            backup = bdir / f"{target.stem}.{stamp}_{n}.json"
            n += 1
        shutil.copy2(target, backup)
    out = build_output(raw, anns)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)
    return {"saved": str(target.relative_to(REPO_ROOT)),
            "backup": str(backup.relative_to(REPO_ROOT)) if backup else None,
            "count": len(out["annotations"])}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # アクセスログは出さない
        pass

    def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status=HTTPStatus.OK):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                self._send((STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/datasets":
                self._json(list_datasets())
            elif u.path == "/api/load":
                self._json(load_for_editor(q["ds"], q["file"]))
            elif u.path == "/api/image":
                path, _ = find_image(dataset_dir(q["ds"]), q["name"])
                if path is None:
                    self._json({"error": "画像が見つかりません"}, HTTPStatus.NOT_FOUND)
                    return
                ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._send(path.read_bytes(), ctype)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if u.path == "/api/save":
                self._json(save(body["ds"], body["file"], body["raw"], body["annotations"]))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    ap = argparse.ArgumentParser(description="ローカル・アノテーション編集アプリ")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        sys.exit(f"ポート{args.port}を使えません({e})。既に起動していないか確認するか、"
                 f"--port で別の番号を指定してください。")
    print(f"アノテーション編集アプリ: http://{args.host}:{args.port}  (Ctrl+Cで終了)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
