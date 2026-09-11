"""工程2（マスクと文字列の紐づけ）の実データ可視化を作る。

マスクごとに色分けし、そのマスクへ帰属した文字領域を同じ色の枠で描くことで、
「どの文字がどの商品に振り分けられたか」を示す。

対象の各 work フォルダ（<結果フォルダ>/work/<ショット>/）に
`link_overlay.png` を出力する。デバッグ出力と同じ場所に置くので、
どのショットの可視化かがフォルダ名で分かる。

使い方:
    python3 make_link_figs.py                    # 既定の結果フォルダを処理
    python3 make_link_figs.py <結果フォルダ>      # 別の結果フォルダを指定
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

RECO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../reco
DEFAULT_RESULT = os.path.join(RECO, 'ROOB-1', 'reco_result_20260828_201346')
OUT_NAME = 'link_overlay.png'

# 隣り合う箱で色が似ないよう、彩度の高い色を離して並べる
PALETTE = [
    (230, 60, 50), (40, 120, 230), (30, 170, 100), (245, 165, 20),
    (160, 70, 200), (0, 175, 185), (235, 100, 165), (120, 165, 45),
    (205, 110, 55), (75, 95, 220), (45, 190, 145), (215, 60, 120),
    (95, 145, 235), (180, 140, 30), (110, 60, 190), (0, 150, 160),
]

NEEDED = ('after_init_rgb.png', 'sam3_service_masks.npz',
          'ocr_result.json', 'multikey_match_debug.json')


def unrotate(poly, h, angle=90):
    """OCRは90度回転画像に対して実行されるため、元の座標系へ戻す。"""
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    if int(angle) % 360 == 90:
        xo, yo = y, (h - 1) - x
    else:
        xo, yo = x, y
    return np.stack([xo, yo], axis=1)


def build(shot_dir):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(shot_dir, 'after_init_rgb.png')),
                       cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    masks = np.load(os.path.join(shot_dir, 'sam3_service_masks.npz'))['masks']
    dbg = json.load(open(os.path.join(shot_dir, 'multikey_match_debug.json')))
    ocr = json.load(open(os.path.join(shot_dir, 'ocr_result.json')))
    polys = ocr.get('dt_polys') or ocr.get('rec_polys') or []
    assign = {e['ocr_index']: e for e in dbg.get('ocr_assignments', [])}

    # --- マスクを色分けして重ねる ---
    base = Image.fromarray(rgb).convert('RGBA')
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    for i in range(len(masks)):
        col = PALETTE[i % len(PALETTE)]
        tile = np.zeros((h, w, 4), dtype=np.uint8)
        tile[masks[i]] = (*col, 80)
        layer = Image.alpha_composite(layer, Image.fromarray(tile))
    img = Image.alpha_composite(base, layer)

    # --- 帰属した文字領域を、そのマスクと同じ色の枠で描く ---
    draw = ImageDraw.Draw(img, 'RGBA')
    n_drawn = 0
    for idx, poly in enumerate(polys, start=1):
        a = assign.get(idx)
        if not a or not a.get('matched'):
            continue
        try:
            mi = int(str(a['matched']).split('_')[1]) - 1
        except (IndexError, ValueError):
            continue
        col = PALETTE[mi % len(PALETTE)]
        pts = [tuple(map(float, p)) for p in unrotate(poly, h)]
        draw.line(pts + [pts[0]], fill=col + (255,), width=4)
        n_drawn += 1

    # 撮影画像は OCR 用に180度回転しているので、見る向きへ戻す
    out = img.convert('RGB').rotate(180, expand=True)
    path = os.path.join(shot_dir, OUT_NAME)
    out.save(path)
    return len(masks), n_drawn


def build_all(result_dir, quiet=False):
    """結果フォルダ配下の全ショットに link_overlay.png を作る。

    width_mm_validation.py から認識完了後に呼ばれる。
    """
    work = os.path.join(result_dir, 'work')
    if not os.path.isdir(work):
        work = result_dir  # work を直接渡された場合
    if not os.path.isdir(work):
        return 0, 0
    shots = sorted(d for d in os.listdir(work)
                   if os.path.isdir(os.path.join(work, d)) and '_attempt' not in d)
    if not quiet:
        print(f'対象: {work}\nショット数: {len(shots)}')
    done = skipped = 0
    for s in shots:
        d = os.path.join(work, s)
        if not all(os.path.exists(os.path.join(d, f)) for f in NEEDED):
            skipped += 1
            continue
        try:
            nm, nt = build(d)
            done += 1
        except Exception as e:  # noqa: BLE001
            if not quiet:
                print(f'  × {s}: {type(e).__name__}: {e}')
            skipped += 1
    if not quiet:
        print(f'生成 {done} 件 / スキップ {skipped} 件  （各フォルダに {OUT_NAME}）')
    return done, skipped


if __name__ == '__main__':
    build_all(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULT)
