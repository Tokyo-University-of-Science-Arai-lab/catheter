"""従来手法と提案手法が選んだ領域を対比する図を作る。

従来手法（型番のみ・領域ごとに独立して最尤候補を選ぶ）を per_mask の
by_key['ref'] から再現し、提案手法（多キー照合＋一対一対応）の選択と並べる。

  赤 : 従来手法が選ぶ領域
  緑 : 提案手法が選ぶ領域（＝実際の出力）
  黄 : 両者が一致した領域

各 work フォルダに `compare_overlay.png` を出力する。

使い方:
    python3 make_compare_figs.py                 # 既定の結果フォルダ
    python3 make_compare_figs.py <結果フォルダ>
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RECO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULT = os.path.join(RECO, 'ROOB-1', 'reco_result_20260828_213247')
OUT_NAME = 'compare_overlay.png'

FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_R = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

RED = (225, 55, 45)      # 従来手法
GREEN = (30, 175, 100)   # 提案手法
YELLOW = (240, 180, 20)  # 一致

NEEDED = ('after_init_rgb.png', 'sam3_service_masks.npz', 'multikey_match_debug.json')


def mask_outline(mask):
    """マスクの外周輪郭を取り出す。"""
    m = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def build(shot_dir):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(shot_dir, 'after_init_rgb.png')),
                       cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    masks = np.load(os.path.join(shot_dir, 'sam3_service_masks.npz'))['masks']
    dbg = json.load(open(os.path.join(shot_dir, 'multikey_match_debug.json')))
    pm = dbg.get('per_mask', [])
    sel = dbg.get('selected_mask')
    if not pm or not sel:
        raise ValueError('per_mask / selected_mask がありません')

    def mi(name):
        return int(str(name).split('_')[1]) - 1

    sel_i = mi(sel)
    # 従来手法：型番スコアが最大の領域を選ぶ（領域ごとに独立して判断）
    best = max(pm, key=lambda e: (e.get('by_key', {}) or {}).get('ref', 0) or 0)
    old_i = mi(best['mask'])
    agree = (sel_i == old_i)

    selrec = next((e for e in pm if e.get('mask') == sel), {})
    sel_ref = (selrec.get('by_key', {}) or {}).get('ref')
    old_ref = (best.get('by_key', {}) or {}).get('ref')

    # 背景を淡くして、選ばれた領域だけを目立たせる
    canvas = (rgb * 0.45 + 255 * 0.55).astype(np.uint8)
    for idx, col in ((old_i, RED), (sel_i, GREEN)):
        if agree:
            col = YELLOW
        tint = canvas.copy()
        tint[masks[idx]] = (canvas[masks[idx]] * 0.35
                            + np.array(col) * 0.65).astype(np.uint8)
        canvas = tint
    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)
    for idx, col in ((old_i, RED), (sel_i, GREEN)):
        if agree:
            col = YELLOW
        for c in mask_outline(masks[idx]):
            pts = [tuple(map(int, p[0])) for p in c]
            if len(pts) > 1:
                d.line(pts + [pts[0]], fill=col, width=6)

    img = img.rotate(180, expand=True)  # OCR用に180度回転して保存されているため戻す

    # ---- 凡例 ----
    LW = 560
    out = Image.new('RGB', (img.width + LW, img.height), 'white')
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    fb = ImageFont.truetype(FONT_B, 26)
    fr = ImageFont.truetype(FONT_R, 21)
    fs = ImageFont.truetype(FONT_R, 18)
    x = img.width + 26
    y = 26
    d.text((x, y), '選ばれた領域の比較', font=fb, fill=(26, 31, 41)); y += 52
    d.text((x, y), f'対象商品：{dbg.get("query", "")}', font=fr, fill=(90, 94, 104)); y += 46

    if agree:
        rows = [(YELLOW, '両手法とも同じ領域', f'型番スコア {old_ref}')]
    else:
        rows = [(RED, '従来手法（型番のみ）', f'{best["mask"]}  型番スコア {old_ref}'),
                (GREEN, '提案手法（多キー・一対一）', f'{sel}  型番スコア {sel_ref}')]
    for col, title, sub in rows:
        d.rectangle([x, y + 4, x + 30, y + 34], fill=col)
        d.text((x + 44, y), title, font=fr, fill=(26, 31, 41))
        d.text((x + 44, y + 30), sub, font=fs, fill=(90, 94, 104))
        y += 76

    y += 10
    d.text((x, y), f'決め手：{dbg.get("winning_key", "")}', font=fr, fill=(26, 31, 41))
    y += 34
    d.text((x, y), f'採用スコア：{dbg.get("selected_score", "")}', font=fs, fill=(90, 94, 104))

    path = os.path.join(shot_dir, OUT_NAME)
    out.save(path)
    return agree


def build_all(result_dir, quiet=False):
    work = os.path.join(result_dir, 'work')
    if not os.path.isdir(work):
        work = result_dir
    if not os.path.isdir(work):
        return 0, 0, 0
    shots = sorted(d for d in os.listdir(work)
                   if os.path.isdir(os.path.join(work, d)) and '_attempt' not in d)
    if not quiet:
        print(f'対象: {work}\nショット数: {len(shots)}')
    done = skipped = agreed = 0
    for s in shots:
        d = os.path.join(work, s)
        if not all(os.path.exists(os.path.join(d, f)) for f in NEEDED):
            skipped += 1
            continue
        try:
            if build(d):
                agreed += 1
            done += 1
        except Exception as e:  # noqa: BLE001
            if not quiet:
                print(f'  × {s}: {type(e).__name__}: {e}')
            skipped += 1
    if not quiet:
        print(f'生成 {done} 件 / スキップ {skipped} 件  '
              f'（うち両手法一致 {agreed} 件・選択が異なる {done - agreed} 件）')
    return done, skipped, agreed


if __name__ == '__main__':
    build_all(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULT)
