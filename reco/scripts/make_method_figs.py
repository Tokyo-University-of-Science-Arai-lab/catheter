"""従来手法と提案手法の照合のしかたを、対になる2枚の図で可視化する。

  legacy_overlay.png   従来手法：全ての領域を目標商品ただ1つと比較する
                       → 凡例の商品名が全行で同一になる
  proposed_overlay.png 提案手法：台帳の全商品を領域へ一対一で割り当てる
                       → 凡例に異なる商品名が並ぶ

既存の `ocr_overlay_all_assignments.png` は矩形（bounding box）で領域を
示すため隣接する箱と重なって分かりにくい。本スクリプトはマスクの輪郭を
そのまま描くことで、細長い箱でも領域の境界が判別できるようにしている。

スコアは、従来手法が per_mask の by_key['ref']（型番との一致度）、
提案手法が all_assignments の score（決め手キーの一致度）。

使い方:
    python3 make_method_figs.py                  # 既定の結果フォルダ
    python3 make_method_figs.py <結果フォルダ>
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RECO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULT = os.path.join(RECO, 'ROOB-1', 'reco_result_20260828_213247')
OUT_LEGACY = 'legacy_overlay.png'
OUT_PROPOSED = 'proposed_overlay.png'

FONT_B = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_M = '/usr/share/fonts/DejaVuSansMono.ttf'
FONT_M_ALT = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

PALETTE = [
    (230, 60, 50), (40, 120, 230), (30, 170, 100), (245, 165, 20),
    (160, 70, 200), (0, 175, 185), (235, 100, 165), (120, 165, 45),
    (205, 110, 55), (75, 95, 220), (45, 190, 145), (215, 60, 120),
    (95, 145, 235), (180, 140, 30), (110, 60, 190), (0, 150, 160),
]

NEEDED = ('after_init_rgb.png', 'sam3_service_masks.npz', 'multikey_match_debug.json')


def mono(size):
    for p in (FONT_M, FONT_M_ALT):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.truetype(FONT_B, size)


def _render(rgb, masks, rows, title, out_path, top_n, draw_only_emph=False):
    """rows: [(mask_index, ラベル文字列, 強調するか), ...] を描画して保存する。"""
    img = Image.fromarray(rgb).convert('RGB')
    d = ImageDraw.Draw(img)
    for i, _, emph in rows[:top_n]:
        if i >= len(masks):
            continue
        if draw_only_emph and not emph:
            continue
        col = PALETTE[i % len(PALETTE)]
        m = (masks[i].astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            pts = [tuple(map(int, p[0])) for p in c]
            if len(pts) > 1:
                d.line(pts + [pts[0]], fill=col, width=10 if emph else 4)
        ys, xs = np.where(masks[i])
        if len(xs):
            bx, by = int(xs.min()), int(ys.min())
            r = 22
            d.ellipse([bx, by, bx + r * 2, by + r * 2], fill=col)
            d.text((bx + r, by + r), str(i + 1), font=mono(24),
                   fill=(255, 255, 255), anchor='mm')
    img = img.rotate(180, expand=True)

    LW = 700
    out = Image.new('RGB', (img.width + LW,
                            max(img.height, 60 + 44 * (min(len(rows), top_n) + 3))),
                    (0, 0, 0))
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)
    x, y = img.width + 26, 22
    d.text((x, y), title, font=mono(28), fill=(255, 235, 0))
    y += 50
    for i, label, emph in rows[:top_n]:
        col = PALETTE[i % len(PALETTE)]
        d.ellipse([x, y + 2, x + 30, y + 32], fill=col)
        d.text((x + 15, y + 17), str(i + 1), font=mono(20),
               fill=(255, 255, 255), anchor='mm')
        d.text((x + 44, y + 6), label, font=mono(20), fill=(255, 255, 255))
        y += 44
    out.save(out_path)


def build(shot_dir, top_n=14):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(shot_dir, 'after_init_rgb.png')),
                       cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    masks = np.load(os.path.join(shot_dir, 'sam3_service_masks.npz'))['masks']
    dbg = json.load(open(os.path.join(shot_dir, 'multikey_match_debug.json')))
    pm = dbg.get('per_mask', [])
    query = dbg.get('query', '')
    if not pm:
        raise ValueError('per_mask がありません')

    def ref_of(e):
        return float((e.get('by_key', {}) or {}).get('ref') or 0.0)

    ranked = sorted(pm, key=ref_of, reverse=True)
    chosen = ranked[0]['mask']          # 従来手法が選ぶ領域（型番スコア最大）

    def mi(name):
        return int(str(name).split('_')[1]) - 1

    # ---- 従来手法：全領域を目標商品1つと比較し、最尤の1領域だけを選ぶ ----
    # 画像に描くのは最終的に選ばれた領域のみ（2026-08-30、ユーザー要望）。
    # 凡例は上位を並べて「比較相手が目標商品ただ1つ」であることを示す。
    legacy_rows = []
    for e in ranked:
        legacy_rows.append((
            mi(e['mask']),
            f"{e['mask']}: {query}  score={ref_of(e):.1f}"
            + ('  [SELECTED]' if e['mask'] == chosen else ''),
            e['mask'] == chosen,
        ))
    _render(rgb, masks, legacy_rows, 'mask --> compared item',
            os.path.join(shot_dir, OUT_LEGACY), top_n, draw_only_emph=True)

    # ---- 提案手法：台帳の全商品を一対一で割り当て ----
    aa = dbg.get('all_assignments', [])
    prop_rows = []
    for e in sorted(aa, key=lambda r: not r.get('is_query')):
        am = e.get('assigned_mask')
        if not am:
            continue
        prop_rows.append((
            mi(am),
            f"{am}: {e.get('book_name','')}  score={float(e.get('score',0)):.1f}"
            + ('  [QUERY]' if e.get('is_query') else ''),
            bool(e.get('is_query')),
        ))
    if prop_rows:
        _render(rgb, masks, prop_rows, 'mask --> assigned item',
                os.path.join(shot_dir, OUT_PROPOSED), max(top_n, len(prop_rows)))
    return chosen


def build_all(result_dir, quiet=False):
    work = os.path.join(result_dir, 'work')
    if not os.path.isdir(work):
        work = result_dir
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
            build(d)
            done += 1
        except Exception as e:  # noqa: BLE001
            if not quiet:
                print(f'  × {s}: {type(e).__name__}: {e}')
            skipped += 1
    if not quiet:
        print(f'生成 {done} 件 / スキップ {skipped} 件  （各フォルダに {OUT_LEGACY} と {OUT_PROPOSED}）')
    return done, skipped


if __name__ == '__main__':
    build_all(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULT)
