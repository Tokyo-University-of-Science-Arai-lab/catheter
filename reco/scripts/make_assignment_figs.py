"""工程4（全商品とマスクの対応付け）の実験画像を作る。

final.png と同じ見た目（正立・把持点なし）を土台に、
  ・目標商品に割り当てられた領域 … 濃い緑
  ・他の商品に割り当てられた領域 … かなり薄い色（商品ごとに別の色）
  ・どの商品にも割り当てられなかった領域 … 塗らない
として重ねる。「目標だけでなく全商品を同時に認識・割当している」ことを1枚で示す。

final.png には把持点（赤い点）が焼き込まれているため、それを土台にせず
after_init_rgb.png から作り直している（把持点なしの指定のため）。
after_init_rgb.png は OCR 用に180度回転して保存されているので、
final.png と同じ向きへ戻してから出力する。

出力: 各 work フォルダの `assignment_overlay.png`

使い方:
    python3 make_assignment_figs.py                       # 既定の結果フォルダ・既定ショット
    python3 make_assignment_figs.py <結果フォルダ> [ショット名 ...]
    python3 make_assignment_figs.py <結果フォルダ> --all   # 全ショット
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

RECO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULT = os.path.join(RECO, 'ROOB-1', 'reco_result_20260828_201346')
DEFAULT_SHOTS = ['7-1-Wallaby_Avenir_Coil_System-10CSW01515']
OUT_NAME = 'assignment_overlay.png'

TARGET_COLOR = (30, 220, 90)   # 目標商品：濃い緑
TARGET_ALPHA = 0.62            # しっかり塗る
OTHER_ALPHA = 0.20             # 他商品：かなり薄く

# 他商品用の色。隣り合う箱で似た色にならないよう離して並べる
OTHER_PALETTE = [
    (230, 60, 50), (40, 120, 230), (245, 165, 20), (160, 70, 200),
    (0, 175, 185), (235, 100, 165), (120, 165, 45), (205, 110, 55),
    (75, 95, 220), (215, 60, 120), (95, 145, 235), (180, 140, 30),
]

NEEDED = ('after_init_rgb.png', 'sam3_service_masks.npz', 'multikey_match_debug.json')


def paint(canvas, mask, color, alpha):
    canvas[mask] = (canvas[mask] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)


def build(shot_dir):
    rgb = cv2.cvtColor(cv2.imread(os.path.join(shot_dir, 'after_init_rgb.png')),
                       cv2.COLOR_BGR2RGB)
    masks = np.load(os.path.join(shot_dir, 'sam3_service_masks.npz'))['masks']
    dbg = json.load(open(os.path.join(shot_dir, 'multikey_match_debug.json')))
    sel = dbg.get('selected_mask')
    aa = dbg.get('all_assignments', [])

    def mi(name):
        return int(str(name).split('_')[1]) - 1

    canvas = rgb.copy()

    # 他商品を先に薄く塗る（目標商品が上に来るように）
    others = 0
    for e in aa:
        am = e.get('assigned_mask')
        if not am or am == sel or e.get('is_query'):
            continue
        i = mi(am)
        if i >= len(masks):
            continue
        paint(canvas, masks[i], OTHER_PALETTE[others % len(OTHER_PALETTE)], OTHER_ALPHA)
        others += 1

    # 目標商品を濃く塗る
    if sel is not None and mi(sel) < len(masks):
        paint(canvas, masks[mi(sel)], TARGET_COLOR, TARGET_ALPHA)

    # final.png と同じ向き（正立）へ戻す。把持点は描かない
    out = Image.fromarray(canvas).rotate(180, expand=True)
    path = os.path.join(shot_dir, OUT_NAME)
    out.save(path)
    return sel, others, dbg.get('query')


def main():
    args = [a for a in sys.argv[1:]]
    result_dir = args[0] if args and not args[0].startswith('--') else DEFAULT_RESULT
    rest = args[1:] if args and not args[0].startswith('--') else args
    work = os.path.join(result_dir, 'work')
    if not os.path.isdir(work):
        work = result_dir

    if '--all' in rest:
        shots = sorted(d for d in os.listdir(work)
                       if os.path.isdir(os.path.join(work, d)) and '_attempt' not in d)
    else:
        shots = [s for s in rest if not s.startswith('--')] or DEFAULT_SHOTS

    for s in shots:
        d = os.path.join(work, s)
        if not all(os.path.exists(os.path.join(d, f)) for f in NEEDED):
            print(f'  × {s}: 素材が揃っていません')
            continue
        sel, others, q = build(d)
        print(f'saved {s}/{OUT_NAME}  query={q} 目標={sel} 他商品={others}件')


if __name__ == '__main__':
    main()
