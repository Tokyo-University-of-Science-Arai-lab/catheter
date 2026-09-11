"""目視確認が F（誤認識）の行を除外した結果表を作る。

誤った箱を選んでいる行では IoU・幅誤差が評価として意味を持たないため、
それらを除いた集計を出す。stand-100 と diagonal-40 の両方を処理する。

出力: <データセット>/identification_filtered_20260823.xlsx
"""
import glob
import os
import statistics

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_NAME = 'identification_filtered_20260823.xlsx'

# データセット名 -> [(結果フォルダ, 短縮名, 説明), ...]  新識別を先に置く
DATASETS = {
    'stand-100': [
        ('reco_result_20260822_124535（最新）', '新識別', '簡易パイプライン + 新識別'),
        ('reco_result_20260822_122603（簡易、過去）', '旧識別', '簡易パイプライン + 旧識別'),
    ],
    'diagonal-40': [
        ('reco_result_20260822_123825（最新）', '新識別', '簡易パイプライン + 新識別'),
        ('reco_result_20260822_122603（過去）', '旧識別', '簡易パイプライン + 旧識別'),
    ],
}

DARK = 'FF333F50'
LIGHT = 'FFEEF0F3'
HDR_FONT = Font(color='FFFFFFFF', bold=True, size=11)


def pick_data_sheet(wb):
    """評価データの入ったシートを選ぶ。

    reco_result_*.xlsx には後から `summary` シートが手で追加されることがあり、
    そちらがアクティブシートになっていると `.active` では列名が取れない
    （2026-08-28に stand-100 で発生）。ヘッダーに「目視確認(T/F)」を持つ
    シートを探すことで、シートが増えても正しいものを拾う。
    """
    for ws in wb.worksheets:
        if '目視確認(T/F)' in [c.value for c in ws[1]]:
            return ws
    return wb.worksheets[0]


def load(dataset, folder):
    xs = glob.glob(os.path.join(HERE, dataset, folder, '*.xlsx'))
    if not xs:
        raise FileNotFoundError(f'{dataset}/{folder}')
    ws = pick_data_sheet(openpyxl.load_workbook(xs[0]))
    hdr = [c.value for c in ws[1]]
    rows = []
    for r in range(2, ws.max_row + 1):
        rows.append({h: ws.cell(r, i + 1).value for i, h in enumerate(hdr)})
    return hdr, rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def is_f(row):
    return str(row['目視確認(T/F)']).strip().upper() == 'F'


def stats(rows):
    iou = [num(r['IoU一致度']) for r in rows]
    iou = [v for v in iou if v is not None]
    err = [num(r['誤差mm']) for r in rows]
    err = [abs(v) for v in err if v is not None]
    t = [num(r['処理時間sec']) for r in rows]
    t = [v for v in t if v is not None]
    d = {'n': len(rows)}
    if iou:
        d.update(iou_mean=statistics.mean(iou), iou_med=statistics.median(iou),
                 iou_min=min(iou), iou_ge50=sum(v >= 0.5 for v in iou))
    if err:
        n = len(err)
        d.update(mae=statistics.mean(err), err_med=statistics.median(err), err_max=max(err),
                 le2=sum(v <= 2 for v in err), le2p=sum(v <= 2 for v in err) / n * 100,
                 le5p=sum(v <= 5 for v in err) / n * 100,
                 le10p=sum(v <= 10 for v in err) / n * 100)
    if t:
        d['time'] = statistics.mean(t)
    return d


def write_sheet(wb, title, hdr, rows):
    ws = wb.create_sheet(title)
    for i, h in enumerate(hdr, 1):
        c = ws.cell(1, i, h)
        c.fill = PatternFill('solid', fgColor=DARK)
        c.font = HDR_FONT
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    for r, row in enumerate(rows, 2):
        for i, h in enumerate(hdr, 1):
            ws.cell(r, i, row[h])
    widths = {'画像ファイル': 30, 'shot': 22, 'query(book_name)': 18,
              'display_name': 26, '認識した文字列': 40, 'メモ': 18, 'エラー': 14}
    for i, h in enumerate(hdr, 1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(h, 13)
    ws.freeze_panes = 'A2'
    ws.auto_filter.ref = f'A1:{get_column_letter(len(hdr))}{len(rows) + 1}'


def build(dataset, spec):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    kept, allrows = {}, {}
    for folder, short, _ in spec:
        hdr, rows = load(dataset, folder)
        keep = [r for r in rows if not is_f(r)]
        kept[short], allrows[short] = keep, rows
        write_sheet(wb, f'{short}_F除外_{len(keep)}件', hdr, keep)

    ws = wb.create_sheet('サマリ', 0)
    ws.column_dimensions['A'].width = 34
    for col in 'BCDE':
        ws.column_dimensions[col].width = 17

    def put(r, vals, bold=False, fill=None):
        for i, v in enumerate(vals, 1):
            c = ws.cell(r, i, v)
            if bold:
                c.font = Font(bold=True)
            if fill:
                c.fill = PatternFill('solid', fgColor=fill)
            if i > 1:
                c.alignment = Alignment(horizontal='center')

    ws.cell(1, 1, f'{dataset}：目視確認 F（誤認識）を除外した集計').font = Font(bold=True, size=14)
    ws.cell(2, 1, '誤った箱を選んでいる行では IoU・幅誤差が評価として意味を持たないため除外した').font = \
        Font(size=10, color='FF5A5E68')

    put(4, ['', '新識別 除外前', '新識別 F除外', '旧識別 除外前', '旧識別 F除外'],
        bold=True, fill=LIGHT)
    S = [stats(allrows['新識別']), stats(kept['新識別']),
         stats(allrows['旧識別']), stats(kept['旧識別'])]

    rows_def = [
        ('件数', 'n', '{:d}'), ('', None, None),
        ('IoU 平均', 'iou_mean', '{:.4f}'), ('IoU 中央値', 'iou_med', '{:.4f}'),
        ('IoU 最小', 'iou_min', '{:.4f}'), ('IoU >= 0.5 の件数', 'iou_ge50', '{:d}'),
        ('', None, None),
        ('幅誤差 MAE [mm]', 'mae', '{:.2f}'), ('幅誤差 中央値 [mm]', 'err_med', '{:.2f}'),
        ('幅誤差 最大 [mm]', 'err_max', '{:.2f}'),
        ('2 mm 以内 [件]', 'le2', '{:d}'), ('2 mm 以内 [%]', 'le2p', '{:.1f}'),
        ('5 mm 以内 [%]', 'le5p', '{:.1f}'), ('10 mm 以内 [%]', 'le10p', '{:.1f}'),
        ('', None, None), ('処理時間 平均 [s]', 'time', '{:.2f}'),
    ]
    r = 5
    for label, key, fmt in rows_def:
        if key is None:
            r += 1
            continue
        put(r, [label] + [fmt.format(s[key]) if key in s else '-' for s in S],
            bold=(key in ('le2p', 'mae')))
        r += 1

    r += 1
    ws.cell(r, 1, '注意：母集団が異なる').font = Font(bold=True, color='FFEB6834')
    r += 1
    n_new, n_old = len(kept['新識別']), len(kept['旧識別'])
    for line in [f'新識別の F除外後は {n_new} 件，旧識別は {n_old} 件で，'
                 '除外された行が異なるため両者を直接比較できない．',
                 '公平に比べるには「両方で T だった行」を見る（下段）．']:
        ws.cell(r, 1, line).font = Font(size=10, color='FF5A5E68')
        r += 1

    key_of = lambda row: (row['shot'], row['query(book_name)'])
    nmap = {key_of(x): x for x in allrows['新識別']}
    omap = {key_of(x): x for x in allrows['旧識別']}
    common = [k for k in omap if not is_f(omap[k]) and k in nmap and not is_f(nmap[k])]

    same = 0
    for k in common:
        a, b = num(omap[k]['誤差mm']), num(nmap[k]['誤差mm'])
        if a is not None and b is not None and abs(a - b) < 1e-9:
            same += 1

    r += 1
    ws.cell(r, 1, f'両方で T だった {len(common)} 件での比較').font = Font(bold=True, size=12)
    r += 1
    cn, co = stats([nmap[k] for k in common]), stats([omap[k] for k in common])
    put(r, ['', '旧識別', '新識別'], bold=True, fill=LIGHT)
    r += 1
    for label, key, fmt in [('件数', 'n', '{:d}'), ('IoU 平均', 'iou_mean', '{:.4f}'),
                            ('幅誤差 MAE [mm]', 'mae', '{:.3f}'),
                            ('2 mm 以内 [件]', 'le2', '{:d}')]:
        put(r, [label, fmt.format(co[key]) if key in co else '-',
                fmt.format(cn[key]) if key in cn else '-'])
        r += 1
    r += 1
    ok = (same == len(common))
    ws.cell(r, 1, f'幅誤差が完全に一致した行：{same} / {len(common)} 件').font = \
        Font(bold=True, color='FF1B8A5A' if ok else 'FFEB6834')
    r += 1
    for line in ['この行はパイプラインが同一で，選ばれるマスクも変わらないため結果が一致する．',
                 '差が出るのは，選ばれるマスクが変わった行だけである．']:
        ws.cell(r, 1, line).font = Font(size=10, color='FF5A5E68')
        r += 1

    # ---- 識別結果の変化の内訳 ----
    fixed = [k for k in omap if is_f(omap[k]) and k in nmap and not is_f(nmap[k])]
    broke = [k for k in omap if not is_f(omap[k]) and k in nmap and is_f(nmap[k])]
    still = [k for k in omap if is_f(omap[k]) and k in nmap and is_f(nmap[k])]

    r += 1
    ws.cell(r, 1, '識別結果の変化（旧識別 → 新識別）').font = Font(bold=True, size=12)
    r += 1
    for label, ks, color in [('改善（旧 F → 新 T）', fixed, 'FF1B8A5A'),
                             ('悪化（旧 T → 新 F）', broke, 'FFC0392B'),
                             ('依然として誤り', still, 'FF5A5E68')]:
        ws.cell(r, 1, label).font = Font(bold=True, color=color)
        c = ws.cell(r, 2, f'{len(ks)} 件')
        c.font = Font(bold=True, color=color)
        c.alignment = Alignment(horizontal='center')
        r += 1
        names = {}
        for k in ks:
            nm = omap[k]['display_name']
            names[nm] = names.get(nm, 0) + 1
        for nm, cnt in sorted(names.items(), key=lambda x: -x[1]):
            ws.cell(r, 1, f'　　{nm}').font = Font(size=10, color='FF5A5E68')
            cc = ws.cell(r, 2, f'{cnt} 件')
            cc.font = Font(size=10, color='FF5A5E68')
            cc.alignment = Alignment(horizontal='center')
            r += 1

    out = os.path.join(HERE, dataset, OUT_NAME)
    wb.save(out)
    return out, kept, common, same


if __name__ == '__main__':
    for dataset, spec in DATASETS.items():
        out, kept, common, same = build(dataset, spec)
        print(f'[{dataset}] saved {os.path.relpath(out, HERE)}')
        print(f'    新識別 F除外 {len(kept["新識別"])} 件 / 旧識別 F除外 {len(kept["旧識別"])} 件'
              f' / 共通T {len(common)} 件・幅誤差一致 {same} 件')
