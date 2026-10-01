#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_image_match_test.py の実行結果(reco/<dataset>/single_image_match_reco_result_<日時>/
work/<画像名>/multikey_match_debug.json)をもとに、目視レビュー用のExcelと、
各(画像, 品目)ぶんの選択箇所ハイライト画像を集めた画像フォルダを作る(2026-09-29追加)。

single_image_match_test.pyは幅推定は行わず「マスタ全品目 x 全マスクの一括対応付け」だけを
見るツールなので、build_width_eval_report.pyとは別のシート構成にしてある(正解幅・誤差列は無く、
代わりに5キーの内訳と対応付け結果(マスク番号 or 該当なし)を列に持つ)。

出力先(single_image_match_reco_result_<日時>/の直下にまとめる。build_width_eval_report.pyと
同じ流儀):
    <result_dir>/images/<画像名>-<品目名>.png   (work/<画像名>/images/<品目名>.png を集約・改名)
    <result_dir>/single_image_match_report.xlsx

実行:
    python3 reco/scripts/build_single_image_match_report.py \
        --result-dir reco/0911/single_image_match_reco_result_20260929_131144
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

REPO_ROOT = Path(__file__).resolve().parents[2]

HEADERS = [
    "認識した順番", "画像", "query(book_name)", "display_name",
    "目視確認(T/F)", "対応付け結果", "該当なしの理由", "足切りスコア(しきい値未満)",
    "スコア(合計)",
    "REFスコア", "display_nameスコア", "日付スコア", "spec_1スコア", "spec_2スコア",
    "識別確信度(confident)", "認識した文字列",
    "足切りしなければ選ばれたマスク", "そのスコア", "そのマスクの文字列",
    "画像ファイル名", "メモ",
]
COLUMN_WIDTHS = {
    "認識した順番": 12, "画像": 10, "query(book_name)": 18, "display_name": 26,
    "目視確認(T/F)": 14, "対応付け結果": 14, "該当なしの理由": 26,
    "足切りスコア(しきい値未満)": 18,
    "スコア(合計)": 12,
    "REFスコア": 10, "display_nameスコア": 14, "日付スコア": 10,
    "spec_1スコア": 10, "spec_2スコア": 10,
    "識別確信度(confident)": 16, "認識した文字列": 40,
    "足切りしなければ選ばれたマスク": 20, "そのスコア": 10, "そのマスクの文字列": 40,
    "画像ファイル名": 30, "メモ": 24,
}


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in (s or ""))


def absent_reason_text(a: dict) -> str:
    """該当なしになった理由を人が読める文にする(2026-09-30追加、ユーザー要望)。

    ユーザーが最初に懸念した「画像に映っていないもの」と「スコアが低くて足切り
    されたもの」の2種類を区別する。multikey_matcher.py側で付与したreject_reason
    ("threshold"=Hungarian法は割り当てたがしきい値未満で除外/"no_assignment"=
    そもそも一度も割り当てられなかった)と、would_be_score(足切りしなければ
    選ばれていたはずのマスクのスコア)を組み合わせて判定する。

    would_be_score==0は、その撮影の中にその品目を示すOCR文字列が一つも
    検出されていない(手がかり自体が無い)ことを意味する。この場合は
    reject_reasonに関わらず「文字列手がかりなし」とする。
    """
    if a.get("assigned_mask"):
        return ""
    score = a.get("would_be_score")
    if score is None:
        return "マスク無し"
    if score == 0:
        return "文字列手がかりなし"
    if a.get("reject_reason") == "threshold":
        return f"しきい値未満で足切り(score={score})"
    return f"対応マスク不足・競合敗北(score={score})"


def add_summary_sheet(wb: Workbook, data_sheet_name: str) -> None:
    """先頭に「集計」シートを追加し、対応付け率・該当なし率を数式で出す。
    build_width_eval_report.pyのadd_summary_sheetと同じ考え方(列全体を参照する数式なので、
    目視確認(T/F)を後から追記・修正しても開き直せば自動で再計算される)。

    2026-09-29追加(ユーザー要望、原因分析用): 「該当なしとされた認識の成功率」
    「対応ありとされた認識の成功率」を別々に出す。対応付け結果列と目視確認(T/F)列を
    組み合わせたCOUNTIFSを使う。
    """
    if "集計" in wb.sheetnames:
        del wb["集計"]
    ws = wb.create_sheet("集計", 0)

    src = f"'{data_sheet_name}'!"
    total_col = get_column_letter(HEADERS.index("query(book_name)") + 1)
    review_col = get_column_letter(HEADERS.index("目視確認(T/F)") + 1)
    result_col = get_column_letter(HEADERS.index("対応付け結果") + 1)
    total_expr = f"(COUNTA({src}{total_col}:{total_col})-1)"
    # ヘッダー行(対応付け結果列の値="対応付け結果")が"<>該当なし"に紛れてカウントされない
    # よう、対応あり件数は「総件数-該当なし件数」で出す(COUNTIFの"該当なし"完全一致は
    # ヘッダー文字列と一致しないので安全)。
    unmatched_total_expr = f'COUNTIF({src}{result_col}:{result_col},"該当なし")'
    matched_total_expr = f"({total_expr}-{unmatched_total_expr})"

    rows = [
        ("項目", "値", "内訳"),
        ("総件数", f"={total_expr}", ""),
        ("対応付け率(目視確認T)", f'=COUNTIF({src}{review_col}:{review_col},"T")/{total_expr}',
         f'=COUNTIF({src}{review_col}:{review_col},"T")&"/"&{total_expr}'),
        ("誤対応付け率(目視確認F)", f'=COUNTIF({src}{review_col}:{review_col},"F")/{total_expr}',
         f'=COUNTIF({src}{review_col}:{review_col},"F")&"/"&{total_expr}'),
        ("該当なし率(対応付け結果)", f'=COUNTIF({src}{result_col}:{result_col},"該当なし")/{total_expr}',
         f'=COUNTIF({src}{result_col}:{result_col},"該当なし")&"/"&{total_expr}'),
        ("対応ありの正解率(原因分析用)",
         f'=COUNTIFS({src}{result_col}:{result_col},"<>該当なし",{src}{review_col}:{review_col},"T")/{matched_total_expr}',
         f'=COUNTIFS({src}{result_col}:{result_col},"<>該当なし",{src}{review_col}:{review_col},"T")&"/"&{matched_total_expr}'),
        ("該当なしの正解率(原因分析用)",
         f'=COUNTIFS({src}{result_col}:{result_col},"該当なし",{src}{review_col}:{review_col},"T")/{unmatched_total_expr}',
         f'=COUNTIFS({src}{result_col}:{result_col},"該当なし",{src}{review_col}:{review_col},"T")&"/"&{unmatched_total_expr}'),
    ]
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            if r_idx == 1:
                cell.font = Font(bold=True)
    for r in range(3, len(rows) + 1):
        ws.cell(row=r, column=2).number_format = "0.0%"
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14


def _load_existing_review(xlsx_path: Path) -> dict[tuple[str, str], dict]:
    """既存のExcelがあれば、(画像, query)ごとの目視確認(T/F)・メモを読み出す
    (2026-09-29追加、ユーザー要望: 目視確認済みのExcelを再生成しても記入内容が
    消えないようにする)。無ければ空dict。"""
    if not xlsx_path.exists():
        return {}
    try:
        wb = load_workbook(xlsx_path)
        if "対応付け結果" not in wb.sheetnames:
            return {}
        ws = wb["対応付け結果"]
        hdr = [c.value for c in ws[1]]
        i_img, i_q = hdr.index("画像"), hdr.index("query(book_name)")
        i_tf, i_memo = hdr.index("目視確認(T/F)"), hdr.index("メモ")
        out = {}
        for row in ws.iter_rows(min_row=2, values_only=True):
            key = (row[i_img], row[i_q])
            if row[i_tf] or row[i_memo]:
                out[key] = {"tf": row[i_tf] or "", "memo": row[i_memo] or ""}
        return out
    except Exception as e:
        print(f"⚠ 既存Excelの目視確認内容を読み込めませんでした(空のまま再生成します): "
              f"{type(e).__name__}: {e}")
        return {}


def build_report(result_dir: Path) -> None:
    result_dir = Path(result_dir).expanduser().resolve()
    work_root = result_dir / "work"
    if not work_root.is_dir():
        raise FileNotFoundError(f"{work_root} がありません(single_image_match_test.pyの結果フォルダか確認してください)")

    images_dir = result_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = result_dir / "single_image_match_report.xlsx"
    existing_review = _load_existing_review(xlsx_path)
    if existing_review:
        print(f"[build_single_image_match_report] 既存の目視確認 {len(existing_review)}件を引き継ぎます")

    rows = []
    order = 0
    for shot_dir in sorted(work_root.iterdir(), key=lambda p: p.name):
        debug_path = shot_dir / "multikey_match_debug.json"
        if not debug_path.exists():
            continue
        debug = json.loads(debug_path.read_text(encoding="utf-8"))
        # per_mask の text はマスク単位の値(そのマスクに帰属したOCR文字列の連結)で、
        # どの品目を対象に採点したか(target_j)に依存しないので、assigned_maskをキーに
        # そのまま引ける(2026-09-29追加、ユーザー要望: 「読み取った文字列を表示してほしい」)。
        mask_text = {p["mask"]: p["text"] for p in debug.get("per_mask", [])}
        for a in debug["all_assignments"]:
            order += 1
            name = a["book_name"] or a["display_name"] or "item"
            src_img = shot_dir / "images" / f"{safe_name(name)}.png"
            img_filename = ""
            if src_img.exists():
                # <画像番号>-<認識した順番>-<品目名>.png (2026-09-29、ユーザー要望)。
                # 認識した順番(order)はこの関数内で全画像を通して振っている連番。
                img_filename = f"{shot_dir.name}-{order}-{safe_name(name)}.png"
                shutil.copyfile(src_img, images_dir / img_filename)
            else:
                print(f"⚠ 品目ごとの画像が見つかりません: {src_img}")
            rows.append({
                "順番": order,
                "画像": shot_dir.name,
                "query": a["book_name"],
                "display_name": a["display_name"],
                "対応付け結果": a["assigned_mask"] or "該当なし",
                "score": a["score"],
                # 2026-09-29追加: multikey_matcher.py側でall_assignmentsに
                # by_key(5キー内訳)を持たせたので、target_j(query)以外の品目も
                # 内訳を埋められるようになった。
                # 2026-09-30追加(ユーザー要望「合計スコアだけじゃ判断できない」):
                # 該当なしの行はby_keyが全部Noneのままだったので、代わりに
                # would_be_by_key(足切りしなければ選ばれていたはずのマスクの5キー内訳)を使う。
                "by_key": (a.get("by_key") or {}) if a.get("assigned_mask") else (a.get("would_be_by_key") or {}),
                # 2026-09-30追加(ユーザー要望「足切りされたものはスコアが出ているのに文字列が
                # 出ていないのはなぜ」): 該当なしの場合、以前は空文字のままだった。by_key列と
                # 同じ考え方で、対応付け済みならassigned_maskの文字列、該当なしならwould_be_mask
                # (足切りしなければ選ばれていたはずのマスク)の文字列を出す。
                "recognized_text": (
                    mask_text.get(a["assigned_mask"], "") if a["assigned_mask"]
                    else (mask_text.get(a.get("would_be_mask"), "") if a.get("would_be_mask") else "")
                ),
                # 2026-09-30追加(ユーザー要望): 該当なしでも、足切りしなければ選ばれて
                # いたはずのマスクとスコア・その文字列を見えるようにする。
                "would_be_mask": a.get("would_be_mask") or "",
                "would_be_score": a.get("would_be_score"),
                "would_be_text": mask_text.get(a.get("would_be_mask"), "") if a.get("would_be_mask") else "",
                "absent_reason": absent_reason_text(a),
                # 2026-09-30追加(ユーザー要望): 足切りされた場合のスコアを専用の数値列にも出す。
                # 「そのスコア」列は対応マスク不足(競合敗北)のケースにも同じ値が入っていて
                # 区別しづらいので、しきい値未満で足切りされたケースだけに絞った列を分ける。
                "threshold_score": a.get("would_be_score") if a.get("reject_reason") == "threshold" else None,
                "image_filename": img_filename,
            })

    if not rows:
        print(f"⚠ {work_root} の下に multikey_match_debug.json が見つからず、何も出力しませんでした")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "対応付け結果"
    ws.append(HEADERS)
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    for col_idx in range(1, len(HEADERS) + 1):
        c = ws.cell(row=1, column=col_idx)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"

    unmatched_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    for r in rows:
        bk = r["by_key"]
        prev = existing_review.get((r["画像"], r["query"]), {})
        ws.append([
            r["順番"], r["画像"], r["query"], r["display_name"],
            prev.get("tf", ""), r["対応付け結果"], r["absent_reason"], r["threshold_score"],
            r["score"],
            bk.get("ref"), bk.get("display_name"), bk.get("date"), bk.get("spec_1"), bk.get("spec_2"),
            "", r["recognized_text"],
            r["would_be_mask"], r["would_be_score"], r["would_be_text"],
            r["image_filename"], prev.get("memo", ""),
        ])
        row_idx = ws.max_row
        if r["対応付け結果"] == "該当なし":
            for col in range(1, len(HEADERS) + 1):
                ws.cell(row=row_idx, column=col).fill = unmatched_fill

    for col_idx, header in enumerate(HEADERS, start=1):
        ws.column_dimensions[get_column_letter(col_idx)].width = COLUMN_WIDTHS.get(header, 14)

    # 目視確認(T/F)列をT/Fの二択プルダウンにする(2026-09-29追加、ユーザー要望。
    # build_width_eval_report.pyと同じ仕組み)。
    n_rows = ws.max_row
    review_col = HEADERS.index("目視確認(T/F)") + 1
    review_letter = get_column_letter(review_col)
    dv = DataValidation(type="list", formula1='"T,F"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{review_letter}2:{review_letter}{n_rows}")

    add_summary_sheet(wb, ws.title)

    wb.save(xlsx_path)
    print(f"✔ Excel -> {xlsx_path}")
    print(f"✔ 画像  -> {images_dir} ({len(list(images_dir.glob('*.png')))}枚)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result-dir", required=True,
                    help="single_image_match_test.py --dataset で作られた reco_result フォルダ")
    args = ap.parse_args()
    build_report(Path(args.result_dir))


if __name__ == "__main__":
    main()
