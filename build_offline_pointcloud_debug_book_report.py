#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_pointcloud_debug_SAM3_book.py の実行結果(<run_dir>/results.json)から、
目視レビュー用のExcelと、各件のfinal.pngを集めた画像フォルダを <run_dir> 直下に作る。
build_width_eval_report.py(reco/*用)と同じ列構成だが、captures/100testにはGT
ポリゴンアノテーションが無いためIoU一致度は含めない。

実行:
    .pro_hand_book_fixed/bin/python3.10 build_offline_pointcloud_debug_book_report.py \
        captures/100test_offline/20260822_171727
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import shutil
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

HEADERS = [
    "認識した順番", "画像ファイル", "shot", "book_name(query)", "display_name",
    "目視確認(T/F)", "正解幅mm", "推定幅mm", "誤差mm", "2mm以内(把持成功目安)",
    "識別スコア", "識別margin", "識別確信度(confident)", "認識した文字列",
    "処理時間sec", "status", "エラー",
]
COLUMN_WIDTHS = {
    "認識した順番": 12, "画像ファイル": 42, "shot": 8, "book_name(query)": 24,
    "display_name": 20, "目視確認(T/F)": 14, "正解幅mm": 10, "推定幅mm": 10,
    "誤差mm": 10, "2mm以内(把持成功目安)": 18, "識別スコア": 12, "識別margin": 12,
    "識別確信度(confident)": 16, "認識した文字列": 40, "処理時間sec": 12,
    "status": 10, "エラー": 20,
}


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_multikey_debug(case_dir: Path) -> dict:
    p = case_dir / "multikey_match_debug.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def recognized_text_for_selected_mask(debug: dict) -> str:
    selected_mask = debug.get("selected_mask")
    for m in debug.get("per_mask", []):
        if m.get("mask") == selected_mask:
            return m.get("text", "")
    return ""


def build_report(run_dir: Path) -> None:
    results_path = run_dir / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    rows.sort(key=lambda r: r["test_index"])

    images_dir = run_dir / "images"
    images_dir.mkdir(exist_ok=True)
    xlsx_path = run_dir / f"{run_dir.name}_report.xlsx"

    wb = Workbook()
    summary_ws = wb.active
    summary_ws.title = "summary"
    ws = wb.create_sheet("results")
    ws.append(HEADERS)
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    for col_idx in range(1, len(HEADERS) + 1):
        c = ws.cell(row=1, column=col_idx)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"

    n_images = 0
    abs_errors = []
    for r in rows:
        case_dir = run_dir / str(r["test_index"])
        debug = load_multikey_debug(case_dir)
        display_name = debug.get("master_row", {}).get("display_name", "")

        img_filename = f"{r['test_index']}-{r['repeat_index']}-{safe_name(display_name or r['book_name'])}.png"
        src = case_dir / "final.png"
        if src.exists():
            shutil.copyfile(src, images_dir / img_filename)
            n_images += 1
        else:
            img_filename = ""

        err_mm = to_float(r.get("abs_error_mm"))
        if err_mm is not None:
            abs_errors.append(err_mm)
        ws.append([
            r["test_index"],
            img_filename,
            r["repeat_index"],
            r["book_name"],
            display_name,
            "",
            to_float(r.get("gt_book_width_mm")),
            to_float(r.get("pred_book_width_mm")),
            err_mm,
            ("○" if err_mm is not None and err_mm <= 2.0 else ("" if err_mm is None else "×")),
            to_float(debug.get("selected_score")),
            to_float(debug.get("margin")),
            ("" if debug.get("confident") is None else str(debug.get("confident"))),
            recognized_text_for_selected_mask(debug),
            to_float(r.get("elapsed_sec")),
            r.get("status", ""),
            r.get("error", ""),
        ])

    n_rows = len(rows) + 1
    ws.auto_filter.ref = f"A1:{get_column_letter(len(HEADERS))}{n_rows}"
    for idx, h in enumerate(HEADERS, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = COLUMN_WIDTHS.get(h, 14)

    review_col = HEADERS.index("目視確認(T/F)") + 1
    review_letter = get_column_letter(review_col)
    dv = DataValidation(type="list", formula1='"T,F"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{review_letter}2:{review_letter}{n_rows}")

    err_col = HEADERS.index("誤差mm") + 1
    for row_idx in range(2, n_rows + 1):
        err_cell = ws.cell(row=row_idx, column=err_col)
        if isinstance(err_cell.value, (int, float)):
            if err_cell.value <= 2.0:
                err_cell.fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
            elif err_cell.value >= 10.0:
                err_cell.fill = PatternFill(start_color="FCE4E4", end_color="FCE4E4", fill_type="solid")

    total = len(rows)
    within_2mm = sum(1 for e in abs_errors if e <= 2.0)
    within_5mm = sum(1 for e in abs_errors if e <= 5.0)
    summary_ws.append([run_dir.name, None])
    summary_ws.append([None, None])
    summary_ws.append(["件数", total])
    summary_ws.append(["誤差2mm未満率", within_2mm / total if total else None])
    summary_ws.append(["誤差5mm未満率", within_5mm / total if total else None])
    summary_ws.append(["IoU平均", "N/A(captures/100testにGTポリゴン注釈が無いため算出不可)"])
    summary_ws.append(["IoU中央値", "N/A(captures/100testにGTポリゴン注釈が無いため算出不可)"])
    summary_ws.column_dimensions["A"].width = 20
    summary_ws.column_dimensions["B"].width = 50

    wb.save(xlsx_path)
    print(f"✔ Excel -> {xlsx_path} ({len(rows)}行)")
    print(f"✔ 画像 -> {images_dir} ({n_images}枚)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="offline_pointcloud_debug_SAM3_book.pyの実行結果フォルダ")
    args = ap.parse_args()
    build_report(args.run_dir.resolve())


if __name__ == "__main__":
    main()
