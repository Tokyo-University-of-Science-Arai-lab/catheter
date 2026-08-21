#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
width_mm_validation.py の実行結果(reco/<dataset>/width_eval_result.csv)をもとに、
データセットごとに目視レビュー用のExcelと、各件のfinal.png(選択された箱をハイライトした
画像)を集めた画像フォルダを作る。

出力先(データセットごとに分離、2026-08-21以降):
  reco/<dataset>/width_eval_report/images/*.png
  reco/<dataset>/width_eval_report/catheter_width_report.xlsx

実行:
    cd ~/pro_book/pro_hand_book_python
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/build_width_eval_report.py
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

RECO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["stand-100", "diagonal-40"]

HEADERS = [
    "画像ファイル", "shot", "query(book_name)", "display_name",
    "正解幅mm", "推定幅mm", "誤差mm", "2mm以内(把持成功目安)",
    "識別スコア", "識別margin", "識別確信度(confident)", "認識した文字列",
    "処理時間sec", "目視確認(T/F)", "メモ", "エラー",
]
COLUMN_WIDTHS = {
    "画像ファイル": 42, "shot": 20, "query(book_name)": 16,
    "display_name": 26, "正解幅mm": 10, "推定幅mm": 10, "誤差mm": 10,
    "2mm以内(把持成功目安)": 18, "識別スコア": 12, "識別margin": 12,
    "識別確信度(confident)": 16, "認識した文字列": 40, "処理時間sec": 12,
    "目視確認(T/F)": 14, "メモ": 24, "エラー": 20,
}


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def image_filename_for(dataset: str, shot: str, display_name: str) -> str:
    """データセットごとの分かりやすい画像ファイル名を作る。

    diagonal-40のshot名は元々品目名そのもの(例: Target_R)で読みやすいが、
    stand-100のshot名は"<画像番号>__<book_nameコード>"(例: 1__ESC0305)で
    コードが読みにくいため、display_nameベースの名前に置き換える。
    """
    if dataset == "stand-100":
        image_id = shot.split("__", 1)[0]
        return f"{image_id}_{safe_name(display_name or shot)}.png"
    return f"{safe_name(shot)}.png"


def load_multikey_debug(work_dir: Path) -> dict:
    p = work_dir / "multikey_match_debug.json"
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


def build_dataset_report(dataset: str) -> None:
    csv_path = RECO_ROOT / dataset / "width_eval_result.csv"
    if not csv_path.exists():
        print(f"⚠ 見つかりません、スキップ: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["shot"])

    out_dir = RECO_ROOT / dataset / "width_eval_report"
    images_dir = out_dir / "images"
    xlsx_path = out_dir / "catheter_width_report.xlsx"
    images_dir.mkdir(parents=True, exist_ok=True)

    for r in rows:
        shot = r["shot"]
        work_dir = RECO_ROOT / dataset / "width_eval_work" / shot
        debug = load_multikey_debug(work_dir)
        r["_selected_score"] = debug.get("selected_score")
        r["_margin"] = debug.get("margin")
        r["_confident"] = debug.get("confident")
        r["_recognized_text"] = recognized_text_for_selected_mask(debug)

        img_filename = image_filename_for(dataset, shot, r.get("display_name", ""))
        src = work_dir / "final.png"
        r["_image_filename"] = ""
        if src.exists():
            shutil.copyfile(src, images_dir / img_filename)
            r["_image_filename"] = img_filename
        else:
            print(f"⚠ final.pngが見つかりません: {src}")

    wb = Workbook()
    ws = wb.active
    ws.title = safe_name(dataset)[:31]

    ws.append(HEADERS)
    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    for col_idx in range(1, len(HEADERS) + 1):
        c = ws.cell(row=1, column=col_idx)
        c.font = header_font
        c.fill = header_fill
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"

    for r in rows:
        true_mm = to_float(r.get("book_width_mm_true"))
        pred_mm = to_float(r.get("book_width_mm_pred"))
        err_mm = to_float(r.get("abs_error_mm"))
        ws.append([
            r["_image_filename"],
            r["shot"],
            r.get("query", ""),
            r.get("display_name", ""),
            true_mm,
            pred_mm,
            err_mm,
            ("○" if err_mm is not None and err_mm <= 2.0 else ("" if err_mm is None else "×")),
            to_float(r["_selected_score"]),
            to_float(r["_margin"]),
            ("" if r["_confident"] is None else str(r["_confident"])),
            r["_recognized_text"],
            to_float(r.get("elapsed_sec")),
            "",
            "",
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

    wb.save(xlsx_path)
    print(f"✔ [{dataset}] Excel -> {xlsx_path} ({len(rows)}行)")
    print(f"✔ [{dataset}] 画像 -> {images_dir} ({len(list(images_dir.glob('*.png')))}枚)")


def main() -> None:
    for dataset in DATASETS:
        build_dataset_report(dataset)


if __name__ == "__main__":
    main()
