#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
width_mm_validation.py の実行結果(reco/<dataset>/width_eval_result.csv)をもとに、
データセットごとに目視レビュー用のExcelと、各件のfinal.png(選択された箱をハイライトした
画像)を集めた画像フォルダを作る。

出力先(データセットごとに分離、実行のsuffixごとに新規フォルダ。2026-08-21、ユーザー要望:
Excel/画像が毎回上書きされず区別できるようにするため):
  reco/<dataset>/reco_result<suffix>/images/*.png
  reco/<dataset>/reco_result<suffix>/catheter_width_report<suffix>.xlsx
  reco/<dataset>/reco_result<suffix>/work/<各認識ケース>/...

Excelと画像フォルダは同じreco_result<suffix>/の直下にまとめる(2026-08-21、
ユーザー要望: 「Excelと画像が別フォルダに分かれているのはNG、1つのフォルダに
まとめてほしい」)。

【2026-08-25変更】以前はwidth_mm_validation.pyの作業フォルダwidth_eval_work<suffix>/
とこのレポート出力先reco_result_<別のタイムスタンプ>/が別物で、reco_result側の"work"は
width_eval_work<suffix>/へのシンボリックリンクだった。ユーザー要望(「width_eval_work
フォルダは不要、reco_result内に入れてほしい」)により、width_mm_validation.py側が
最初からreco_result<suffix>/work/を直接の作業フォルダとして使うように変更したため、
このスクリプトはもう別フォルダのシンボリックリンクを作らない(work/は最初からそこにある)。
suffixもwidth_mm_validation.py実行時の1つのタイムスタンプに統一した(以前はCSV/work用の
suffixと、レポート生成時刻report_timestampの2つのタイムスタンプが混在していた)。

実行:
    python3 reco/scripts/build_width_eval_report.py
    python3 reco/scripts/build_width_eval_report.py --suffix _20260825_195856
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment, PatternFill

RECO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["stand-100", "diagonal-40"]

# データセットごとのCVAT/COCO形式アノテーション(GTポリゴン、IoU一致度の算出に使う)。
ANNOTATIONS_JSON = {
    "stand-100": RECO_ROOT / "stand-100" / "annotations" / "instances_default_100.json",
    "diagonal-40": RECO_ROOT / "diagonal-40" / "annotations" / "instances_default.json",
}

# パイプラインごとに最終選択マスクの保存ファイル名が異なる(2026-08-22判明:
# simplifiedパイプラインはfinal_mask.pngを作らずselected_mask_refined.pngを使う)。
# 存在する方を優先順に試す。
SELECTED_MASK_FILENAMES = ["selected_mask_refined.png", "final_mask.png", "selected_mask_raw.png"]

# キー別スコア列の並び。multikey_matcher.pyのkey_namesと同じ(ref/display_name/date/
# spec_1/spec_2)。過去の実行(2026-08-25のSPEC_1/SPEC_2導入前、by_keyが3〜4キーしか
# 無い・キー名が違う("color")場合がある)はget()で欠損させ空欄にする(2026-08-25、
# ユーザー要望: 「全クエリのキーとクエリごとのscoreを書くようにしてほしい」
# 「REFの認識でも一対一対応はしているか」を目視確認できるようにするため)。
KEY_SCORE_COLUMNS = [
    ("REFスコア", "ref"),
    ("display_nameスコア", "display_name"),
    ("日付スコア", "date"),
    ("spec_1スコア", "spec_1"),
    ("spec_2スコア", "spec_2"),
]

HEADERS = [
    "認識した順番", "query(book_name)", "display_name",
    "目視確認(T/F)", "IoU一致度",
    "正解幅mm", "推定幅mm", "誤差mm", "2mm以内(把持成功目安)", "リトライ回数",
    *[h for h, _ in KEY_SCORE_COLUMNS],
    "勝ったキー", "識別margin", "識別確信度(confident)",
    "独立選択(一対一対応無し)", "一対一対応で選択変化",
    "認識した文字列",
    "処理時間sec", "メモ", "エラー",
]
COLUMN_WIDTHS = {
    "認識した順番": 12, "query(book_name)": 16,
    "display_name": 26, "目視確認(T/F)": 14, "IoU一致度": 12,
    "正解幅mm": 10, "推定幅mm": 10, "誤差mm": 10,
    "2mm以内(把持成功目安)": 18, "リトライ回数": 12,
    **{h: 12 for h, _ in KEY_SCORE_COLUMNS},
    "勝ったキー": 14, "識別margin": 12,
    "識別確信度(confident)": 16,
    "独立選択(一対一対応無し)": 20, "一対一対応で選択変化": 18,
    "認識した文字列": 40, "処理時間sec": 12,
    "メモ": 24, "エラー": 20,
}


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def image_filename_for(dataset: str, shot: str, display_name: str, work_dir_name: str = "") -> str:
    """データセットごとの分かりやすい画像ファイル名を作る。

    diagonal-40のshot名は元々品目名そのもの(例: Target_R)で読みやすいが、
    stand-100のshot名は"<画像番号>__<book_nameコード>"(例: 1__ESC0305)で
    コードが読みにくいため、display_nameベースの名前に置き換える。

    stand-100・diagonal-40とも、実際に解決された作業フォルダ名(work_dir_name、例:
    `1-1-Target_XL`・`23-AXS_DAC_L`)をそのまま使う(2026-08-21/22、ユーザー要望:
    「images内の画像の命名規則もフォルダと同様にしてください」)。こうすることで
    区切り文字も含めてworkフォルダ名と常に一致することが保証される(別々に
    safe_name組み立てをやり直すと表記がずれる恐れがあるため、単一のソース=
    work_dir.nameから作る)。
    """
    if work_dir_name:
        return f"{work_dir_name}.png"
    if dataset == "stand-100":
        return f"{safe_name(display_name or shot)}.png"
    return f"{safe_name(shot)}.png"


def resolve_work_dir(base_dir: Path, dataset: str, shot: str, display_name: str = "") -> Path:
    """shotに対応する作業フォルダを解決する。

    stand-100のwidth_eval_work_rotfixは、shot名(`{画像番号}__{REF}`)のまま
    ではなく`{認識番号}-{画像番号}-{display_name}`形式にリネーム済み(2026-08-21、
    別タスクで実施)なので、直接一致しない場合はこのパターンでフォールバック探索する。
    diagonal-40も2026-08-22から作業フォルダ名の先頭に処理順を付けるようになった
    (`{処理順}-{shot名}`)ため、同様にフォールバック探索する。

    【2026-08-21 重大バグ修正】画像番号のみ(例: `^\\d+-1-`)で前方一致させていたため、
    同じ画像番号を共有する20件全部が同じフォルダ(ソート順で最初に見つかったもの)に
    解決されてしまい、Excelの識別スコア以降の列が20件ずつ同じ値になっていた
    (ユーザー報告により発覚)。display_nameのsafe_name化した値まで完全一致させる
    ことで、1件ずつ正しく一意なフォルダに解決するよう修正した。

    【2026-08-25 再発・修正】display_nameが複数品目で重複する場合(例:
    master_catheter_2.jsonは5品目とも display_name="OPTIMA")、上記の
    display_name一致だけでは再び複数候補が一致してしまい、同じ症状(final.pngが
    見つからない等)が再発した。width_mm_validation.py側でフォルダ名にbook_name
    (query、マスタのキーなので必ず一意)を付けるよう修正したので、まずそちらの
    新形式(`{処理順}-{image_id}-{display_name}-{book_name}`)で一意に探し、
    見つからなければ旧形式(book_name無し)にフォールバックする。
    """
    direct = base_dir / shot
    if direct.exists():
        return direct
    # diagonal-40だけ1画像=1品目固定の特別扱い。それ以外(stand-100・新規データセットとも)は
    # width_mm_validation.pyのis_cross_modeと同じ「画像×マスタ全品目」方式のフォルダ名なので、
    # 同じ判定で探索する(2026-08-25、ユーザー要望: データセット名を"stand-100"に限定せず一般化)。
    is_cross_mode = dataset != "diagonal-40"
    if is_cross_mode and "__" in shot:
        image_id, book_name = shot.split("__", 1)
        candidates = sorted(base_dir.iterdir()) if base_dir.exists() else []

        bn = safe_name(book_name)
        pattern_new = re.compile(rf"^\d+-{re.escape(image_id)}-.+-{re.escape(bn)}$")
        matches = [c for c in candidates if c.is_dir() and pattern_new.match(c.name)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"⚠ resolve_work_dir: shot={shot} book_name={book_name!r} に対して"
                  f"複数候補が一致し一意に決まりません(新形式): {[m.name for m in matches]}")

        if display_name:
            expected = f"{image_id}-{safe_name(display_name)}"
            pattern_old = re.compile(rf"^\d+-{re.escape(expected)}$")
            matches = [c for c in candidates if c.is_dir() and pattern_old.match(c.name)]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                print(f"⚠ resolve_work_dir: shot={shot} display_name={display_name!r} に対して"
                      f"複数候補が一致し一意に決まりません(旧形式): {[m.name for m in matches]}")
    elif not is_cross_mode:
        expected = safe_name(shot)
        pattern = re.compile(rf"^\d+-{re.escape(expected)}$")
        matches = [
            cand for cand in (sorted(base_dir.iterdir()) if base_dir.exists() else [])
            if cand.is_dir() and pattern.match(cand.name)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"⚠ resolve_work_dir: shot={shot} に対して"
                  f"複数候補が一致し一意に決まりません: {[m.name for m in matches]}")
    return direct


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


def by_key_scores_for_selected_mask(debug: dict) -> dict:
    """選択マスクの、5キー(ref/display_name/date/spec_1/spec_2)それぞれの生スコアを返す。
    古い実行(SPEC_1/SPEC_2導入前)はキーが無い/違う場合があるので、無ければ空のまま。"""
    selected_mask = debug.get("selected_mask")
    for m in debug.get("per_mask", []):
        if m.get("mask") == selected_mask:
            return m.get("by_key", {})
    return {}


def decode_uncompressed_rle(counts: list[int], height: int, width: int) -> np.ndarray:
    """COCOの非圧縮RLE(countsがintのリスト)を0/255マスクにデコードする。列優先(Fortran順)で
    0/1が交互に続くランレングス形式(reco/stand-100/scripts/compare_quad_fit.pyの同名関数と同じ)。"""
    flat = np.zeros(height * width, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if c:
            flat[idx: idx + c] = val * 255
        idx += c
        val = 1 - val
    return flat.reshape((height, width), order="F")


def segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """COCOセグメンテーション(ポリゴン形式 or 非圧縮RLE形式)を0/255マスクにラスタライズする。"""
    if isinstance(segmentation, dict):
        counts = segmentation["counts"]
        if isinstance(counts, str):
            raise NotImplementedError(
                "圧縮RLE(countsが文字列)は未対応です(pycocotools未導入のため)。"
            )
        return decode_uncompressed_rle(counts, height, width)

    mask = np.zeros((height, width), dtype=np.uint8)
    for part in segmentation:
        pts = np.array(part, dtype=np.float64).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def load_gt_masks_by_image(dataset: str) -> dict[str, list[np.ndarray]]:
    """アノテーションJSON(CVAT/COCO形式)を読み、画像を特定するキー->GTマスク一覧、を返す。

    stand-100は画像ファイル名が"1.png"〜"5.png"なのでキーは画像番号("1"等)、
    diagonal-40は画像ファイル名がshot名そのもの("AXS_DAC_L.png"等)なのでキーはshot名。
    1画像につき複数(stand-100は20個/枚)のGTポリゴンがあるので、IoU一致度は
    「選択マスクと、その画像内の全GTポリゴンとの最大IoU」とする(2026-08-22、
    ユーザー要望: 「目視確認の右隣にIoU一致度を入れたい」。個々のGTがどのbook_nameに
    対応するかの正解ラベル付けは無いため、識別の正誤とは独立に「セグメンテーション
    そのものの幾何精度」を見る指標として算出する)。
    """
    # 既知の2データセットは専用パス(stand-100はannotations統合版のファイル名が
    # 異なるため個別指定)。それ以外の新規データセットは、reco/<dataset>/annotations/
    # instances_default.json という共通の置き場所を既定として探す(2026-08-25、
    # ユーザー要望: 新規データセットでもIoUを計算したい)。
    ann_path = ANNOTATIONS_JSON.get(dataset) or (RECO_ROOT / dataset / "annotations" / "instances_default.json")
    if not ann_path.exists():
        return {}
    try:
        d = json.loads(ann_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    images_by_id = {img["id"]: img for img in d.get("images", [])}
    anns_by_image_id: dict[int, list] = {}
    for ann in d.get("annotations", []):
        anns_by_image_id.setdefault(ann["image_id"], []).append(ann)

    # stand-100/diagonal-40は、annotations/(images/*.png)がdepth_shots側と180度向きが
    # 異なる旧来の撮影・アノテーション手順で作られている(2026-08-21、ユーザー確認済み:
    # depth_shotsが本来正しい向きで、annotations側の方を回転させる必要がある。実測でも
    # 無回転だと最良IoUが0.357、180度回転後は0.882まで跳ね上がることを確認済み)。
    # 2026-08-25、ユーザー確認: 新しいcapture_anno.pyはimages/とdepth_shots/を同一フレーム
    # から同時に保存するため回転関係が無く、新規データセット(OPTIMA-No1・OPTIMA-2等、
    # 今後作るものも含む)では回転してはいけない。この2データセットだけ回転する。
    NEEDS_180_ROTATION = {"stand-100", "diagonal-40"}
    needs_rotation = dataset in NEEDS_180_ROTATION

    result: dict[str, list[np.ndarray]] = {}
    for image_id, img in images_by_id.items():
        file_name = img.get("file_name", "")
        key = Path(file_name).stem  # "1.png"->"1", "AXS_DAC_L.png"->"AXS_DAC_L"
        h, w = img.get("height"), img.get("width")
        if not h or not w:
            continue
        masks = []
        for ann in anns_by_image_id.get(image_id, []):
            seg = ann.get("segmentation")
            if not seg:
                continue
            try:
                mask = segmentation_to_mask(seg, h, w)
                if needs_rotation:
                    mask = np.rot90(mask, 2)
                masks.append(mask)
            except NotImplementedError as e:
                print(f"⚠ GTアノテーション(id={ann.get('id')})のIoU算出をスキップ: {e}")
        result[key] = masks
    return result


def gt_image_key_for_shot(dataset: str, shot: str) -> str:
    if dataset != "diagonal-40" and "__" in shot:
        return shot.split("__", 1)[0]
    return shot


def load_selected_mask(work_dir: Path) -> np.ndarray | None:
    for name in SELECTED_MASK_FILENAMES:
        p = work_dir / name
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if m is not None:
                return m
    return None


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
    a = mask_a > 0
    b = mask_b > 0
    if a.shape != b.shape:
        return None
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return None
    return float(inter) / float(union)


def best_iou_for_row(dataset: str, shot: str, work_dir: Path,
                      gt_masks_by_image: dict[str, list[np.ndarray]]) -> float | None:
    gt_masks = gt_masks_by_image.get(gt_image_key_for_shot(dataset, shot))
    if not gt_masks:
        return None
    selected = load_selected_mask(work_dir)
    if selected is None:
        return None
    ious = [iou for gt in gt_masks if (iou := compute_iou(selected, gt)) is not None]
    return max(ious) if ious else None


def add_summary_sheet(wb: Workbook, data_sheet_name: str) -> None:
    """先頭に「集計」シートを追加し、選択率・誤選択率・誤差2mm未満率を数式で出す。

    2026-09-11、ユーザー要望: 「Excelの集計が終わるたびに選択率と誤差2mm未満率を
    計算してもらうのは面倒」とのことで、build_dataset_report()の出力に毎回自動で
    含めるようにした。列全体(D:D, I:I等)を参照する数式にしてあるので、目視確認
    (T/F)を後から追記・修正しても開き直せば自動で再計算される。
    """
    if "集計" in wb.sheetnames:
        del wb["集計"]
    ws = wb.create_sheet("集計", 0)

    src = f"'{data_sheet_name}'!"
    total_col = get_column_letter(HEADERS.index("query(book_name)") + 1)
    review_col = get_column_letter(HEADERS.index("目視確認(T/F)") + 1)
    within2mm_col = get_column_letter(HEADERS.index("2mm以内(把持成功目安)") + 1)
    total_expr = f"(COUNTA({src}{total_col}:{total_col})-1)"

    rows = [
        ("項目", "値", "内訳"),
        ("総件数", f"={total_expr}", ""),
        ("選択率(目視確認T)", f'=COUNTIF({src}{review_col}:{review_col},"T")/{total_expr}',
         f'=COUNTIF({src}{review_col}:{review_col},"T")&"/"&{total_expr}'),
        ("誤選択率(目視確認F)", f'=COUNTIF({src}{review_col}:{review_col},"F")/{total_expr}',
         f'=COUNTIF({src}{review_col}:{review_col},"F")&"/"&{total_expr}'),
        ("誤差2mm未満率", f'=COUNTIF({src}{within2mm_col}:{within2mm_col},"○")/{total_expr}',
         f'=COUNTIF({src}{within2mm_col}:{within2mm_col},"○")&"/"&{total_expr}'),
    ]
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, val in enumerate(row, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            if r_idx == 1:
                cell.font = Font(bold=True)
    for r in range(3, 6):
        ws.cell(row=r, column=2).number_format = "0.0%"
    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14


def build_dataset_report(dataset: str, suffix: str = "") -> None:
    """suffixを指定すると、入力データは width_eval_result{suffix}.csv /
    reco_result{suffix}/work/ から読む(2026-08-21、回転バグ修正版=_rotfixを反映する際に追加。
    2026-08-25、work_root自体をreco_result{suffix}/work/に変更したのに合わせて更新)。

    xlsxと画像とwork/は全て reco_result{suffix}/ の直下にまとまって入る
    (2026-08-21、ユーザー要望: 「Excelファイルが毎回上書きされるのは困る」
    「画像が入るフォルダ名の区別もつきにくい」への対応。2026-08-25、ユーザー要望:
    「width_eval_workフォルダは不要、reco_result内に入れてほしい」)。
    width_mm_validation.py が最初からreco_result{suffix}/work/を作業フォルダとして
    使うため、このスクリプトはそこにあるwork/をそのまま読むだけで、シンボリックリンクは
    作らない(以前はwidth_eval_work{suffix}/という別フォルダへのリンクだった)。
    """
    csv_path = RECO_ROOT / dataset / f"width_eval_result{suffix}.csv"
    if not csv_path.exists():
        print(f"⚠ 見つかりません、スキップ: {csv_path}")
        return

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # 認識した順番 = CSVへの追記順(=実際に処理された順、resume実行をまたいでも保たれる)。
    # stand-100のwork/フォルダ名・images/ファイル名の先頭数字と同じ値(2026-08-21、
    # ユーザー要望: 「エクセルの一番左列に認識した順番を加えてほしい」)。
    for i, r in enumerate(rows):
        r["_proc_order"] = i + 1
    rows.sort(key=lambda r: r["shot"])

    out_dir = RECO_ROOT / dataset / f"reco_result{suffix}"
    work_base_dir = out_dir / "work"
    images_dir = out_dir / "images"
    xlsx_path = out_dir / f"catheter_width_report{suffix}.xlsx"
    images_dir.mkdir(parents=True, exist_ok=True)

    gt_masks_by_image = load_gt_masks_by_image(dataset)

    for r in rows:
        shot = r["shot"]
        work_dir = resolve_work_dir(work_base_dir, dataset, shot, r.get("display_name", ""))
        debug = load_multikey_debug(work_dir)
        r["_selected_score"] = debug.get("selected_score")
        r["_winning_key"] = debug.get("winning_key", "")
        r["_margin"] = debug.get("margin")
        r["_confident"] = debug.get("confident")
        r["_independent_selected_mask"] = debug.get("independent_selected_mask")
        r["_hungarian_changed"] = debug.get("hungarian_changed_selection")
        r["_recognized_text"] = recognized_text_for_selected_mask(debug)
        r["_by_key"] = by_key_scores_for_selected_mask(debug)
        r["_iou"] = best_iou_for_row(dataset, shot, work_dir, gt_masks_by_image)

        img_filename = image_filename_for(dataset, shot, r.get("display_name", ""), work_dir.name)
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
            r["_proc_order"],
            r.get("query", ""),
            r.get("display_name", ""),
            "",
            (round(r["_iou"], 3) if r["_iou"] is not None else ""),
            true_mm,
            pred_mm,
            err_mm,
            ("○" if err_mm is not None and err_mm <= 2.0 else ("" if err_mm is None else "×")),
            to_float(r.get("retry_count")),
            *[to_float(r["_by_key"].get(key)) for _, key in KEY_SCORE_COLUMNS],
            r["_winning_key"],
            to_float(r["_margin"]),
            ("" if r["_confident"] is None else str(r["_confident"])),
            (r["_independent_selected_mask"] or ""),
            ("" if r["_hungarian_changed"] is None else str(r["_hungarian_changed"])),
            r["_recognized_text"],
            to_float(r.get("elapsed_sec")),
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

    add_summary_sheet(wb, ws.title)

    wb.save(xlsx_path)
    print(f"✔ [{dataset}] Excel -> {xlsx_path} ({len(rows)}行)")
    print(f"✔ [{dataset}] 画像 -> {images_dir} ({len(list(images_dir.glob('*.png')))}枚)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="reco/<dataset>/ のレポートだけを作る(例: 新規に追加したデータセット名)。"
                         "省略時は従来通りDATASETS(stand-100, diagonal-40)を両方処理する"
                         "(2026-08-25、ユーザー要望: データセット名を固定2種に限定せず"
                         "一般化してほしい)。")
    ap.add_argument("--suffix", default="",
                    help="入力データの接尾辞(例: _rotfix、width_mm_validation.pyの"
                         "--work-suffixに対応)。")
    args = ap.parse_args()
    targets = [args.dataset] if args.dataset else DATASETS
    for dataset in targets:
        build_dataset_report(dataset, suffix=args.suffix)


if __name__ == "__main__":
    main()
