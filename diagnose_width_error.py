#!/usr/bin/env python3
"""
offline_pointcloud_debug.py の実行結果を段階別に診断するスクリプト。

各テストケースの以下の情報を収集して CSV とサマリを出力する:
  ① OCR選択     : 最高スコア・閾値以上か・選択テキストに正解が含まれるか
  ② マスク品質  : clean_rectangle / IoU / width_cv / needs_refine
  ③ 深度フィルタ: 除去比率・スパイン補完で追加された画素数
  ④ 幅推定手法  : M1(needs_refine=True) / M2(False)
  ⑤ 最終誤差   : abs_error_mm

使い方:
  python3 diagnose_width_error.py [run_dir]

  run_dir を省略すると captures/100test_offline/ 内の最新実行を使う。
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_BASE = BASE_DIR / "captures" / "100test_offline"
MASTER_JSON  = BASE_DIR / "master_20260216.json"
OCR_THRESHOLD = 40  # similarity_scores.json の合格スコア


# ── ユーティリティ ───────────────────────────────────────────

def latest_run_dir(offline_base: Path) -> Path:
    dirs = sorted(
        [d for d in offline_base.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )
    if not dirs:
        raise FileNotFoundError(f"実行結果が見つかりません: {offline_base}")
    return dirs[-1]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_glob_first(directory: Path, pattern: str):
    """ワイルドカードで最初にマッチしたファイルを返す。"""
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def method_short(method_str: str) -> str:
    if method_str is None:
        return "?"
    if "no_refine" in method_str:
        return "M2"
    return "M1"


# ── 1ケース分の情報収集 ─────────────────────────────────────

def collect_case(case_dir: Path, gt_width_mm: float, book_name: str, query: str) -> dict:
    row = {
        "case_dir":    str(case_dir),
        "book_name":   book_name,
        "query":       query,
        "gt_mm":       gt_width_mm,
        "pred_mm":     None,
        "abs_err_mm":  None,
        "status":      "unknown",
        # ① OCR
        "ocr_top_score":    None,
        "ocr_top_text":     None,
        "ocr_pass":         None,   # top_score >= threshold
        "query_in_ocr_text": None,  # クエリ文字列がOCR結果に含まれるか
        # ② マスク品質
        "clean_rect":       None,
        "needs_refine":     None,
        "rect_iou":         None,
        "width_cv":         None,
        "approx_vertices":  None,
        # ③ 深度フィルタ
        "depth_removed_px":    None,
        "depth_remaining_ratio": None,
        "spine_added_px":      None,
        "mask_px_final":       None,
        # ④ 幅推定手法
        "method":           None,
    }

    # case_result.json
    cr = load_json(case_dir / "case_result.json")
    if cr:
        row["pred_mm"]    = cr.get("pred_book_width_mm")
        row["abs_err_mm"] = cr.get("abs_error_mm")
        row["status"]     = cr.get("status", "unknown")

    # ① similarity_scores.json
    ss = load_json(case_dir / "similarity_scores.json")
    if ss and ss.get("scores"):
        top = ss["scores"][0]
        score = top.get("score", 0)
        text  = top.get("text", "")
        row["ocr_top_score"]     = score
        row["ocr_top_text"]      = text[:60]
        row["ocr_pass"]          = score >= OCR_THRESHOLD
        # クエリの最初の単語（例: "Medtronic"）がOCRテキストに含まれるかで判定
        key = (query.split()[0] if query.strip() else query).lower()
        row["query_in_ocr_text"] = (key in text.lower()) if text else False

    # ② ③ ④  pca_result_offline.json から取得（mask番号が可変なのでglobで探す）
    pca_path = find_glob_first(case_dir, "pca_result_offline.json")
    if pca_path is None:
        pca_path = find_glob_first(case_dir, "pca_result.json")
    if pca_path:
        pca = load_json(pca_path)
        if pca:
            bwi = pca.get("book_width_info", {})
            row["method"] = method_short(bwi.get("method"))

            sr = bwi.get("shape_rectangularity", {})
            row["clean_rect"]      = sr.get("clean_rectangle")
            row["needs_refine"]    = sr.get("needs_refine")
            row["rect_iou"]        = sr.get("rotated_rect_iou")
            row["width_cv"]        = sr.get("width_cv")
            row["approx_vertices"] = sr.get("approx_vertices")

            dpf = sr.get("depth_prefilter", {})
            flt = dpf.get("depth_filter_detail", {})
            row["depth_removed_px"]       = flt.get("removed_pixel_count")
            row["depth_remaining_ratio"]  = flt.get("remaining_ratio")
            spc = dpf.get("spine_completion_after_depth_prefilter", {})
            row["spine_added_px"]  = spc.get("added_pixel_count")
            row["mask_px_final"]   = spc.get("area_after")

    return row


# ── サマリ表示 ───────────────────────────────────────────────

def _fmt(v, fmt=".2f"):
    return f"{v:{fmt}}" if v is not None else "  N/A "


def print_summary(rows: list[dict], books: list[dict]):
    n_books = len(books)
    n_rounds = max((len(rows) // n_books), 1)

    print("\n" + "=" * 90)
    print("  書籍別サマリ")
    print("=" * 90)

    header = (
        f"{'書籍名':<34} {'GT':>5} "
        f"{'pred平均':>8} {'誤差平均':>8} {'誤差最大':>8} "
        f"{'OCR合格率':>9} {'M1率':>5} {'M2率':>5}"
    )
    print(header)
    print("-" * 90)

    for bi, book in enumerate(books):
        name = book["book_name"]
        gt   = book["book_width_mm"]
        book_rows = [r for r in rows if r["book_name"] == name]
        if not book_rows:
            continue

        preds  = [r["pred_mm"]    for r in book_rows if r["pred_mm"]    is not None]
        errors = [r["abs_err_mm"] for r in book_rows if r["abs_err_mm"] is not None]
        ocr_ok = [r for r in book_rows if r["ocr_pass"] is True]
        m1     = [r for r in book_rows if r["method"] == "M1"]
        m2     = [r for r in book_rows if r["method"] == "M2"]
        n      = len(book_rows)

        pred_mean = sum(preds)  / len(preds)  if preds  else None
        err_mean  = sum(errors) / len(errors) if errors else None
        err_max   = max(errors)               if errors else None
        ocr_rate  = len(ocr_ok) / n * 100    if n > 0  else None
        m1_rate   = len(m1) / n * 100         if n > 0  else None
        m2_rate   = len(m2) / n * 100         if n > 0  else None

        name_s = name[:33]
        print(
            f"{name_s:<34} {gt:>5.1f} "
            f"{_fmt(pred_mean):>8} {_fmt(err_mean):>8} {_fmt(err_max):>8} "
            f"{_fmt(ocr_rate, '.0f')+'%':>9} "
            f"{_fmt(m1_rate, '.0f')+'%':>5} {_fmt(m2_rate, '.0f')+'%':>5}"
        )

    print()
    print("=" * 90)
    print("  ラウンド別誤差（書籍×ラウンド マトリクス）")
    print("=" * 90)

    round_header = f"{'書籍名':<34}" + "".join(f" R{r+1:>2}" for r in range(n_rounds))
    print(round_header)
    print("-" * (34 + n_rounds * 4))

    for book in books:
        name = book["book_name"]
        n_books_total = len(books)
        # test_index = round * n_books + book_index + 1  という配置を前提
        line = f"{name[:33]:<34}"
        for r in range(n_rounds):
            # ラウンド r のこの書籍に対応する行を探す
            book_rows = [
                row for row in rows
                if row["book_name"] == name
            ]
            if r < len(book_rows):
                e = book_rows[r]["abs_err_mm"]
                if e is None:
                    line += " --- "
                elif e >= 5.0:
                    line += f" \033[91m{e:4.1f}\033[0m"  # 赤 (5mm以上)
                elif e >= 2.0:
                    line += f" \033[93m{e:4.1f}\033[0m"  # 黄 (2mm以上)
                else:
                    line += f" {e:4.1f}"
            else:
                line += "     "
        print(line)

    print("\n  ※ \033[91m赤\033[0m ≥ 5mm  \033[93m黄\033[0m ≥ 2mm  白 < 2mm\n")

    print("=" * 90)
    print("  OCRスコアが低い / クエリ不一致のケース")
    print("=" * 90)
    problem_ocr = [
        r for r in rows
        if (r["ocr_pass"] is False) or (r["query_in_ocr_text"] is False)
    ]
    if problem_ocr:
        print(f"{'case_dir':<60} {'書籍名':<20} {'score':>6} {'クエリ一致':>8} {'誤差':>7}")
        print("-" * 104)
        for r in sorted(problem_ocr, key=lambda x: x["abs_err_mm"] or 0, reverse=True):
            cd = Path(r["case_dir"]).name
            print(
                f"{cd:<60} {r['book_name'][:19]:<20} "
                f"{_fmt(r['ocr_top_score'], '.0f'):>6} "
                f"{'✓' if r['query_in_ocr_text'] else '✗':>8} "
                f"{_fmt(r['abs_err_mm']):>7}"
            )
    else:
        print("  問題なし（全ケースでOCRスコアが閾値以上かつクエリ一致）")

    print()
    print("=" * 90)
    print("  誤差が大きいケース TOP10")
    print("=" * 90)
    top_err = sorted([r for r in rows if r["abs_err_mm"] is not None],
                     key=lambda x: x["abs_err_mm"], reverse=True)[:10]
    print(f"{'dir':<6} {'書籍名':<34} {'GT':>5} {'予測':>7} {'誤差':>7} {'手法':>4} {'OCR合格':>7} {'IoU':>6} {'wCV':>6}")
    print("-" * 90)
    for r in top_err:
        cd = Path(r["case_dir"]).name
        print(
            f"{cd:<6} {r['book_name'][:33]:<34} "
            f"{_fmt(r['gt_mm']):>5} {_fmt(r['pred_mm']):>7} {_fmt(r['abs_err_mm']):>7} "
            f"{str(r['method']):>4} "
            f"{'✓' if r['ocr_pass'] else '✗':>7} "
            f"{_fmt(r['rect_iou']):>6} {_fmt(r['width_cv']):>6}"
        )


# ── メイン ───────────────────────────────────────────────────

def main():
    # 実行ディレクトリの決定
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1])
    else:
        run_dir = latest_run_dir(OFFLINE_BASE)

    if not run_dir.exists():
        print(f"エラー: {run_dir} が存在しません")
        sys.exit(1)

    print(f"診断対象: {run_dir}")

    # master.json 読み込み
    master_data = load_json(MASTER_JSON)
    if master_data is None:
        print(f"エラー: master.json が見つかりません: {MASTER_JSON}")
        sys.exit(1)
    books = [
        {"book_name": b["book_name"], "book_width_mm": float(b["book_width"])}
        for b in master_data
    ]
    n_books = len(books)

    # テストケースを走査
    case_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    if not case_dirs:
        print("テストケースが見つかりません")
        sys.exit(1)

    rows = []
    for case_dir in case_dirs:
        idx = int(case_dir.name)
        book_idx  = (idx - 1) % n_books
        round_num = (idx - 1) // n_books + 1
        book      = books[book_idx]

        print(f"  [{idx:3d}] R{round_num} {book['book_name'][:30]}", end="\r")
        row = collect_case(
            case_dir,
            gt_width_mm=book["book_width_mm"],
            book_name=book["book_name"],
            query=book["book_name"],
        )
        row["test_index"]  = idx
        row["round"]       = round_num
        row["book_index"]  = book_idx
        rows.append(row)

    print(f"  {len(rows)} ケース収集完了              ")

    # CSV 出力
    out_csv = run_dir / "diagnosis.csv"
    fieldnames = [
        "test_index", "round", "book_index", "book_name", "query",
        "gt_mm", "pred_mm", "abs_err_mm", "status",
        "ocr_top_score", "ocr_pass", "query_in_ocr_text", "ocr_top_text",
        "clean_rect", "needs_refine", "rect_iou", "width_cv", "approx_vertices",
        "method",
        "depth_removed_px", "depth_remaining_ratio",
        "spine_added_px", "mask_px_final",
        "case_dir",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV保存: {out_csv}")

    # サマリ表示
    print_summary(rows, books)


if __name__ == "__main__":
    main()
