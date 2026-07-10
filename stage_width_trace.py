#!/usr/bin/env python3
"""
幅推定の各工程で「その時点のマスクだけを使ったら何mmになるか」を計算し、
GTとの差分を工程ごとに数値で追跡するスクリプト。

工程定義（すべて debug_mask_shape/*_rectangularity.json と pca_result_offline.json
に既に保存されている値から算出。画像は一切読まない）:

  Stage1 「深度フィルタ＋背表紙帯補完後マスク」
      width_median_px（マスクを長辺方向に短冊分割した際の幅の中央値）。
      SAM/OCRでマスクが選択され、深度フィルタと背表紙帯補完を経た直後の、
      最終アルゴリズム(M1/M2)にかける直前の「素のマスク形状」から出る幅。

  Stage2 「最終手法（M1 or M2）が出した幅」
      book_width_mm そのもの（実際にパイプラインが出力する最終値）。

  ※ 「深度フィルタ前の生マスク」も候補として検討したが、その時点では
     rect_size（長辺）がまだ意味を持たず area/length近似が破綻して非現実的な
     値になるため、信頼できる区間としてStage1以降のみを対象にした。

Stage1 は px 単位なので、z_median_raw と normal ベクトル・カメラ内部パラメータ
（fx, fy）を使って mm に変換する（パイプライン本体の scale_m_per_px 計算式と同一）。

使い方:
  python3 stage_width_trace.py [run_dir]
  run_dir を省略すると captures/100test_offline/ 内の最新実行を使う。

出力:
  <run_dir>/stage_width_trace.csv
  ケースごとの内訳 + 書籍別の工程別平均ズレのサマリを標準出力に表示
"""

import csv
import json
import math
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_BASE = BASE_DIR / "captures" / "100test_offline"
MASTER_JSON = BASE_DIR / "master_20260216.json"
THRESHOLD_MM = 2.0

CATEGORY_LABELS = {
    "1_ok_to_ok": "1. 2mm未満 → 2mm未満（維持）",
    "2_ok_to_ng": "2. 2mm未満 → 2mm以上（悪化）",
    "3_ng_to_ok": "3. 2mm以上 → 2mm未満（改善）",
    "4_ng_to_ng": "4. 2mm以上 → 2mm以上（維持）",
}


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
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


def scale_m_per_px(z_median_raw, depth_scale, normal, fx, fy):
    z_m = z_median_raw * depth_scale
    nx, ny = normal
    return z_m * math.sqrt((nx / fx) ** 2 + (ny / fy) ** 2)


def collect_case(case_dir: Path, gt_mm: float, book_name: str, cam: dict) -> dict:
    row = {
        "case_dir": str(case_dir),
        "book_name": book_name,
        "gt_mm": gt_mm,
        "method": None,
        "stage1_filtered_mask_mm": None,
        "stage2_final_mm": None,
        "err1_vs_gt": None,
        "err2_vs_gt": None,
        "delta_final_algorithm": None,     # stage2 - stage1
    }

    rect_path = find_glob_first(case_dir / "debug_mask_shape", "*_rectangularity.json")
    rect = load_json(rect_path) if rect_path else None

    pca_path = find_glob_first(case_dir, "pca_result_offline.json") or find_glob_first(case_dir, "pca_result.json")
    pca = load_json(pca_path) if pca_path else None

    if pca:
        row["stage2_final_mm"] = pca.get("book_width_mm")
        bwi = pca.get("book_width_info", {})
        row["method"] = "M2" if "no_refine" in str(bwi.get("method")) else ("M1" if bwi.get("method") else None)

    if rect:
        dp = rect.get("depth_prefilter", {})
        dfd = dp.get("depth_filter_detail", {})
        sc = dp.get("spine_completion_after_depth_prefilter", {})
        z_median_raw = dfd.get("z_median_raw")
        normal = sc.get("normal")
        width_median_px = rect.get("width_median_px")

        if z_median_raw and normal and cam and width_median_px is not None:
            scale = scale_m_per_px(z_median_raw, cam["depth_scale"], normal, cam["fx"], cam["fy"])
            row["stage1_filtered_mask_mm"] = width_median_px * scale * 1000

    for key, stage_key in [("err1_vs_gt", "stage1_filtered_mask_mm"),
                            ("err2_vs_gt", "stage2_final_mm")]:
        v = row[stage_key]
        row[key] = (v - gt_mm) if v is not None else None

    if row["stage1_filtered_mask_mm"] is not None and row["stage2_final_mm"] is not None:
        row["delta_final_algorithm"] = row["stage2_final_mm"] - row["stage1_filtered_mask_mm"]

    row["category"] = classify(row["err1_vs_gt"], row["err2_vs_gt"])
    row["root_cause"] = root_cause(row["err1_vs_gt"], row["err2_vs_gt"], row["delta_final_algorithm"])

    return row


def classify(err1, err2):
    """誤差1(mask段階)→誤差2(最終段階)の絶対値2mm閾値での遷移を4分類する。"""
    if err1 is None or err2 is None:
        return None
    ok1 = abs(err1) < THRESHOLD_MM
    ok2 = abs(err2) < THRESHOLD_MM
    if ok1 and ok2:
        return "1_ok_to_ok"
    if ok1 and not ok2:
        return "2_ok_to_ng"
    if not ok1 and ok2:
        return "3_ng_to_ok"
    return "4_ng_to_ng"


def root_cause(err1, err2, delta):
    """最終誤差(err2)が2mm以上の場合に、どの工程が主因かを切り分ける。

    - final_algorithm : マスク段階は正常(<2mm)だったのに最終手法が壊した
    - mask_stage       : マスク段階で既に2mm以上ズレており、最終手法はほぼ動かしていない(|delta|<2mm)
    - both             : マスク段階のズレに加え、最終手法もさらに2mm以上動かしている(複合)
    - None             : 最終誤差が2mm未満（問題なし）、または判定に必要な値が欠損
    """
    if err2 is None or abs(err2) < THRESHOLD_MM:
        return None
    if err1 is None or delta is None:
        return "unknown"
    if abs(err1) < THRESHOLD_MM:
        return "final_algorithm"
    if abs(delta) < THRESHOLD_MM:
        return "mask_stage"
    return "both"


def _fmt(v, fmt=".1f"):
    return f"{v:{fmt}}" if v is not None else "  N/A"


def _fmt_signed(v, fmt="+.1f"):
    return f"{v:{fmt}}" if v is not None else "  N/A"


def print_case_table(rows):
    print(f"\n{'idx':>4} {'book_name':<28} {'GT':>6} {'St1msk':>7} {'St2final':>8} "
          f"{'誤差1':>7} {'誤差2':>7} | {'最終手法での増減':>12} {'method':>4}")
    print("-" * 100)
    for r in rows:
        print(
            f"{Path(r['case_dir']).name:>4} {r['book_name'][:28]:<28} {r['gt_mm']:>6.1f} "
            f"{_fmt(r['stage1_filtered_mask_mm']):>7} {_fmt(r['stage2_final_mm']):>8} "
            f"{_fmt(r['err1_vs_gt']):>7} {_fmt(r['err2_vs_gt']):>7} | "
            f"{_fmt(r['delta_final_algorithm']):>12} {str(r['method']):>4}"
        )


def print_book_summary(rows, books):
    print("\n" + "=" * 90)
    print("  書籍別: 工程ごとの平均ズレ（GTからの平均誤差 mm、符号あり）")
    print("=" * 90)
    print(f"{'book_name':<40} {'n':>3} {'誤差1(mask段階)':>14} {'誤差2(最終)':>10} {'最終手法での増減':>12}")
    print("-" * 90)
    for book in books:
        name = book["book_name"]
        book_rows = [r for r in rows if r["book_name"] == name]
        if not book_rows:
            continue
        e1 = [r["err1_vs_gt"] for r in book_rows if r["err1_vs_gt"] is not None]
        e2 = [r["err2_vs_gt"] for r in book_rows if r["err2_vs_gt"] is not None]
        d = [r["delta_final_algorithm"] for r in book_rows if r["delta_final_algorithm"] is not None]
        m1 = sum(e1) / len(e1) if e1 else None
        m2 = sum(e2) / len(e2) if e2 else None
        md = sum(d) / len(d) if d else None
        print(f"{name[:40]:<40} {len(book_rows):>3} {_fmt(m1):>14} {_fmt(m2):>10} {_fmt(md):>12}")

    print("\n  ※ 誤差1: 深度フィルタ＋背表紙帯補完後マスクの中央幅（M1/M2にかける直前）とGTの差")
    print("  ※ 誤差2: 最終手法(M1/M2)が実際に出力した幅とGTの差（= 現状の abs_error_mm と同義）")
    print("  ※ 最終手法での増減 = 誤差2-誤差1。プラスに大きい書籍ほど、マスクは大きく崩れていないのに")
    print("     最終の幅推定アルゴリズム(M1のpx幅計算 or M2の点群PCAスライス)自体が誤差を上乗せしている。")
    print("  ※ 誤差1が既に大きい書籍は、深度フィルタ/背表紙補完より前（SAM/OCRのマスク選択そのもの）を疑う。\n")


def print_category_breakdown(rows):
    print("\n" + "=" * 100)
    print(f"  誤差2mm閾値での遷移分類（マスク段階 誤差1 → 最終段階 誤差2）")
    print("=" * 100)

    classified = [r for r in rows if r["category"] is not None]
    n_fail = len(rows) - len(classified)

    for key in ["1_ok_to_ok", "2_ok_to_ng", "3_ng_to_ok", "4_ng_to_ng"]:
        group = [r for r in classified if r["category"] == key]
        print(f"\n--- {CATEGORY_LABELS[key]}  ({len(group)}件) ---")
        if not group:
            continue
        print(f"{'idx':>4} {'book_name':<28} {'GT':>6} {'誤差1':>8} {'誤差2':>8} {'増減(誤差2-誤差1)':>14} {'method':>4}")
        for r in sorted(group, key=lambda x: abs(x["err2_vs_gt"]), reverse=True):
            print(
                f"{Path(r['case_dir']).name:>4} {r['book_name'][:28]:<28} {r['gt_mm']:>6.1f} "
                f"{_fmt_signed(r['err1_vs_gt']):>8} {_fmt_signed(r['err2_vs_gt']):>8} "
                f"{_fmt_signed(r['delta_final_algorithm']):>14} {str(r['method']):>4}"
            )

    print(f"\n  ※ +は「GTより大きく推定」、-は「GTより小さく推定」、増減の+は最終手法で誤差が拡大、-は縮小したことを示す。")
    print(f"  ※ fail等でstage1/stage2いずれかが欠損しているケース: {n_fail}件（分類対象外）\n")


ROOT_CAUSE_LABELS = {
    "final_algorithm": "①最終アルゴリズム原因（マスクは正常<2mmだが最終手法が壊した）",
    "mask_stage": "②マスク/それ以前原因（マスク段階で既に2mm以上ズレ、最終手法はほぼ無関係）",
    "both": "③複合原因（マスク段階のズレに最終手法がさらに2mm以上上乗せ）",
    "unknown": "④不明（データ欠損）",
}


def print_root_cause_breakdown(rows):
    print("\n" + "=" * 100)
    print("  最終誤差2mm以上(32件)の原因工程切り分け")
    print("=" * 100)

    bad = [r for r in rows if r["root_cause"] is not None]
    for key in ["final_algorithm", "mask_stage", "both", "unknown"]:
        group = [r for r in bad if r["root_cause"] == key]
        print(f"\n--- {ROOT_CAUSE_LABELS[key]}  ({len(group)}件) ---")
        if not group:
            continue
        counts = {}
        for r in group:
            counts[r["book_name"]] = counts.get(r["book_name"], 0) + 1
        for name, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {name[:45]:<45} {c}件")

    print("\n  ※ ①は最終手法(M1のpx幅計算 or M2の点群PCAスライス)自体を疑うべきケース")
    print("  ※ ②はマスク段階以前（深度フィルタ/背表紙補完、あるいはさらに手前のSAM/OCRマスク選択）を疑うべきケース")
    print("  ※ ③は両方に問題がある、または連鎖的に悪化しているケース\n")


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else latest_run_dir(OFFLINE_BASE)
    if not run_dir.exists():
        print(f"エラー: {run_dir} が存在しません")
        sys.exit(1)

    print(f"対象ディレクトリ: {run_dir}")

    master_data = load_json(MASTER_JSON)
    if master_data is None:
        print(f"エラー: master.json が見つかりません: {MASTER_JSON}")
        sys.exit(1)
    books = [
        {"book_name": b["book_name"], "book_width_mm": float(b["book_width"])}
        for b in master_data
    ]
    n_books = len(books)

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
        book = books[(idx - 1) % n_books]
        cam = load_json(case_dir / "camera_params.json")
        row = collect_case(case_dir, gt_mm=book["book_width_mm"], book_name=book["book_name"], cam=cam)
        row["test_index"] = idx
        rows.append(row)

    out_csv = run_dir / "stage_width_trace.csv"
    fieldnames = [
        "test_index", "book_name", "gt_mm", "method",
        "stage1_filtered_mask_mm", "stage2_final_mm",
        "err1_vs_gt", "err2_vs_gt",
        "delta_final_algorithm", "category", "root_cause",
        "case_dir",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV保存: {out_csv}")

    print_case_table(sorted(rows, key=lambda r: (r["err2_vs_gt"] is None, -(abs(r["err2_vs_gt"]) if r["err2_vs_gt"] is not None else 0)))[:40])
    print_book_summary(rows, books)
    print_category_breakdown(rows)
    print_root_cause_breakdown(rows)


if __name__ == "__main__":
    main()
