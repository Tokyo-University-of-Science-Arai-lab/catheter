#!/usr/bin/env python3
"""
book_width_eval_results_*.csv の誤差を集計するスクリプト。

- abs_error_mm < 2mm のケース数をカウント
- 2mm以上（または fail）のケースを誤差の大きい順に表示

使い方:
  python3 analyze_width_error.py [csv_path]

  csv_path を省略すると 20260708_203730 の結果を使う。
"""

import csv
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = (
    BASE_DIR / "captures" / "100test_offline" / "20260708_203730"
    / "book_width_eval_results_20260708_203730.csv"
)
THRESHOLD_MM = 2.0


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_CSV
    if not csv_path.exists():
        print(f"エラー: {csv_path} が見つかりません")
        sys.exit(1)

    ok_rows = []
    ng_rows = []
    fail_rows = []

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "success" or not row["abs_error_mm"]:
                fail_rows.append(row)
                continue
            err = float(row["abs_error_mm"])
            if err < THRESHOLD_MM:
                ok_rows.append(row)
            else:
                ng_rows.append(row)

    total = len(ok_rows) + len(ng_rows) + len(fail_rows)
    print(f"総ケース数: {total}")
    print(f"誤差 < {THRESHOLD_MM}mm : {len(ok_rows)} 件")
    print(f"誤差 >= {THRESHOLD_MM}mm: {len(ng_rows)} 件")
    print(f"失敗（fail）      : {len(fail_rows)} 件")

    print(f"\n=== 誤差 >= {THRESHOLD_MM}mm のケース（誤差の大きい順） ===")
    print(f"{'test_index':>10} {'book_name':<40} {'gt_mm':>7} {'pred_mm':>10} {'abs_error_mm':>13} {'case_dir'}")
    for row in sorted(ng_rows, key=lambda r: float(r["abs_error_mm"]), reverse=True):
        print(
            f"{row['test_index']:>10} {row['book_name'][:40]:<40} "
            f"{row['gt_book_width_mm']:>7} {row['pred_book_width_mm']:>10} "
            f"{row['abs_error_mm']:>13} {row['run_shot_dir']}"
        )

    if fail_rows:
        print(f"\n=== 失敗ケース ({len(fail_rows)}件) ===")
        print(f"{'test_index':>10} {'book_name':<40} {'error':<50} {'case_dir'}")
        for row in fail_rows:
            print(
                f"{row['test_index']:>10} {row['book_name'][:40]:<40} "
                f"{row['error'][:50]:<50} {row['run_shot_dir']}"
            )


if __name__ == "__main__":
    main()
