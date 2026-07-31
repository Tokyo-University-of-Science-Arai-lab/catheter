#!/usr/bin/env python3
"""
offline_pointcloud_debug_SAM3.py の実行結果から、撮影画像ごとの要約CSVを作る。

出力: <run_dir>/shot<N>/recognition_summary.csv
  書籍名(display_name), 推定幅[mm], 誤差[mm](符号付き)

誤差は 推定幅 - 正解幅 なので、正なら過大推定、負なら過小推定。

使い方:
  python summarize_eval_csv.py                       # 最新の実行結果を対象にする
  python summarize_eval_csv.py <run_dir>             # 実行結果を指定する
"""

import csv
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_BASE_DIR = BASE_DIR / "captures" / "5shot_catheter_offline"
MASTER_JSON = BASE_DIR / "master_20260216.json"

OUT_NAME = "recognition_summary.csv"
HEADER = ["書籍名", "推定幅[mm]", "誤差[mm]"]


def latest_run_dir(base_dir: Path) -> Path:
    runs = [p for p in base_dir.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"results.csv を持つ実行結果がありません: {base_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_display_names(master_json: Path) -> dict[str, str]:
    """book_name -> display_name の対応表を作る。"""
    data = json.loads(master_json.read_text(encoding="utf-8"))
    return {
        item["book_name"]: item.get("display_name") or item["book_name"]
        for item in data
    }


def fmt(value, digits=3):
    """空欄や変換できない値は '-' にする（失敗ケース用）。"""
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def main():
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_run_dir(OFFLINE_BASE_DIR)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv がありません: {results_csv}")

    display_names = load_display_names(MASTER_JSON)

    with open(results_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # shot_id ごとに仕分ける
    by_shot: dict[str, list[dict]] = {}
    for r in rows:
        by_shot.setdefault(r["shot_id"], []).append(r)

    print(f"対象: {run_dir}")
    total = 0

    for shot_id in sorted(by_shot, key=int):
        shot_rows = sorted(by_shot[shot_id], key=lambda r: int(r["master_index"]))
        out_path = run_dir / f"shot{shot_id}" / OUT_NAME
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Excelで開いても文字化けしないよう utf-8-sig で書く
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for r in shot_rows:
                writer.writerow([
                    display_names.get(r["book_name"], r["book_name"]),
                    fmt(r["pred_book_width_mm"]),
                    fmt(r["signed_error_mm"]),
                ])

        total += len(shot_rows)
        print(f"  {out_path.relative_to(run_dir)}  ({len(shot_rows)} 行)")

    print(f"合計 {total} 行を書き出しました。")


if __name__ == "__main__":
    main()
