#!/usr/bin/env python3
"""100回試験の SAM2（2026-07-08）と SAM3 の結果を test_index どうしで突き合わせる。

使い方（SAM3側は既定で 100test_offline の最新runを拾う）:
    python compare_100test_sam2_vs_sam3.py
    python compare_100test_sam2_vs_sam3.py --sam3 captures/100test_offline/<timestamp>

出力: <SAM3のrun>/comparison_vs_sam2.csv と、標準出力のサマリ。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# SAM2時代のベースライン（旧環境 ~/pro_book に残っているもの）
DEFAULT_SAM2_RUN = Path(
    "/home/book/pro_book/pro_hand_book_python/captures/100test_offline/20260708_203730"
)

SAM3_BASE = BASE_DIR / "captures" / "100test_offline"

ERROR_THRESHOLDS_MM = [1.0, 1.5, 2.0]


def find_results_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("book_width_eval_results_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"結果CSVが見つかりません: {run_dir}")
    return candidates[-1]


def latest_sam3_run() -> Path:
    runs = [p for p in SAM3_BASE.iterdir() if p.is_dir() and list(p.glob("book_width_eval_results_*.csv"))]
    if not runs:
        raise FileNotFoundError(
            f"SAM3の実行結果が {SAM3_BASE} にありません。先に offline_100test_SAM3.py を実行してください。"
        )
    return sorted(runs, key=lambda p: p.name)[-1]


def load_rows(csv_path: Path) -> dict[int, dict]:
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {int(r["test_index"]): r for r in rows}


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(errors, total):
    out = {}
    for th in ERROR_THRESHOLDS_MM:
        n = sum(1 for e in errors if e is not None and e <= th)
        out[f"within_{th:.1f}mm"] = (n, n / total * 100 if total else 0.0)
    valid = [e for e in errors if e is not None]
    out["n_success"] = len(valid)
    out["mean"] = statistics.mean(valid) if valid else None
    out["median"] = statistics.median(valid) if valid else None
    out["max"] = max(valid) if valid else None
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare SAM2 vs SAM3 on the 100-case test.")
    parser.add_argument("--sam2", type=str, default=str(DEFAULT_SAM2_RUN), help="SAM2側のrunディレクトリ")
    parser.add_argument("--sam3", type=str, default=None, help="SAM3側のrunディレクトリ（既定: 最新）")
    args = parser.parse_args()

    sam2_run = Path(args.sam2).resolve()
    sam3_run = Path(args.sam3).resolve() if args.sam3 else latest_sam3_run()

    sam2 = load_rows(find_results_csv(sam2_run))
    sam3 = load_rows(find_results_csv(sam3_run))

    shared = sorted(set(sam2) & set(sam3))
    print(f"SAM2 run : {sam2_run}  ({len(sam2)} ケース)")
    print(f"SAM3 run : {sam3_run}  ({len(sam3)} ケース)")
    print(f"共通ケース: {len(shared)}\n")

    out_rows = []
    for i in shared:
        a, b = sam2[i], sam3[i]
        e2, e3 = to_float(a["abs_error_mm"]), to_float(b["abs_error_mm"])
        out_rows.append({
            "test_index": i,
            "book_name": b["book_name"],
            "gt_width_mm": b["gt_book_width_mm"],
            "sam2_pred_mm": a["pred_book_width_mm"],
            "sam3_pred_mm": b["pred_book_width_mm"],
            "sam2_abs_error_mm": a["abs_error_mm"],
            "sam3_abs_error_mm": b["abs_error_mm"],
            "delta_abs_error_mm": (e3 - e2) if (e2 is not None and e3 is not None) else "",
            "sam2_status": a["status"],
            "sam3_status": b["status"],
            "sam2_gt_width_mm": a["gt_book_width_mm"],
            "gt_width_matches": a["gt_book_width_mm"] == b["gt_book_width_mm"],
        })

    out_csv = sam3_run / "comparison_vs_sam2.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    mismatched = [r["test_index"] for r in out_rows if not r["gt_width_matches"]]
    if mismatched:
        print(f"⚠ 正解幅が両者で違うケース: {len(mismatched)} 件 → 誤差の直接比較はできません")
        print(f"   例: {mismatched[:10]}\n")

    n = len(shared)
    s2 = summarize([to_float(sam2[i]["abs_error_mm"]) for i in shared], n)
    s3 = summarize([to_float(sam3[i]["abs_error_mm"]) for i in shared], n)

    print(f"{'指標':<20} {'SAM2':>14} {'SAM3':>14}")
    print("-" * 50)
    print(f"{'成功':<20} {s2['n_success']:>10} /{n:<3} {s3['n_success']:>10} /{n:<3}")
    for th in ERROR_THRESHOLDS_MM:
        k = f"within_{th:.1f}mm"
        print(f"{'≦' + f'{th:.1f}mm':<20} {s2[k][0]:>6} ({s2[k][1]:4.1f}%) {s3[k][0]:>6} ({s3[k][1]:4.1f}%)")
    for k, label in (("mean", "平均誤差[mm]"), ("median", "中央値[mm]"), ("max", "最大誤差[mm]")):
        f2 = f"{s2[k]:.3f}" if s2[k] is not None else "-"
        f3 = f"{s3[k]:.3f}" if s3[k] is not None else "-"
        print(f"{label:<20} {f2:>14} {f3:>14}")

    improved = [r for r in out_rows if isinstance(r["delta_abs_error_mm"], float) and r["delta_abs_error_mm"] < 0]
    worsened = [r for r in out_rows if isinstance(r["delta_abs_error_mm"], float) and r["delta_abs_error_mm"] > 0]
    print(f"\n改善したケース: {len(improved)} / 悪化したケース: {len(worsened)}")

    print(f"\n突き合わせCSV: {out_csv}")

    save = sam3_run / "comparison_vs_sam2_summary.json"
    save.write_text(
        json.dumps({"sam2_run": str(sam2_run), "sam3_run": str(sam3_run),
                    "n_shared": n, "sam2": s2, "sam3": s3,
                    "gt_width_mismatched_cases": mismatched},
                   ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"サマリJSON  : {save}")


if __name__ == "__main__":
    main()
