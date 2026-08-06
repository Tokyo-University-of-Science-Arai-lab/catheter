#!/usr/bin/env python3
"""100回試験の run から OCR結果と similarity_scores.json を1箇所に集める。

各ケースのフォルダに散らばっている OCR とマスク照合の情報を、
まとめて見比べられる形に並べ直す。認識の中身は一切変更しない。

使い方（既定は 100test_offline の最新run）:
    python collect_ocr_similarity.py
    python collect_ocr_similarity.py --run captures/100test_offline/SAM3_20260731_200810
    python collect_ocr_similarity.py --with-overlay      # 目視用のPNGも集める（重い）
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RUN_BASE = BASE_DIR / "captures" / "100test_offline"

# 集める対象。ケースフォルダ直下にある想定。
FILES = {
    "ocr_result": "ocr_result.json",
    "similarity_scores": "similarity_scores.json",
    "ocr_runtime_info": "ocr_runtime_info.json",
}
OVERLAY_FILES = {
    "ocr_overlay": "ocr_overlay.png",
}
AXIS_OVERLAY_FILES = {
    "ocr_axis_overlay": "ocr_axis_overlay.png",
}


def latest_run() -> Path:
    runs = [p for p in RUN_BASE.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"run が見つかりません: {RUN_BASE}")
    return sorted(runs, key=lambda p: p.name)[-1]


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def case_dirs(run_dir: Path):
    """<run>/<数字> のフォルダを番号順に返す。"""
    dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def load_results(run_dir: Path) -> dict[int, dict]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        matches = sorted(run_dir.glob("book_width_eval_results_*.csv"))
        if not matches:
            return {}
        csv_path = matches[-1]
    with open(csv_path, encoding="utf-8-sig") as f:
        return {int(r["test_index"]): r for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser(
        description="Collect OCR results and similarity scores from a 100-case run."
    )
    parser.add_argument("--run", type=str, default=None, help="対象のrunディレクトリ（既定: 最新）")
    parser.add_argument("--out", type=str, default=None, help="出力先（既定: <run>/_ocr_similarity）")
    parser.add_argument(
        "--with-overlay", action="store_true",
        help="ocr_overlay.png も集める（1ケースあたり約1MB）",
    )
    parser.add_argument(
        "--with-axis-overlay", action="store_true",
        help="軸確認用の ocr_axis_overlay.png も集める（1ケースあたり約1.3MB）",
    )
    args = parser.parse_args()

    run_dir = Path(args.run).resolve() if args.run else latest_run()
    out_dir = Path(args.out).resolve() if args.out else run_dir / "_ocr_similarity"

    results = load_results(run_dir)
    cases = case_dirs(run_dir)
    if not cases:
        raise FileNotFoundError(f"ケースフォルダが見つかりません: {run_dir}")

    print(f"run     : {run_dir}")
    print(f"ケース数: {len(cases)}")
    print(f"出力先  : {out_dir}\n")

    targets = dict(FILES)
    if args.with_overlay:
        targets.update(OVERLAY_FILES)
    if args.with_axis_overlay:
        targets.update(AXIS_OVERLAY_FILES)

    for key in targets:
        (out_dir / key).mkdir(parents=True, exist_ok=True)

    ocr_rows = []      # ケースごとのOCR概要
    text_rows = []     # OCRが読んだ1行ずつ
    score_rows = []    # マスクごとの照合スコア
    missing = []

    for case in cases:
        idx = int(case.name)
        res = results.get(idx, {})
        book_name = res.get("book_name", "")
        stem = f"{idx:03d}_{book_name}" if book_name else f"{idx:03d}"

        for key, filename in targets.items():
            src = case / filename
            if not src.exists():
                missing.append(str(src))
                continue
            shutil.copy2(src, out_dir / key / f"{stem}{src.suffix}")

        # --- OCR ---
        ocr_path = case / FILES["ocr_result"]
        n_texts = 0
        if ocr_path.exists():
            ocr = load_json(ocr_path)
            texts = ocr.get("rec_texts", [])
            scores = ocr.get("rec_scores", [])
            n_texts = len(texts)
            for i, text in enumerate(texts):
                text_rows.append({
                    "test_index": idx,
                    "query": book_name,
                    "line_no": i,
                    "text": text,
                    "rec_score": scores[i] if i < len(scores) else "",
                })

        # --- 照合スコア ---
        sim_path = case / FILES["similarity_scores"]
        top = {}
        n_over = 0
        threshold = ""
        if sim_path.exists():
            sim = load_json(sim_path)
            threshold = sim.get("threshold", "")
            entries = sim.get("scores", []) or []
            for e in entries:
                score_rows.append({
                    "test_index": idx,
                    "query": sim.get("query", book_name),
                    "mask": e.get("name", ""),
                    "score": e.get("score", ""),
                    "over_threshold": (
                        e.get("score", 0) >= threshold
                        if isinstance(threshold, (int, float)) else ""
                    ),
                    "text": e.get("text", ""),
                })
            if isinstance(threshold, (int, float)):
                n_over = sum(1 for e in entries if e.get("score", 0) >= threshold)
            if entries:
                top = max(entries, key=lambda e: e.get("score", 0))

        ocr_rows.append({
            "test_index": idx,
            "query": book_name,
            "gt_width_mm": res.get("gt_book_width_mm", ""),
            "pred_width_mm": res.get("pred_book_width_mm", ""),
            "abs_error_mm": res.get("abs_error_mm", ""),
            "status": res.get("status", ""),
            "ocr_line_count": n_texts,
            "threshold": threshold,
            "n_masks_scored": sum(1 for r in score_rows if r["test_index"] == idx),
            "n_masks_over_threshold": n_over,
            "top_mask": top.get("name", ""),
            "top_score": top.get("score", ""),
            "top_mask_text": top.get("text", ""),
        })

    def write_csv(path: Path, rows):
        if not rows:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {path.name:<26} {len(rows)} 行")

    print("一覧CSV:")
    write_csv(out_dir / "case_overview.csv", ocr_rows)
    write_csv(out_dir / "ocr_texts.csv", text_rows)
    write_csv(out_dir / "similarity_scores.csv", score_rows)

    index = {
        "run_dir": str(run_dir),
        "case_count": len(cases),
        "collected_files": targets,
        "with_overlay": bool(args.with_overlay),
        "with_axis_overlay": bool(args.with_axis_overlay),
        "naming": "<test_index 3桁>_<query>.<拡張子>",
        "csv": {
            "case_overview.csv": "ケース1行。OCR行数・最高スコアのマスク・推定幅と誤差",
            "ocr_texts.csv": "OCRが読んだ文字を1行ずつ。rec_score は認識信頼度",
            "similarity_scores.csv": "マスクごとの照合スコア。over_threshold で足切りを確認できる",
        },
        "missing_files": missing,
    }
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if missing:
        print(f"\n⚠ 見つからなかったファイル: {len(missing)} 件（index.json に記録）")

    print(f"\n完了: {out_dir}")


if __name__ == "__main__":
    main()
