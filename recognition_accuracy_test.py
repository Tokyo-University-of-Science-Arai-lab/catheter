#!/usr/bin/env python3
"""
認識精度テストスクリプト
master_20260216.json 内の全認識対象を順番に認識し、精度と開口幅を記録する。
マニピュレータは動かさない。
"""

import os
import sys

# onnxruntime-gpu が cuDNN を見つけられるよう LD_LIBRARY_PATH を設定して再起動
_cudnn_lib = (
    "/home/book/pro_book_SAM3/pro_hand_book_python"
    "/.pro_hand_book_fixed/lib/python3.10/site-packages/nvidia/cudnn/lib"
)
if _cudnn_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _cudnn_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detection.pro_handbook.sam_py_demo.get_book_points import run_capture_and_pca

MASTER_JSON = Path(__file__).parent / "master_20260216.json"
RESULT_DIR  = Path(__file__).parent / "recognition_accuracy_results"


def main():
    with open(MASTER_JSON, encoding="utf-8") as f:
        books = json.load(f)

    total = len(books)
    results = []
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  認識精度テスト  全 {total} 件")
    print("=" * 65)

    for i, book in enumerate(books, 1):
        book_name = book["book_name"]
        isbn      = book.get("ISBN_number", "N/A")

        print(f"\n[{i}/{total}] 認識対象名 : {book_name}")
        print(f"       ISBN       : {isbn}")

        roll, p_xmax, book_width, shot_dir = run_capture_and_pca(query=book_name)

        success = p_xmax is not None
        width_str = f"{book_width:.2f} mm" if success else "N/A"

        print(f"  結果     : {'✔  成功' if success else '✗  失敗 (p_xmax が None)'}")
        print(f"  書籍幅   : {width_str}")

        results.append({
            "index"         : i,
            "book_name"     : book_name,
            "isbn"          : isbn,
            "success"       : success,
            "book_width_mm" : float(book_width) if book_width is not None else None,
            "shot_dir"      : str(shot_dir) if shot_dir is not None else None,
        })

    # ===== サマリ =====
    n_success = sum(1 for r in results if r["success"])
    accuracy  = 100.0 * n_success / total if total > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"  認識精度 : {n_success} / {total} ({accuracy:.1f}%)")
    print("=" * 65)

    col_name = 40
    col_res  = 6
    col_bw   = 10

    header = f"{'認識対象名':<{col_name}} {'結果':<{col_res}} {'書籍幅':>{col_bw}}"
    print(header)
    print("-" * 58)

    for r in results:
        name   = r["book_name"][:col_name-2] + ".." if len(r["book_name"]) > col_name else r["book_name"]
        status = "✔" if r["success"] else "✗"
        bw_str = f"{r['book_width_mm']:.1f} mm" if r["book_width_mm"] is not None else "N/A"
        print(f"{name:<{col_name}} {status:<{col_res}} {bw_str:>{col_bw}}")

    # ===== 結果保存 =====
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = RESULT_DIR / f"result_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"accuracy_pct": accuracy, "n_success": n_success, "total": total, "results": results},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\n結果を保存しました : {out_path}")


if __name__ == "__main__":
    main()
