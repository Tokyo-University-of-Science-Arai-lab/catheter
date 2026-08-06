#!/usr/bin/env python3
"""
offline_POINTCLOUD_DEBUG_SAM3.py の結果から final.png だけを集める。

captures/100test_offline/<run>/<test_index>/final.png を、品目ごとに
まとまって見えるファイル名にして1箇所にコピーする。選ばれた箱が
正しいか、把持点がずれていないかを目視で確認するため。

出力: <run_dir>/final_images/<master_index>_<book_name>_round<repeat_index>.png

使い方:
  python collect_100test_final_images.py                # 最新の実行結果を対象にする
  python collect_100test_final_images.py <run_dir>       # 実行結果を指定する
"""

import csv
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_BASE_DIR = BASE_DIR / "captures" / "100test_offline"

SOURCE_NAME = "final.png"
OUT_DIR_NAME = "final_images"


def latest_run_dir(base_dir: Path) -> Path:
    runs = [p for p in base_dir.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"results.csv を持つ実行結果がありません: {base_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def safe_file_name(name: str) -> str:
    """ファイル名に使えない文字だけを置き換える。"""
    for ch in '/\\:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip() or "unnamed"


def main():
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_run_dir(OFFLINE_BASE_DIR)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv がありません: {results_csv}")

    with open(results_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out_dir = run_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    # 前回の残りが紛れないよう、このフォルダのpngは作り直す
    for old in out_dir.glob("*.png"):
        old.unlink()

    print(f"対象: {run_dir}")

    copied = 0
    missing = []
    for r in sorted(rows, key=lambda r: (int(r["master_index"]), int(r["repeat_index"]))):
        src = Path(r["run_shot_dir"]) / SOURCE_NAME
        if not src.exists():
            missing.append((r["test_index"], r["book_name"], str(src)))
            continue

        name = safe_file_name(
            f"{int(r['master_index']):02d}_{r['book_name']}_round{r['repeat_index']}"
        )
        shutil.copy2(src, out_dir / f"{name}.png")
        copied += 1

    print(f"{out_dir} に {copied} 枚をコピーしました。")
    if missing:
        print(f"{SOURCE_NAME} が無かった試行（失敗ケースなど）: {len(missing)} 件")
        for test_index, book_name, path in missing:
            print(f"  test_index={test_index} book_name={book_name} -> {path}")


if __name__ == "__main__":
    main()
