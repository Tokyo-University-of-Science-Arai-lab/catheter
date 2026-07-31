#!/usr/bin/env python3
"""
offline_pointcloud_debug_SAM3.py の結果から final.png だけを集める。

各品目のフォルダに散らばっている final.png を、書籍名（display_name）を
ファイル名にして1箇所にまとめる。選ばれた箱が正しいかを目視で確認するため。

出力: <run_dir>/shot<N>/final_images/<display_name>.png

display_name が重複する品目（TransForm が2つある等）は、
区別できるよう型番を後ろに付ける: TransForm_ESC0407.png

使い方:
  python collect_final_images.py                 # 最新の実行結果を対象にする
  python collect_final_images.py <run_dir>       # 実行結果を指定する
"""

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_BASE_DIR = BASE_DIR / "captures" / "5shot_catheter_offline"
MASTER_JSON = BASE_DIR / "master_20260216.json"

SOURCE_NAME = "final.png"
OUT_DIR_NAME = "final_images"


def latest_run_dir(base_dir: Path) -> Path:
    runs = [p for p in base_dir.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    if not runs:
        raise FileNotFoundError(f"results.csv を持つ実行結果がありません: {base_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_display_names(master_json: Path) -> dict[str, str]:
    data = json.loads(master_json.read_text(encoding="utf-8"))
    return {
        item["book_name"]: item.get("display_name") or item["book_name"]
        for item in data
    }


def safe_file_name(name: str) -> str:
    """ファイル名に使えない文字だけを置き換える（日本語や® はそのまま残す）。"""
    for ch in '/\\:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip() or "unnamed"


def main():
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else latest_run_dir(OFFLINE_BASE_DIR)
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv がありません: {results_csv}")

    display_names = load_display_names(MASTER_JSON)

    with open(results_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # display_name が重複する品目を洗い出す（型番を足して区別するため）
    name_counts = Counter(display_names.get(r["book_name"], r["book_name"]) for r in rows)
    shots = sorted({r["shot_id"] for r in rows}, key=int)
    duplicated = {
        name for name, c in name_counts.items() if c > len(shots)
    }

    by_shot: dict[str, list[dict]] = {}
    for r in rows:
        by_shot.setdefault(r["shot_id"], []).append(r)

    print(f"対象: {run_dir}")
    if duplicated:
        print(f"display_name が重複する品目（型番を付けて区別）: {sorted(duplicated)}")

    total_copied, total_missing = 0, []

    for shot_id in shots:
        out_dir = run_dir / f"shot{shot_id}" / OUT_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)

        # 前回の残りが紛れないよう、このフォルダのpngは作り直す
        for old in out_dir.glob("*.png"):
            old.unlink()

        copied = 0
        for r in sorted(by_shot[shot_id], key=lambda r: int(r["master_index"])):
            src = Path(r["run_shot_dir"]) / SOURCE_NAME
            if not src.exists():
                total_missing.append(str(src))
                continue

            name = display_names.get(r["book_name"], r["book_name"])
            if name in duplicated:
                name = f"{name}_{r['book_name']}"
            shutil.copy2(src, out_dir / f"{safe_file_name(name)}.png")
            copied += 1

        total_copied += copied
        print(f"  shot{shot_id}/{OUT_DIR_NAME}/  ({copied} 枚)")

    print(f"合計 {total_copied} 枚をコピーしました。")
    if total_missing:
        print(f"{SOURCE_NAME} が無かった試行: {len(total_missing)} 件")
        for p in total_missing[:5]:
            print(f"  {p}")


if __name__ == "__main__":
    main()
