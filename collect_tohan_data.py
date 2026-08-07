#!/usr/bin/env python3
"""Collect timestamped after_init_rgb.png files without overwriting outputs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


DIRECTORY_PATTERN = re.compile(r"^(\d{8})_(\d{6})$")
START = datetime(2026, 4, 7, 0, 0, 0)
END = datetime(2026, 4, 17, 23, 59, 59)
SOURCE_NAME = "after_init_rgb.png"
DESTINATION_NAME = "TOHAN_data"


def target_directories(captures_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for path in captures_dir.iterdir():
        if not path.is_dir():
            continue

        match = DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None:
            continue

        try:
            timestamp = datetime.strptime(path.name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue

        if START <= timestamp <= END:
            targets.append(path)

    return sorted(targets, key=lambda path: path.name)


def copy_exclusive(source: Path, destination: Path) -> None:
    """Copy source to a newly created destination, failing if it already exists."""
    file_descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        source.stat().st_mode & 0o777,
    )
    try:
        with os.fdopen(file_descriptor, "wb") as destination_file:
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, destination_file)
        shutil.copystat(source, destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def collect(captures_dir: Path) -> int:
    captures_dir = captures_dir.resolve()
    if not captures_dir.is_dir():
        raise NotADirectoryError(f"capturesディレクトリが見つかりません: {captures_dir}")

    destination_dir = captures_dir / DESTINATION_NAME
    destination_dir.mkdir(exist_ok=True)
    if not destination_dir.is_dir():
        raise NotADirectoryError(f"コピー先がディレクトリではありません: {destination_dir}")

    targets = target_directories(captures_dir)
    missing: list[str] = []
    duplicates: list[str] = []
    copied: list[str] = []

    for directory in targets:
        source = directory / SOURCE_NAME
        destination = destination_dir / f"{directory.name}.png"

        if not source.is_file():
            missing.append(directory.name)
            continue

        try:
            copy_exclusive(source, destination)
        except FileExistsError:
            duplicates.append(destination.name)
        else:
            copied.append(destination.name)

    if missing:
        print("after_init_rgb.png未検出ディレクトリ:")
        for name in missing:
            print(f"  {name}")

    if duplicates:
        print("重複（上書きせずスキップ）:")
        for name in duplicates:
            print(f"  {name}")

    print("処理結果:")
    print(f"  対象ディレクトリ数: {len(targets)}")
    print(f"  コピー成功数: {len(copied)}")
    print(f"  未検出数: {len(missing)}")
    print(f"  重複数: {len(duplicates)}")
    print(f"  コピー先: {destination_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="指定期間のafter_init_rgb.pngをTOHAN_dataへ安全にコピーします。"
    )
    parser.add_argument(
        "captures_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "captures",
        help="capturesディレクトリ（既定: このスクリプトと同じ場所のcaptures）",
    )
    args = parser.parse_args()
    return collect(args.captures_dir)


if __name__ == "__main__":
    raise SystemExit(main())
