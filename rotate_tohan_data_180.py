#!/usr/bin/env python3
"""Rotate every PNG in captures/TOHAN_data by 180 degrees in place."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise ValueError(f"PNGではありません: {path}")
        return image.copy()


def rotate_all(destination_dir: Path) -> int:
    destination_dir = destination_dir.resolve()
    if not destination_dir.is_dir():
        raise NotADirectoryError(f"対象ディレクトリが見つかりません: {destination_dir}")

    files = sorted(
        path
        for path in destination_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not files:
        print(f"PNG画像がありません: {destination_dir}")
        return 0

    # Fail before making changes if any input image cannot be decoded as PNG.
    for path in files:
        load_image(path)

    rotated_count = 0
    for path in files:
        original = load_image(path)
        rotated = original.transpose(Image.Transpose.ROTATE_180)
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                suffix=".tmp.png",
                dir=destination_dir,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            rotated.save(temporary_path, format="PNG")
            saved = load_image(temporary_path)
            if (
                saved.mode != rotated.mode
                or saved.size != rotated.size
                or saved.tobytes() != rotated.tobytes()
            ):
                raise RuntimeError(f"保存後の画素検証に失敗しました: {path.name}")

            os.chmod(temporary_path, path.stat().st_mode & 0o777)
            os.replace(temporary_path, path)
            temporary_path = None
            rotated_count += 1
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    print("処理結果:")
    print(f"  180度回転成功数: {rotated_count}")
    print(f"  対象ディレクトリ: {destination_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TOHAN_data内の全PNG画像を180度回転します。"
    )
    parser.add_argument(
        "destination_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "captures" / "TOHAN_data",
        help="TOHAN_dataディレクトリ",
    )
    args = parser.parse_args()
    return rotate_all(args.destination_dir)


if __name__ == "__main__":
    raise SystemExit(main())
