#!/usr/bin/env python3
"""IMG_9277+n.JPG を n+1.jpg にリネームする(nは0から始まる整数)。"""
from pathlib import Path

TARGET_DIR = Path("captures/resolution_test/4/4032x3024")
BASE = 9277

for n in range(22):
    src = TARGET_DIR / f"IMG_{BASE + n}.JPG"
    dst = TARGET_DIR / f"{n + 1}.jpg"
    print(f"{src.name} -> {dst.name}")
    src.rename(dst)
