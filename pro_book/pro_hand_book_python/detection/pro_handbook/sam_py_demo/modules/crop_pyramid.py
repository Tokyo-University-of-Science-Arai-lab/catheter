#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import cv2
from PIL import Image


__all__ = [
"gen_square_crops",
"offset_full_mask",
"bbox_add_offset",
]


def gen_square_crops(oh: int, ow: int, n_layers: int, overlap: float) -> Iterator[Tuple[int,int,int,int,int]]:
    """Generate square crops as a pyramid.
    Yields: (x0, y0, w, h, layer_index) for layer = 1..n_layers
    - Base window = short_side / (2**layer)
    - Overlap is applied per layer (0..0.5 recommended)
    """
    short = int(min(oh, ow))
    overlap = float(np.clip(overlap, 0.0, 0.5))
    for i in range(1, max(0, int(n_layers)) + 1):
        win = max(32, int(short / (2 ** i)))
        step = max(1, int(win * (1.0 - overlap)))
        xs = list(range(0, max(1, ow - win + 1), step))
        ys = list(range(0, max(1, oh - win + 1), step))
        if xs[-1] != ow - win:
            xs.append(ow - win)
        if ys[-1] != oh - win:
            ys.append(oh - win)
        for y0 in ys:
            for x0 in xs:
                yield (x0, y0, win, win, i)


def offset_full_mask(tile_mask_bin, full_hw: tuple[int,int], x0: int, y0: int):
    """Paste a tile mask (bool/0-1) into full-size canvas with offset."""
    ohF, owF = full_hw
    m = (tile_mask_bin.astype(np.uint8) > 0).astype(np.uint8)
    full = np.zeros((ohF, owF), dtype=np.uint8)
    th, tw = m.shape
    y1, x1 = min(ohF, y0 + th), min(owF, x0 + tw)
    full[y0:y1, x0:x1] = m[: (y1 - y0), : (x1 - x0)]
    return (full > 0)


def bbox_add_offset(xywh: tuple[int,int,int,int], x0: int, y0: int) -> tuple[int,int,int,int]:
    x, y, w, h = xywh
    return (int(x + x0), int(y + y0), int(w), int(h))