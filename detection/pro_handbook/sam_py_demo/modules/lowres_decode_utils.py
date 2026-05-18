#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import cv2


__all__ = [
"postprocess_like_sam",
"stability_score_from_lowres",
"candidate_from_lowres",
"lowres_iou_binary",
]


def _bbox_from_mask_u8(m_u8: np.ndarray) -> tuple[int,int,int,int]:
    x, y, w, h = cv2.boundingRect(m_u8)
    return int(x), int(y), int(w), int(h)


def _to_prob_0_1(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    xmin, xmax = float(x.min()), float(x.max())
    if xmin >= -1e-6 and xmax <= 1.0 + 1e-6:
        return np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def postprocess_like_sam(lowres_256: np.ndarray, orig_hw: Tuple[int,int], new_hw: Tuple[int,int],
    target_len: int = 1024, is_logits: bool = True) -> np.ndarray:
    oh, ow = orig_hw
    nh, nw = new_hw
    src = lowres_256.astype(np.float32)
    # upsample 256 -> 1024 -> crop back to resized (nh,nw) -> upsample to orig (ow,oh)
    up_1024 = cv2.resize(src, (target_len, target_len), interpolation=cv2.INTER_LINEAR)
    cropped = up_1024[:nh, :nw]
    up_full = cv2.resize(cropped, (ow, oh), interpolation=cv2.INTER_LINEAR)
    thr = 0.0 if is_logits else 0.5
    m_bin = (up_full > thr)
    return (m_bin.astype(np.uint8) * 255)


def stability_score_from_lowres(lowres_256: np.ndarray, *, delta: float = 0.05, is_logits: bool = True) -> float:
    arr = lowres_256.astype(np.float32)
    if is_logits:
        hi = arr > (delta)
        lo = arr > (-delta)
    else:
        hi = arr > (0.5 + delta)
        lo = arr > (0.5 - delta)
    inter = np.logical_and(hi, lo).sum()
    uni = np.logical_or(hi, lo).sum()
    return 0.0 if uni == 0 else float(inter) / float(uni)


def candidate_from_lowres(lowres_256: np.ndarray, *, score: float, orig_hw: Tuple[int,int], new_hw: Tuple[int,int], is_logits: bool) -> Dict[str, Any]:
    m_u8 = postprocess_like_sam(lowres_256, orig_hw, new_hw, is_logits=is_logits)
    mask_bin = (m_u8 > 0)
    area = int(mask_bin.sum())
    bbox = _bbox_from_mask_u8(m_u8)
    return {
        "mask": mask_bin,
        "bbox": bbox, # (x,y,w,h)
        "area": area,
        "score": float(score),
        "_m_u8": m_u8,
        }


def lowres_iou_binary(a: np.ndarray, b: np.ndarray, is_logits: bool, side: int = 128) -> float:
    A = cv2.resize(a.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    B = cv2.resize(b.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
    mA = A > (0.0 if is_logits else 0.5)
    mB = B > (0.0 if is_logits else 0.5)
    inter = np.logical_and(mA, mB).sum()
    uni = np.logical_or(mA, mB).sum()
    return 0.0 if uni == 0 else float(inter) / float(uni)