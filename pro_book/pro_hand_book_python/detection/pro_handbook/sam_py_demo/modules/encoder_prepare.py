#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM encoder 前の前処理（画像→1024正方形パディング→エンコーダ入力テンソル）を
安全に“ひとかたまり”で扱うためのユーティリティ。

主なエントリポイント:
- preprocess_for_encoder(image, enc_input, target_length=1024)
    PIL.Image または ndarray(RGB,HWC) を受け取り、
    ・1024 パディング済み画像
    ・エンコーダ ONNX への feed 配列（HWC/NHWC/CHW/NCHW 自動対応）
    ・レイアウト文字列 "HWC"/"NHWC"/"CHW"/"NCHW"
    ・原寸/縮小後サイズ、resizer などのメタ情報
  を PreEncPack で返します。

- PreEncPack.coords_to_1024(coords_xy)
    原画像座標 (x,y) 群を 1024 空間へスケール（左上パディング前提）。

使い方（既存スクリプト側）:
    from sam_pre import preprocess_for_encoder
    pack = preprocess_for_encoder(base_img, encoder_sess.get_inputs()[0], 1024)
    emb = encoder_sess.run(None, {pack.enc_name: pack.feed_img})[0]
    # run_point() 内では pack.coords_to_1024(...)

注意:
- HWC/NHWC 入力のエンコーダには "正規化なし" でそのまま渡し、
  CHW/NCHW の場合は (img-mean)/std → CHW/NCHW を自動適用します。
- 画像は RGB 前提です（PIL なら .convert("RGB") 済み）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import cv2
from PIL import Image

# 公式実装の前処理と揃えた定数
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class ResizeLongestSide:
    """長辺=target_length、左上に貼って右と下だけパディング（公式と同じ）"""
    def __init__(self, target_length: int):
        self.target_length = target_length

    @staticmethod
    def _get_preprocess_shape(oldh: int, oldw: int, long_side: int):
        scale = long_side / max(oldh, oldw)
        newh = int(oldh * scale)  # floor
        neww = int(oldw * scale)  # floor
        return newh, neww, scale

    def apply_image(self, img: np.ndarray):
        h, w = img.shape[:2]
        newh, neww, _ = self._get_preprocess_shape(h, w, self.target_length)
        rsz = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = self.target_length - newh, self.target_length - neww
        pad = cv2.copyMakeBorder(rsz, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return pad, (h, w, newh, neww)

    def apply_coords(self, coords: np.ndarray, orig_hw: Tuple[int, int]) -> np.ndarray:
        oh, ow = orig_hw
        newh, neww, _ = self._get_preprocess_shape(oh, ow, self.target_length)
        scale = np.array([neww / ow, newh / oh], dtype=np.float32)
        return coords.astype(np.float32) * scale


def _normalize_to_chw(x: np.ndarray) -> np.ndarray:
    """(H,W,3) RGB を (3,H,W) にして (x-mean)/std を適用"""
    x = (x - PIXEL_MEAN) / PIXEL_STD
    return x.transpose(2, 0, 1).astype(np.float32)


def prepare_encoder_feed(padded_rgb: np.ndarray, enc_input) -> tuple[np.ndarray, str]:
    """
    エンコーダ ONNX の最初の入力（enc_input）の shape を見て、
    HWC/NHWC/CHW/NCHW のいずれかを判定し、適切な配列とレイアウト文字列を返す。

    HWC/NHWC の場合は正規化なし、CHW/NCHW の場合は (img-mean)/std を適用。

    Returns
    -------
    feed_array : np.ndarray
    layout     : str ("HWC" / "NHWC" / "CHW" / "NCHW")
    """
    shape = getattr(enc_input, "shape", None) or []
    rank = len(shape)

    img = padded_rgb.astype(np.float32, copy=False)

    if rank == 3:
        if shape[-1] == 3:       # H, W, 3  (HWC)
            return img, "HWC"
        if shape[0] == 3:        # 3, H, W  (CHW)
            return _normalize_to_chw(img), "CHW"
        # fallback: HWC
        return img, "HWC"

    if rank == 4:
        if shape[-1] == 3:       # N, H, W, 3  (NHWC)
            return img[None, ...], "NHWC"
        if shape[1] == 3:        # N, 3, H, W  (NCHW)
            return _normalize_to_chw(img)[None, ...], "NCHW"
        # fallback: NHWC
        return img[None, ...], "NHWC"

    # shape 未定義など → HWC にフォールバック
    return img, "HWC"


@dataclass
class PreEncPack:
    """前処理結果を 1 つに束ねた入出力・メタ情報。"""
    enc_name: str                  # エンコーダ ONNX の最初の入力名
    feed_img: np.ndarray           # エンコーダへ渡す配列（レイアウト済み）
    layout: str                    # "HWC"/"NHWC"/"CHW"/"NCHW"
    padded_rgb: np.ndarray         # 1024 左上パディング済み画像 (H,W,3)
    orig_hw: Tuple[int, int]       # (oh, ow)
    new_hw: Tuple[int, int]        # (nh, nw)
    resizer: ResizeLongestSide     # 同じ計算式を使い回したいときに
    target_length: int = 1024

    def coords_to_1024(self, coords_xy: np.ndarray) -> np.ndarray:
        """原画像座標群 (N,2) を 1024 空間へスケール（左上パディング想定）。"""
        return self.resizer.apply_coords(coords_xy, self.orig_hw)


ImageLike = Union[Image.Image, np.ndarray]

#このモジュールのメイン関数はこれやね
def preprocess_for_encoder(image: ImageLike, enc_input, target_length: int = 1024) -> PreEncPack:
    """
    画像を SAM エンコーダに食わせる前の処理を一括で行い、PreEncPack として返す。

    Parameters
    ----------
    image : PIL.Image | np.ndarray
        RGB 前提。PIL の場合は内部で .convert("RGB") します。
    enc_input : onnxruntime.NodeArg
        encoder_sess.get_inputs()[0] で取得した入力メタ。（name, shape を使用）
    target_length : int
        長辺をこのサイズに合わせて左上パディング（既定: 1024）。

    Returns
    -------
    PreEncPack
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))  # HWC, RGB
    else:
        # ndarray の場合: HWC,RGB を期待。必要ならここで変換を追加
        img_rgb = image
        if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
            raise ValueError("`image` must be RGB, shape=(H,W,3)")

    resizer = ResizeLongestSide(target_length)
    padded, (oh, ow, nh, nw) = resizer.apply_image(img_rgb)

    feed_img, layout = prepare_encoder_feed(padded, enc_input)
    enc_name = getattr(enc_input, "name", None) or "image"

    return PreEncPack(
        enc_name=enc_name,
        feed_img=feed_img,
        layout=layout,
        padded_rgb=padded,
        orig_hw=(oh, ow),
        new_hw=(nh, nw),
        resizer=resizer,
        target_length=target_length,
    )
