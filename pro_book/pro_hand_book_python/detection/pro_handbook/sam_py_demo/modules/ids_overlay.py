# modules/ids_overlay.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

def draw_mask_ids_on_overlay(
        overlay_pil: Image.Image,
        masks,
        start_index: int = 1,
        color=(255, 255, 255),      # 文字色
        outline=(0, 0, 0),          # 縁取り
        bottom_margin_px: int = 250, # 互換用（未使用）
        mode: str = "top_mid",      # "top_mid" / "axis_bottom" / "centroid"
    ) -> Image.Image:

    img = np.array(overlay_pil)                  # RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)   # → BGR

    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = max(0.8, min(2.2, min(H, W) / 550.0))

    for idx, m in enumerate(masks, start=start_index):
        u8 = (m.astype(np.uint8) * 255)
        cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        # ---- アンカー決定 ----
        if mode == "top_mid":
            ys = cnt[:, 0, 1]
            xs = cnt[:, 0, 0]
            ymin = int(ys.min())
            band = max(2, int(0.006 * min(H, W)))   # 上端から細い帯
            top_idx = np.where(ys <= ymin + band)[0]
            if len(top_idx) >= 2:
                xs_top = xs[top_idx]
                ys_top = ys[top_idx]
                x_anchor = int(round(0.5 * (xs_top.min() + xs_top.max())))
                y_anchor = int(ys_top.min())
            else:
                x0b, y0b, w0b, h0b = cv2.boundingRect(cnt)
                x_anchor = int(x0b + w0b / 2)
                y_anchor = int(y0b)

        elif mode == "axis_bottom":
            (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
            if w < h: ang += 90.0
            x_anchor = int(np.clip(round(cx), 0, W - 1))
            y_anchor = int(np.clip(int(cnt[:, 0, 1].max()), 0, H - 1))

        else:  # "centroid"
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                x_anchor = int(M["m10"] / M["m00"])
                y_anchor = int(M["m01"] / M["m00"])
            else:
                x0b, y0b, w0b, h0b = cv2.boundingRect(cnt)
                x_anchor = x0b + w0b // 2
                y_anchor = y0b + h0b // 2

        # ---- 文字サイズ決定（AABBのx幅を使用）----
        xs = cnt[:, 0, 0]
        x_len = max(1, int(xs.max()) - int(xs.min()) + 1)
        r = float(x_len) / float(W)           # 相対幅 0〜1
        scale = base_scale * (0.7 + 1.2 * r)

        th_out = max(3, int(round(3.0 * scale)))   # 縁取り
        th_in  = max(2, int(round(2.0 * scale)))   # 本文字

        text = str(idx)
        (text_w, text_h), base = cv2.getTextSize(text, font, scale, th_in)

        # ---- テキスト画像を作成 → 180°回転 → トリミング ----
        pad = max(th_out + 40, 20)
        buf_h = text_h + 2 * pad
        buf_w = text_w + 2 * pad
        text_img = np.zeros((buf_h, buf_w, 3), dtype=np.uint8)
        org = (pad, pad + text_h - base)
        cv2.putText(text_img, text, org, font, scale, outline, th_out, lineType=cv2.LINE_AA)
        cv2.putText(text_img, text, org, font, scale, color,   th_in,  lineType=cv2.LINE_AA)

        text_img = cv2.rotate(text_img, cv2.ROTATE_180)

        mask_gray = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        nz = cv2.findNonZero(mask_bin)
        if nz is None:
            continue
        x_b, y_b, w_b, h_b = cv2.boundingRect(nz)
        glyph = text_img[y_b:y_b+h_b, x_b:x_b+w_b]
        glyph_mask = mask_bin[y_b:y_b+h_b, x_b:x_b+w_b]

        # ---- 貼り付け位置：アンカーに“グリフ中心”を一致 ----
        x0 = int(np.clip(round(x_anchor - w_b / 2), 0, W - w_b))
        y0 = int(np.clip(round(y_anchor - h_b / 2), 0, H - h_b))

        roi = img[y0:y0+h_b, x0:x0+w_b]
        mask_inv = cv2.bitwise_not(glyph_mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(glyph, glyph, mask=glyph_mask)
        img[y0:y0+h_b, x0:x0+w_b] = cv2.add(bg, fg)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)
