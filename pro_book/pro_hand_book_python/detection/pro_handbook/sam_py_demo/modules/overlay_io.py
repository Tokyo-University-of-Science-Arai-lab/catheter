# modules/overlay_io.py
from __future__ import annotations
from pathlib import Path
from typing import List, Iterable
import numpy as np
from PIL import Image
import cv2

# 元コードと同一のパレット
PALETTE_30 = [
    (255, 0, 0), (0, 180, 255), (0, 200, 0), (255, 165, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 105, 180),
    (160, 82, 45), (0, 128, 128), (220, 20, 60), (154, 205, 50),
    (75, 0, 130), (255, 215, 0), (0, 191, 255), (124, 252, 0),
    (255, 69, 0), (199, 21, 133), (70, 130, 180), (186, 85, 211),
    (46, 139, 87), (210, 105, 30), (123, 104, 238), (100, 149, 237),
    (255, 20, 147), (47, 79, 79), (189, 183, 107), (72, 209, 204),
    (240, 128, 128), (0, 250, 154),
]

# 任意依存: ID描画があれば使う（無ければ黙ってスキップ）
try:
    from .ids_overlay import draw_mask_ids_on_overlay
except Exception:
    draw_mask_ids_on_overlay = None  # type: ignore

def _save_points_and_overlay(
    base_img: Image.Image,
    masks: List[np.ndarray],
    out_dir: Path | str,
    prefix: str,
    draw_ids: bool = False,
    rotate_after_nms_only: bool = True,
) -> None:
    """points.npy と overlay.jpg を保存。元コードと同一仕様。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # points.npy（外部輪郭のみ）
    all_points = []
    for m in masks:
        m = np.array(m)
        if m.ndim < 2:
            continue

        if m.ndim > 2:
            m = m[:, :, 0]

        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = np.concatenate(cnts, axis=0) if cnts else np.zeros((0, 1, 2), np.int32)
        all_points.append(pts.squeeze(1))
    np.save(out_dir / f"{prefix}_points.npy", np.array(all_points, dtype=object))

    # overlay（半透明合成、必要ならID描画）
    overlay = base_img.convert("RGBA")
    for i, m in enumerate(masks):
        m = np.array(m)
        if m.ndim < 2:
            continue 

        if m.ndim > 2:
            m = m[:, :, 0]

        alpha = Image.fromarray((m.astype(np.uint8) * 140))
        if alpha.size != base_img.size:
            alpha = alpha.resize(base_img.size, resample=Image.NEAREST)
        tint  = Image.new("RGBA", base_img.size, (*PALETTE_30[i % len(PALETTE_30)], 0))
        tint.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, tint)
    overlay = overlay.convert("RGB")
    if draw_ids and draw_mask_ids_on_overlay is not None:
        overlay = draw_mask_ids_on_overlay(overlay, masks, start_index=1)

    # after_nms のときだけ 180° 回転（既存仕様）
    if rotate_after_nms_only and prefix.startswith("after_nms"):
        overlay = overlay.rotate(180, expand=False)

    overlay.save(out_dir / f"{prefix}_overlay.jpg", quality=92)

def _render_overlay_bgr(
    base_img: Image.Image,
    masks: Iterable[np.ndarray],
    draw_ids: bool = False,
) -> np.ndarray:

    overlay = base_img.convert("RGBA")

    for i, m in enumerate(masks):
        m = np.array(m)

        if m.ndim < 2:
            continue

        if m.ndim > 2:
            m = m[:, :, 0]

        alpha = Image.fromarray((m.astype(np.uint8) * 140))

        # 🔥 これが今回の本質（絶対必要）
        if alpha.size != base_img.size:
            alpha = alpha.resize(base_img.size, resample=Image.NEAREST)

        tint = Image.new("RGBA", base_img.size, (*PALETTE_30[i % len(PALETTE_30)], 0))
        tint.putalpha(alpha)

        overlay = Image.alpha_composite(overlay, tint)

    overlay = overlay.convert("RGB")

    if draw_ids and draw_mask_ids_on_overlay is not None:
        overlay = draw_mask_ids_on_overlay(overlay, list(masks), start_index=1)

    rgb = np.array(overlay)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)