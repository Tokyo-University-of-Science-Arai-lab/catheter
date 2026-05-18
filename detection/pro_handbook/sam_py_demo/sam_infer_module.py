#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Importable SAM batch inference module + CLI (no JSON I/O).

パイプライン:
  SAM推論 → 候補生成（同一点で上位K/近似重複排除） → Box-NMS → Mask-IoUデデュープ
  → 縦向きフィルタ(fold角) → 飛び地除去 → 同軸統合 → zscore
  → 重複削り取り（縦長優先で overlap 部分だけカット） → スムージング

保存（points.npy + overlay.jpg の4段階）:
  1) after_nms
  2) before_smooth
  3) after_smooth  ※番号描画あり（ids_overlayの新レイアウト）
  4) mask{idx}_selected（選択後）

選択マスクから左右端の x のみ（uv_left, uv_right）を返す（左右入替可）
モジュール import / CLI 兼用
出力先デフォルト: ./captures
"""
from __future__ import annotations
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
import cv2

# --- add this near the top of sam_infer_module.py ---
# modules/ ディレクトリを import path に追加
from pathlib import Path as _Path
for p in _Path(__file__).resolve().parents:
    if (p / "modules").is_dir():
        sys.path.insert(0, str(p))
        break
# --- end add ---
# 先頭の依存に追加
from modules.axis_centerline import axis_x_on_centerline

# ===== 依存モジュール =====
from modules.encoder_prepare import preprocess_for_encoder

# NMS / デデュープ（新規: modules/nms.py）
from modules.nms import (
    quality_filter,
    nms_box_greedy_spatial,
    dedup_by_mask_iou_spatial,
)

# 角度フィルタ（fold角ベース） + 主軸情報
try:
    from modules.axis_angle_filter import filter_keep_vertical_by_fold, get_axis_info
except ImportError:
    # 互換レイヤ（古い axis_angle_filter しか無い環境用）
    from modules.axis_angle_filter import filter_by_angle_window as _fallback_angle
    def filter_keep_vertical_by_fold(masks, min_fold_deg=70.0, min_aspect=1.3, min_len_px=0.0):
        kept, infos = _fallback_angle(masks, keep_min_deg=45.0, keep_max_deg=135.0)
        out_m, out_i = [], []
        for m in kept:
            info = get_axis_info(m) if 'get_axis_info' in globals() else None
            if info is None:
                out_m.append(m); out_i.append(None); continue
            if info.L >= min_len_px and (info.L / max(info.W, 1e-6)) >= min_aspect:
                out_m.append(m); out_i.append(info)
        return out_m, out_i
    def get_axis_info(m):  # type: ignore
        return None

# 飛び地除去（最大連結成分のみ残す）
try:
    from modules.island import remove_islands_from_masks
except ImportError:
    # フォールバック実装
    def remove_islands_from_masks(masks: List[np.ndarray]) -> List[np.ndarray]:
        cleaned = []
        for m in masks:
            u8 = (m.astype(np.uint8) > 0).astype(np.uint8)
            if u8.max() == 0:
                cleaned.append(u8.astype(bool)); continue
            n, labels = cv2.connectedComponents(u8)
            if n <= 2:
                cleaned.append(u8.astype(bool)); continue
            # 最大連結成分（背景=0を除く1..n-1）
            best_idx, best_area = 1, (labels == 1).sum()
            for k in range(2, n):
                a = (labels == k).sum()
                if a > best_area:
                    best_idx, best_area = k, a
            cleaned.append((labels == best_idx))
        return cleaned

# 同軸統合・長さzscore・スムージング・描画・左右端
from modules.mask_merge import merge_coaxial_rect_masks
from modules.mask_length_zscore_filter import prune_by_major_axis_zscore
from modules.mask_smoothing import smooth_masks_with_dp
from modules.ids_overlay import draw_mask_ids_on_overlay  # 新・描画位置ロジックに対応

# 重複削り取り（縦長優先で overlap 部だけ横長側をカット）
try:
    from modules.overlap_filter import cut_overlap_by_vertical
except ImportError:
    # フォールバック実装（簡易版）
    def _aspect_ratio(m: np.ndarray) -> float:
        info = get_axis_info(m) if 'get_axis_info' in globals() else None
        if info is None or info.W <= 0:
            return 1.0
        return float(info.L / max(info.W, 1e-6))
    def cut_overlap_by_vertical(masks: List[np.ndarray]) -> List[np.ndarray]:
        ms = [m.astype(bool) for m in masks]
        n = len(ms)
        for i in range(n):
            for j in range(i+1, n):
                inter = ms[i] & ms[j]
                if inter.any():
                    ai, aj = _aspect_ratio(ms[i]), _aspect_ratio(ms[j])
                    if ai >= aj:
                        ms[j] = ms[j] & (~inter)   # jから重なり分を削る
                    else:
                        ms[i] = ms[i] & (~inter)   # iから重なり分を削る
        return ms

# 左右端抽出ユーティリティ
from modules.pixel_coordinates import mask_lr_on_row

# ===== 設定 =====
DEFAULT_OUT_DIR = Path("./captures")
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

# ===== 低レベルヘルパ =====
def build_providers(device: str) -> List[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "gpu":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]

def to_prob_0_1(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    xmin, xmax = float(x.min()), float(x.max())
    if xmin >= -1e-6 and xmax <= 1.0 + 1e-6:
        return np.clip(x, 0.0, 1.0)
    return 0.5 * (1.0 + np.tanh(0.5 * x))

# 置換：postprocess_like_sam（logits対応）
def postprocess_like_sam(lowres_256: np.ndarray, orig_hw, new_hw,
                         target_len=1024, is_logits: bool = True) -> np.ndarray:
    oh, ow = orig_hw
    nh, nw = new_hw
    up_1024 = cv2.resize(lowres_256, (target_len, target_len), interpolation=cv2.INTER_LINEAR)
    cropped  = up_1024[:nh, :nw]
    up_full  = cv2.resize(cropped, (ow, oh), interpolation=cv2.INTER_LINEAR)
    thr = 0.0 if is_logits else 0.5
    m_bin = (up_full > thr)
    return (m_bin.astype(np.uint8) * 255)

# --- 新規: デコーダ直後の候補辞書を作るユーティリティ ---
def _bbox_from_mask_u8(m_u8: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(m_u8)
    return int(x), int(y), int(w), int(h)

def _stability_score_from_lowres(lowres_256: np.ndarray, *, delta: float = 0.05, is_logits: bool = True) -> float:
    """SAM風の簡易安定度: しきい値を±deltaずらしたときのマスク差で評価（0..1）。"""
    arr = lowres_256.astype(np.float32)
    if is_logits:
        A = arr > 0.0
        B = arr > (delta)
        C = arr > (-delta)
    else:
        A = arr > 0.5
        B = arr > (0.5 + delta)
        C = arr > (0.5 - delta)
    # C と B の共通領域が多いほど安定
    inter = np.logical_and(B, C).sum()
    u = np.logical_or(B, C).sum()
    return 0.0 if u == 0 else float(inter) / float(u)

def _candidate_from_lowres(lowres_256: np.ndarray, *, score: float, orig_hw: Tuple[int,int], new_hw: Tuple[int,int], is_logits: bool) -> Dict[str, Any]:
    m_u8 = postprocess_like_sam(lowres_256, orig_hw, new_hw, is_logits=is_logits)
    mask_bin = (m_u8 > 0)
    area = int(mask_bin.sum())
    bbox = _bbox_from_mask_u8(m_u8)
    return {
        "mask": mask_bin,
        "bbox": bbox,        # (x,y,w,h)
        "area": area,
        "score": float(score),
        # 参照用
        "_m_u8": m_u8,
    }

def _mask_iou_downsample(a: np.ndarray, b: np.ndarray, *, side: int = 48) -> float:
    au8 = (a.astype(np.uint8) * 255)
    bu8 = (b.astype(np.uint8) * 255)
    if au8.shape != (side, side):
        au8 = cv2.resize(au8, (side, side), interpolation=cv2.INTER_NEAREST)
    if bu8.shape != (side, side):
        bu8 = cv2.resize(bu8, (side, side), interpolation=cv2.INTER_NEAREST)
    A = au8 > 0
    B = bu8 > 0
    inter = np.logical_and(A, B).sum()
    u = A.sum() + B.sum() - inter
    return 0.0 if u == 0 else float(inter) / float(u)

def _bbox_iou_xywh(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    x1, y1, w1, h1 = a; x2, y2, w2, h2 = b
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter <= 0:
        return 0.0
    u = w1*h1 + w2*h2 - inter
    return 0.0 if u == 0 else inter / u

# Helpers のあたりに追加（NMS後の穴埋め）
def gap_fill_by_xbins(selected: List[Dict[str,Any]],
                      candidates: List[Dict[str,Any]],
                      width: int, *,
                      bins: int = 24,
                      iou_thr: float = 0.98,
                      downsample_side: int = 48) -> List[Dict[str,Any]]:
    import numpy as np
    def xc(bb): x, y, w, h = bb; return (x + x + w) // 2
    taken = np.zeros(bins, dtype=bool)
    binw = max(1, width // bins)

    for s in selected:
        bb = s.get("bbox")
        if not bb: continue
        b = int(min(bins-1, max(0, xc(bb)//binw)))
        taken[b] = True

    pool = sorted(candidates, key=lambda d: d.get("score", 0.0), reverse=True)
    for c in pool:
        bb, m = c.get("bbox"), c.get("mask")
        if not bb or m is None: continue
        b = int(min(bins-1, max(0, xc(bb)//binw)))
        if taken[b]:
            continue
        dup = False
        for s in selected:
            sbb, sm = s.get("bbox"), s.get("mask")
            if not sbb or sm is None:
                continue
            if _bbox_iou_xywh(bb, sbb) < 0.5:   # bbox がそこそこ離れていればマスク比較しない
                continue
            if _mask_iou_downsample(m, sm, side=downsample_side) > iou_thr:
                dup = True
                break
        if not dup:
            selected.append(c)
            taken[b] = True
    return selected

# ===== 表示/保存ユーティリティ =====

def _save_points_and_overlay(
    base_img: Image.Image,
    masks: List[np.ndarray],
    out_dir: Path,
    prefix: str,
    draw_ids: bool = False,
) -> None:
    """点群のみ NPY 保存 + overlay JPG 保存（30色）。
    after_nms のときだけ 180° 回転して保存。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # points.npy（※座標は回転させず保存）
    all_points = []
    for m in masks:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = np.concatenate(cnts, axis=0) if cnts else np.zeros((0, 1, 2), np.int32)
        all_points.append(pts.squeeze(1))
    np.save(out_dir / f"{prefix}_points.npy", np.array(all_points, dtype=object))

    # overlay.jpg
    overlay = base_img.convert("RGBA")
    for i, m in enumerate(masks):
        alpha = Image.fromarray((m.astype(np.uint8) * 140))
        tint  = Image.new("RGBA", base_img.size, (*PALETTE_30[i % len(PALETTE_30)], 0))
        tint.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, tint)
    overlay = overlay.convert("RGB")

    if draw_ids:
        # 新しい ids_overlay のデフォルト（mode="top_mid"）が効く
        overlay = draw_mask_ids_on_overlay(overlay, masks, start_index=1)

    if prefix.startswith("after_nms"):
        overlay = overlay.rotate(180, expand=False)
    overlay = overlay.transpose(Image.ROTATE_180)
    overlay.save(out_dir / f"{prefix}_overlay.jpg", quality=92)
    


def _render_overlay_bgr(
    base_img: Image.Image,
    masks: List[np.ndarray],
    draw_ids: bool = False,
) -> np.ndarray:
    """オーバーレイを作って BGR(np.ndarray) で返す（imshow 用）。"""
    overlay = base_img.convert("RGBA")
    for i, m in enumerate(masks):
        alpha = Image.fromarray((m.astype(np.uint8) * 140))
        tint  = Image.new("RGBA", base_img.size, (*PALETTE_30[i % len(PALETTE_30)], 0))
        tint.putalpha(alpha)
        overlay = Image.alpha_composite(overlay, tint)
    overlay = overlay.convert("RGB")
    if draw_ids:
        overlay = draw_mask_ids_on_overlay(overlay, masks, start_index=1)
    rgb = np.array(overlay)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# ===== 推論器 =====
@dataclass
class SamConfig:
    encoder_path: str = "models/sam_vit_h_4b8939.encoder.onnx"
    decoder_path: str = "models/sam_vit_h_4b8939.decoder.onnx"
    device: str = "gpu"
    target_len: int = 1024
    pts_side: int = 24
    min_area: int = 500
    iou_thr: float = 0.8

@dataclass
class StageSaveCfg:
    out_dir: Path = DEFAULT_OUT_DIR
    save_after_nms: bool = True
    save_before_smooth: bool = True
    save_after_smooth: bool = True
    save_selected: bool = True

class SamBatchInfer:
    def __init__(self, cfg: SamConfig):
        self.cfg = cfg
        providers = build_providers(cfg.device)
        self.encoder_sess = ort.InferenceSession(cfg.encoder_path, providers=providers)
        self.decoder_sess = ort.InferenceSession(cfg.decoder_path, providers=providers)
        self.enc_input = self.encoder_sess.get_inputs()[0]

    # 低解像度で候補の“違い”を判定（同一点で上位K）
    def _lowres_iou_binary(self, a: np.ndarray, b: np.ndarray, is_logits: bool, side: int = 128) -> float:
        A = cv2.resize(a.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
        B = cv2.resize(b.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
        mA = A > (0.0 if is_logits else 0.5)
        mB = B > (0.0 if is_logits else 0.5)
        inter = np.logical_and(mA, mB).sum()
        union = np.logical_or(mA, mB).sum()
        return 0.0 if union == 0 else inter / union

    def _run_point_multi(self, pack, image_embeddings, x, y, oh, ow, nh, nw,
                         k_keep: int = 3, lowres_dist_thr: float = 0.98):
        pt_1024 = pack.coords_to_1024(np.array([[x, y]], np.float32))[None, ...]
        lbl     = np.array([[1]], np.float32)
        feeds = {
            "image_embeddings": image_embeddings,
            "point_coords":     pt_1024,
            "point_labels":     lbl,
            "mask_input":       np.zeros((1,1,256,256), np.float32),
            "has_mask_input":   np.array([0], np.float32),
            "orig_im_size":     np.array([oh, ow], np.float32),
        }
        outs_names = [o.name for o in self.decoder_sess.get_outputs()]
        outs_vals  = self.decoder_sess.run(outs_names, feeds)
        outs       = {k: v for k, v in zip(outs_names, outs_vals)}

        use_key   = "low_res_masks" if "low_res_masks" in outs else "masks"
        is_logits = (use_key == "low_res_masks")
        if "iou_predictions" in outs:
            order = list(np.argsort(outs["iou_predictions"][0])[::-1])
            iou_pred = outs["iou_predictions"][0]
        else:
            order = [0]; iou_pred = None

        picked = []
        picked_lowres = []
        for idx in order:
            lowres = outs[use_key][0, idx]  # (256,256)
            stab = _stability_score_from_lowres(lowres, delta=0.05, is_logits=is_logits)
            iou_p = float(iou_pred[idx]) if iou_pred is not None else 1.0
            score = iou_p * (0.2 + 0.8 * stab)  # 安定性の影響を少し弱める

            # 形が既存と十分違うか？
            if all(self._lowres_iou_binary(lowres, prev, is_logits, side=128) < lowres_dist_thr
                   for prev in picked_lowres):
                cand = _candidate_from_lowres(
                    lowres_256=lowres, score=score,
                    orig_hw=(oh, ow), new_hw=(nh, nw), is_logits=is_logits
                )
                picked.append(cand); picked_lowres.append(lowres)
                if len(picked) >= k_keep:
                    break
        return picked

    def _run_point(self, pack, image_embeddings, x, y, oh, ow, nh, nw):
        pt_1024 = pack.coords_to_1024(np.array([[x, y]], np.float32))[None, ...]
        lbl     = np.array([[1]], np.float32)
        feeds = {
            "image_embeddings": image_embeddings,
            "point_coords":     pt_1024,
            "point_labels":     lbl,
            "mask_input":       np.zeros((1,1,256,256), np.float32),
            "has_mask_input":   np.array([0], np.float32),
            "orig_im_size":     np.array([oh, ow], np.float32),
        }
        outs_names = [o.name for o in self.decoder_sess.get_outputs()]
        outs_vals  = self.decoder_sess.run(outs_names, feeds)
        outs       = {k: v for k, v in zip(outs_names, outs_vals)}
        best = int(np.argmax(outs["iou_predictions"][0])) if "iou_predictions" in outs else 0
        arr_256 = outs.get("low_res_masks", outs.get("masks"))[0, best]
        m_u8    = postprocess_like_sam(arr_256, (oh, ow), (nh, nw))
        return (m_u8 > 0)

    def infer_masks(
        self,
        img: Image.Image,
        *,
        # ← fold角ベースの縦フィルタ
        min_fold_deg: float = 70.0,
        min_aspect: float = 1.3,
        # 同軸統合ほか
        axis_angle_tol_deg: float = 10.0,
        offset_width_factor: float = 0.25,
        gap_len_factor: float = 0.50,
        z_lower: Optional[float] = -0.1,
        min_len_px: float = 0.0,
        stage_save: Optional[StageSaveCfg] = None,
        stem_for_save: str = "image",
    ) -> List[np.ndarray]:
        """
        前処理→グリッド→候補生成（各点で上位K/形状重複抑制）→Box-NMS→Mask-IoUデデュープ
        →縦向きフィルタ→飛び地除去→同軸統合→zscore→重複削り取り→スムージング→最終マスク群
        """
        cfg = self.cfg
        pack = preprocess_for_encoder(img, self.enc_input, target_length=cfg.target_len)
        emb = self.encoder_sess.run(None, {pack.enc_name: pack.feed_img})[0]
        if emb.ndim == 3:
            emb = emb[None, ...]
        image_embeddings = emb.astype(np.float32)
        oh, ow = pack.orig_hw
        nh, nw = pack.new_hw

        # グリッド
        xs = np.linspace(5, ow - 5, cfg.pts_side, dtype=np.float32)
        ys = np.linspace(5, oh - 5, cfg.pts_side, dtype=np.float32)

        # --- デコード & 面積フィルタ（同一点で上位Kを拾う） ---
        cands: List[Dict[str,Any]] = []
        MIN_AREA = max(3000, self.cfg.min_area)  # 細い背表紙救済のため少し下げ気味推奨
        for y in ys:
            for x in xs:
                for c in self._run_point_multi(pack, image_embeddings, x, y, oh, ow, nh, nw,
                                               k_keep=3, lowres_dist_thr=0.98):
                    if c["area"] >= MIN_AREA:
                        cands.append(c)

        # （任意）半ピッチずらしでもう1周
        if len(xs) > 1:
            pitch = xs[1] - xs[0]
            xs_shift = np.clip(xs + pitch/2, 5, ow-5)
            for y in ys:
                for x in xs_shift:
                    for c in self._run_point_multi(pack, image_embeddings, x, y, oh, ow, nh, nw,
                                                   k_keep=3, lowres_dist_thr=0.98):
                        if c["area"] >= MIN_AREA:
                            cands.append(c)

        # --- NMS/デデュープ（modules/nms.py を使用） ---
        cands = quality_filter(cands, pred_iou_thr=None, stability_thr=None)
        kept = nms_box_greedy_spatial(cands, iou_thr=0.7, score_key="score")
        selected = dedup_by_mask_iou_spatial(
            kept,
            iou_thr=0.97,
            bbox_gate=0.70,
            downsample_side=48,
            score_key="score",
        )

        # X方向の穴埋め（中央の帯の取りこぼし救済）
        selected = gap_fill_by_xbins(
            selected, cands, width=ow,
            bins=24, iou_thr=0.98, downsample_side=48
        )

        masks = [c["mask"] for c in selected]
        if stage_save and stage_save.save_after_nms:
            _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_after_nms", draw_ids=False)

        # === 縦向きフィルタ（fold角） ===
        masks, _ = filter_keep_vertical_by_fold(
            masks,
            min_fold_deg=min_fold_deg,
            min_aspect=min_aspect,
            min_len_px=min_len_px,
        )

        # === 飛び地除去（最大連結成分のみ残す） ===
        masks = remove_islands_from_masks(masks)

        # === 同軸統合 → zscore ===
        masks = merge_coaxial_rect_masks(
            masks,
            axis_angle_tol_deg=axis_angle_tol_deg,
            offset_width_factor=offset_width_factor,
            gap_len_factor=gap_len_factor,
            rectify=False
        )
        if z_lower is not None:
            masks = prune_by_major_axis_zscore(masks, z_lower=z_lower, min_len_px=min_len_px)

        # === 重複削り取り（縦長優先で overlap 部だけ横長側をカット） ===
        masks = cut_overlap_by_vertical(masks)

        if stage_save and stage_save.save_before_smooth:
            _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_before_smooth", draw_ids=False)

        # スムージング
        masks = smooth_masks_with_dp(masks, epsilon_factor=0.004)
        if stage_save and stage_save.save_after_smooth:
            _save_points_and_overlay(img, masks, stage_save.out_dir, f"{stem_for_save}_after_smooth", draw_ids=True)

        return masks

    @staticmethod
    def uv_from_mask(
        m: np.ndarray,
        *,
        swap_lr_output: bool = False,
        y_from_top_px: int = 230,
        search_radius_px: int = 80,
    ) -> Dict[str, int]:
        """
        指定 y（上から y_from_top_px [px]）の水平ラインで左右端 x を返す。
        戻り値: {"uv_left": int, "uv_right": int}
        """
        H, W = m.shape
        y0 = int(np.clip(y_from_top_px, 0, H - 1))

        # まず指定行
        lr = mask_lr_on_row(m, y0)
        if lr is not None:
            xL, xR = lr
        else:
            # ダメなら ±search_radius で近い行を探索（上→下の順）
            found = False
            for d in range(1, int(search_radius_px) + 1):
                yp = y0 - d
                if yp >= 0:
                    lr = mask_lr_on_row(m, yp)
                    if lr is not None:
                        xL, xR = lr
                        y0 = yp
                        found = True
                        break
                yn = y0 + d
                if yn < H:
                    lr = mask_lr_on_row(m, yn)
                    if lr is not None:
                        xL, xR = lr
                        y0 = yn
                        found = True
                        break
            if not found:
                # それでも見つからなければフォールバック：全体の min/max x
                ys, xs = np.where(m > 0)
                if xs.size == 0:
                    return {"uv_left": None, "uv_right": None}
                xL, xR = int(xs.min()), int(xs.max())

        if swap_lr_output:
            xL, xR = xR, xL
        # mask_lr_on_row が nd-array 返す可能性に対処
        try:
            xL = int(xL[0]) if isinstance(xL, (np.ndarray, list, tuple)) else int(xL)
            xR = int(xR[0]) if isinstance(xR, (np.ndarray, list, tuple)) else int(xR)
        except Exception:
            xL, xR = int(xL), int(xR)
        return {"uv_left": xL, "uv_right": xR}

    # ---------- 画像パス入力 ----------
    def run_on_image_path(
        self,
        img_path: Path | str,
        *,
        stage_save_dir: Optional[Path] = None,   # NoneならDEFAULT_OUT_DIR
        swap_lr_output: bool = False,
        # fold角フィルタ
        min_fold_deg: float = 70.0,
        min_aspect: float = 1.3,
        # 同軸統合ほか
        axis_angle_tol_deg: float = 10.0,
        offset_width_factor: float = 0.25,
        gap_len_factor: float = 0.50,
        z_lower: Optional[float] = -0.1,
        min_len_px: float = 0.0,

        y_from_top_px: int = 230,
        scan_radius_px: int = 80,
    ) -> Dict[str, Any]:
        out_dir = stage_save_dir or DEFAULT_OUT_DIR
        img_path = Path(img_path)
        img = Image.open(img_path).convert("RGB")
        stem = img_path.stem

        # after_nms / before_smooth / after_smooth を保存（after_smooth は番号描画あり）
        stage_cfg = StageSaveCfg(out_dir=out_dir)
        masks = self.infer_masks(
            img,
            min_fold_deg=min_fold_deg,
            min_aspect=min_aspect,
            axis_angle_tol_deg=axis_angle_tol_deg,
            offset_width_factor=offset_width_factor,
            gap_len_factor=gap_len_factor,
            z_lower=z_lower,
            min_len_px=min_len_px,
            stage_save=stage_cfg,
            stem_for_save=stem,
        )

        if not masks:
            print("[SAM] No masks found.")
            return {"image_path": str(img_path), "num_masks": 0, "selected_index_1based": None, "uv_left": None, "uv_right": None}

        # 画面にも番号付きで表示（見れない環境でも保存物は out_dir にあります）
        ov_bgr = _render_overlay_bgr(img, masks, draw_ids=True)
        try:
            cv2.imshow("SAM after_smooth", ov_bgr)
            cv2.waitKey(1)
        except cv2.error:
            pass

        # ターミナルで番号入力（必須 / qで中断）
        while True:
            try:
                sel = input(f"[select mask id 1-{len(masks)} / q=skip] > ").strip()
            except EOFError:
                sel = "q"
            if sel.lower() in ("q", "quit", "exit"):
                try: cv2.destroyWindow("SAM after_smooth")
                except cv2.error: pass
                return {"image_path": str(img_path), "num_masks": len(masks), "selected_index_1based": None, "uv_left": None, "uv_right": None}
            try:
                idx = int(sel)
                if 1 <= idx <= len(masks):
                    break
            except ValueError:
                pass
            print("  !! invalid id. try again.")

        sel_mask = masks[idx - 1]

        # 選択後の保存（任意）
        _save_points_and_overlay(img, [sel_mask], out_dir, f"{stem}_mask{idx}_selected", draw_ids=False)

        uv = self.uv_from_mask(
            sel_mask,
            swap_lr_output=swap_lr_output,
            y_from_top_px=y_from_top_px,
            search_radius_px=scan_radius_px,
        )
        H, W = img.size[1], img.size[0]
        top_x, bottom_x = axis_x_on_centerline(
            sel_mask, y_top=120, y_bottom=400, img_hw=(H, W)
        )

        try: cv2.destroyWindow("SAM after_smooth")
        except cv2.error: pass

        return {
            "image_path": str(img_path),
            "num_masks": len(masks),
            "selected_index_1based": idx,
            **uv,   # {"uv_left": int, "uv_right": int}
            "bottom": top_x,      # y=120 上の中心軸交点 x
            "top": bottom_x # y=400 上の中心軸交点 x
        }

    # ---------- BGRフレーム（対話選択） ----------
    def run_on_rgb_frame(
        self,
        rgb_frame,                             # np.ndarray(H,W,3) BGR
        *,
        stage_save_dir: Optional[Path] = None, # NoneならDEFAULT_OUT_DIR
        swap_lr_output: bool = False,
        stem_for_save: str = "frame",
        # fold角フィルタ
        min_fold_deg: float = 70.0,
        min_aspect: float = 1.3,
        # 同軸統合ほか
        axis_angle_tol_deg: float = 10.0,
        offset_width_factor: float = 0.25,
        gap_len_factor: float = 0.50,
        z_lower: Optional[float] = -2.0,
        min_len_px: float = 0.0,
        y_from_top_px: int = 230,
        search_radius_px: int = 80,
    ) -> Dict[str, Any]:
        if rgb_frame is None or rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError("rgb_frame must be HxWx3 BGR np.ndarray")
        out_dir = stage_save_dir or DEFAULT_OUT_DIR

        img_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        base_img = Image.fromarray(img_rgb)

        stage_cfg = StageSaveCfg(out_dir=out_dir)
        masks = self.infer_masks(
            base_img,
            min_fold_deg=min_fold_deg,
            min_aspect=min_aspect,
            axis_angle_tol_deg=axis_angle_tol_deg,
            offset_width_factor=offset_width_factor,
            gap_len_factor=gap_len_factor,
            z_lower=z_lower,
            min_len_px=min_len_px,
            stage_save=stage_cfg,
            stem_for_save=stem_for_save,
        )

        if not masks:
            print("[SAM] No masks found.")
            return {"uv_left": None, "uv_right": None, "selected_index_1based": None, "num_masks": 0}

        ov_bgr = _render_overlay_bgr(base_img, masks, draw_ids=True)
        # try:
        #     cv2.imshow("SAM after_smooth", ov_bgr)
        #     cv2.waitKey(1)
        # except cv2.error:
        #     pass

        while True:
            try:
                sel = input(f"[select mask id 1-{len(masks)} / q=skip] > ").strip()
            except EOFError:
                sel = "q"
            if sel.lower() in ("q", "quit", "exit"):
                try: cv2.destroyWindow("SAM after_smooth")
                except cv2.error: pass
                return {"uv_left": None, "uv_right": None, "selected_index_1based": None, "num_masks": len(masks)}#ホントは、型揃えないとね
            try:
                idx = int(sel)
                if 1 <= idx <= len(masks):
                    break
            except ValueError:
                pass
            print("  !! invalid id. try again.")

        sel_mask = masks[idx - 1]
        _save_points_and_overlay(base_img, [sel_mask], out_dir, f"{stem_for_save}_mask{idx}_selected", draw_ids=False)
        uv = self.uv_from_mask(sel_mask, swap_lr_output=swap_lr_output, y_from_top_px=y_from_top_px, search_radius_px=search_radius_px)
        
        H, W = base_img.size[1], base_img.size[0]
        top_x, bottom_x = axis_x_on_centerline(
        sel_mask, y_top=120, y_bottom=400, img_hw=(H, W)
        )
        try: cv2.destroyWindow("SAM after_smooth")
        except cv2.error: pass

        return {"uv_left": uv["uv_left"], "uv_right": uv["uv_right"],
                "top": top_x, "bottom": bottom_x,
                "selected_index_1based": idx, "num_masks": len(masks)}

# ===== CLI =====
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True, help="入力画像パス（1枚）")
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR), help="出力フォルダ")
    ap.add_argument("--encoder", default="models/sam_vit_h_4b8939.encoder.onnx")
    ap.add_argument("--decoder", default="models/sam_vit_h_4b8939.decoder.onnx")
    ap.add_argument("--device", choices=["gpu", "cpu", "auto"], default="gpu")

    ap.add_argument("--pts_side", type=int, default=30)
    ap.add_argument("--min_area", type=int, default=500)
    ap.add_argument("--iou_thr", type=float, default=0.8)

    ap.add_argument("--z_lower", type=float, default=-0.1)
    ap.add_argument("--min_len_px", type=float, default=0.0)

    # ← 新しい閾値（fold角ベース）
    ap.add_argument("--min_fold_deg", type=float, default=70.0, help="fold角の下限（0=水平, 90=垂直）")
    ap.add_argument("--min_aspect", type=float, default=1.3, help="縦長判定の L/W 下限")

    ap.add_argument("--axis_angle_tol", type=float, default=10.0)
    ap.add_argument("--offset_width_factor", type=float, default=0.25)
    ap.add_argument("--gap_len_factor", type=float, default=0.50)

    ap.add_argument("--swap_lr_output", action="store_true", help="左右の出力を入れ替える")
    ap.add_argument("--y_from_top_px", type=int, default=230,
                    help="左右端抽出に用いる水平ライン（画像上端からのpx）")
    ap.add_argument("--scan_radius_px", type=int, default=80,
                    help="指定ラインで交差しない場合に±探索する半径[px]")
    return ap.parse_args()

def _main_cli():
    args = _parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SamConfig(
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        device=args.device,
        pts_side=args.pts_side,
        min_area=args.min_area,
        iou_thr=args.iou_thr,
    )
    runner = SamBatchInfer(cfg)

    res = runner.run_on_image_path(
        args.image,
        stage_save_dir=out_dir,
        swap_lr_output=args.swap_lr_output,
        # fold角フィルタ
        min_fold_deg=args.min_fold_deg,
        min_aspect=args.min_aspect,
        # 同軸統合ほか
        axis_angle_tol_deg=args.axis_angle_tol,
        offset_width_factor=args.offset_width_factor,
        gap_len_factor=args.gap_len_factor,
        z_lower=args.z_lower,
        min_len_px=args.min_len_px,
        y_from_top_px=args.y_from_top_px,
        scan_radius_px=args.scan_radius_px,
    )
    if res.get("selected_index_1based") is None:
        print("NO_MASK_OR_SKIPPED")
    else:
        print(f"UV_LEFT_X={res['uv_left']} UV_RIGHT_X={res['uv_right']} SEL={res['selected_index_1based']}")

if __name__ == "__main__":
    _main_cli()
