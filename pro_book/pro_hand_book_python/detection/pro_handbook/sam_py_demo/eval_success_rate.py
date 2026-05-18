#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2


# ============================================================
#  I/O
# ============================================================

def load_rgba_or_bgr(path: Path) -> np.ndarray:
    """PNGのalphaを取りたいのでIMREAD_UNCHANGEDで読む。"""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read: {path}")
    return img


def gt_mask_from_alpha(img: np.ndarray, alpha_thr: int = 1) -> np.ndarray:
    """
    透明背景のPNGを想定：alpha > alpha_thr を GT=1 とする。
    img: (H,W,4) BGRA 期待
    """
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        return (alpha >= alpha_thr).astype(np.uint8)
    raise ValueError("GT image has no alpha channel (not BGRA).")


def gt_mask_fallback_red(img_bgr: np.ndarray) -> np.ndarray:
    """
    alphaが無い場合の保険：赤っぽい部分をGTとして抽出
    """
    if img_bgr.ndim == 3 and img_bgr.shape[2] >= 3:
        b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
        m = (r > 80) & (r.astype(np.int16) - g.astype(np.int16) > 40) & (r.astype(np.int16) - b.astype(np.int16) > 40)
        return m.astype(np.uint8)
    raise ValueError("unexpected GT image shape")


import numpy as np
import cv2

def _unwrap_object_array(x):
    """np.load(allow_pickle=True) で出る object 配列を中身に剥がす"""
    if isinstance(x, np.ndarray) and x.dtype == object:
        # よくある: array([something], dtype=object)
        if x.size == 1:
            return x.item()
    return x

def load_npy_mask(path, H: int, W: int) -> np.ndarray:
    """
    .npy から「(H,W) の0/1マスク」を作って返す。
    対応:
      - (H,W) 2値/確率マップ
      - (N,2) or (1,N,2) などのポリゴン座標列
      - object配列で包まれているケース
    """
    arr = np.load(str(path), allow_pickle=True)
    arr = _unwrap_object_array(arr)
    arr = np.asarray(arr)

    # 1) すでに画像マスクっぽい
    if arr.ndim == 2:
        m = arr
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

        # 値域で閾値を自動選択（0..1 or 0..255想定）
        m_min, m_max = float(np.min(m)), float(np.max(m))
        thr = 0.5 if m_max <= 2.0 else 127.0
        return (m > thr).astype(np.uint8)

    # 2) ポリゴン座標列っぽい（例: (71,2) / (1,71,2)）
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[-1] == 2:
        pts = arr[0]
    elif arr.ndim == 2 and arr.shape[-1] == 2:
        pts = arr
    else:
        raise ValueError(f"Unsupported npy shape: {arr.shape}, dtype={arr.dtype}")

    # pts: (N,2) を int32 にして塗りつぶし
    pts = np.asarray(pts, dtype=np.float32)
    pts = np.round(pts).astype(np.int32)

    # 念のため画像範囲にクリップ
    pts[:, 0] = np.clip(pts[:, 0], 0, W-1)  # x
    pts[:, 1] = np.clip(pts[:, 1], 0, H-1)  # y

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask



# ============================================================
#  Boundary & Hausdorff (あなたの元アルゴリズム)
# ============================================================

def mask_to_boundary(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    kernel = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
    grad[grad != 0] = 1
    return grad


def hausdorff_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    ba = mask_to_boundary(mask_a)
    bb = mask_to_boundary(mask_b)

    if ba.sum() == 0 or bb.sum() == 0:
        return float("inf")

    da = cv2.distanceTransform(1 - ba, cv2.DIST_L2, 3)
    db = cv2.distanceTransform(1 - bb, cv2.DIST_L2, 3)

    ya, xa = np.where(ba > 0)
    yb, xb = np.where(bb > 0)

    d_ab = float(db[ya, xa].max())
    d_ba = float(da[yb, xb].max())

    return max(d_ab, d_ba)


# ============================================================
#  Pairing
# ============================================================

import re
from pathlib import Path
from typing import List, Tuple

def _extract_img_index(p: Path) -> int:
    """
    after_init_rgb.png      -> 0
    after_init_rgb (12).png -> 12
    """
    s = p.stem
    m = re.search(r'\((\d+)\)\s*$', s)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d+)\s*$', s)
    if m:
        return int(m.group(1))
    return 0  # 数字なしは0扱い（after_init_rgb.png）

def _extract_mask_index(p: Path) -> int:
    """
    mask13.npy -> 12 にしたい（= mask番号 - 1）
    """
    s = p.stem
    m = re.search(r'(\d+)\s*$', s)
    if not m:
        raise ValueError(f"Cannot parse index from mask name: {p.name}")
    return int(m.group(1)) - 1


def collect_pairs_by_index(
    directory: Path,
    img_glob: str,
    mask_glob: str
) -> List[Tuple[Path, Path]]:
    imgs = list(directory.glob(img_glob))
    masks = list(directory.glob(mask_glob))

    if not imgs:
        raise RuntimeError(f"No images matched: {img_glob} in {directory}")
    if not masks:
        raise RuntimeError(f"No masks matched: {mask_glob} in {directory}")

   
    img_map = {}
    for p in imgs:
        idx = _extract_img_index(p)
        img_map[idx] = p

    mask_map = {}
    for p in masks:
        idx = _extract_mask_index(p)
        mask_map[idx] = p

    common = sorted(set(img_map.keys()) & set(mask_map.keys()))
    if not common:
        raise RuntimeError("No common indices between GT images and masks. Check filename patterns.")

    # ずれてる原因調査用ログ（必要なら残してOK）
    missing_img = sorted(set(mask_map.keys()) - set(img_map.keys()))
    missing_mask = sorted(set(img_map.keys()) - set(mask_map.keys()))
    if missing_img:
        print(f"[WARN] masks exist but GT missing for indices: {missing_img[:20]}{'...' if len(missing_img)>20 else ''}")
    if missing_mask:
        print(f"[WARN] GT exist but mask missing for indices: {missing_mask[:20]}{'...' if len(missing_mask)>20 else ''}")

    pairs = [(img_map[i], mask_map[i]) for i in common]
    return pairs


# ============================================================
#  Evaluate
# ============================================================

def eval_one_pair(gt_img_path: Path, pred_npy_path: Path, tol_px: float, alpha_thr: int) -> Tuple[float, bool, int, int]:
    gt_img = load_rgba_or_bgr(gt_img_path)

    # GT: alpha優先。alphaが無いなら赤抽出にフォールバック
    try:
        gt = gt_mask_from_alpha(gt_img, alpha_thr=alpha_thr)
    except Exception:
        # UNCHANGEDで読んだ結果がBGRだった場合など
        if gt_img.ndim == 3 and gt_img.shape[2] >= 3:
            # BGRAでなければBGRとして扱う
            gt = gt_mask_fallback_red(gt_img[:, :, :3])
        else:
            raise

    H, W = gt.shape
    pred = load_npy_mask(pred_npy_path, H, W)


    # サイズ合わせ（GT基準・最近傍）
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    d = hausdorff_distance(gt, pred)
    ok = d <= tol_px
    return d, ok, int(gt.sum()), int(pred.sum())


def main():
    ap = argparse.ArgumentParser(description="透明背景GT(alpha) vs npy推論マスクをHausdorff距離で評価")
    ap.add_argument("--dir", default="/home/tai/Downloads/masks", help="ペアが入ったディレクトリ")
    ap.add_argument("--img_glob", default="after_init_rgb*.png", help="GT画像のglob (例: after_init_rgb*.png)")
    ap.add_argument("--mask_glob", default="mask*.npy", help="推論マスクnpyのglob (例: mask*.npy)")
    ap.add_argument("--tol", type=float, default=10.0, help="Hausdorff許容px")
    ap.add_argument("--alpha_thr", type=int, default=1, help="GTのalpha閾値 (>= alpha_thr を前景)")
    ap.add_argument("--csv", default=None, help="結果CSV保存先(任意)")
    args = ap.parse_args()

    directory = Path(args.dir)
    pairs = collect_pairs_by_index(directory, args.img_glob, args.mask_glob)


    rows = []
    ok_count = 0

    for i, (imgp, maskp) in enumerate(pairs):
        d, ok, gt_area, pred_area = eval_one_pair(imgp, maskp, args.tol, args.alpha_thr)
        ok_count += int(ok)

        print("======================================")
        print(f"[{i:04d}] GT  : {imgp.name}")
        print(f"       PRED: {maskp.name}")
        print(f"       Hausdorff = {d:.3f} px (tol={args.tol})  -> {'SUCCESS' if ok else 'FAIL'}")
        print(f"       area: GT={gt_area}  PRED={pred_area}")

        rows.append({
            "index": i,
            "gt_image": imgp.name,
            "pred_mask": maskp.name,
            "hausdorff_px": d,
            "tol_px": args.tol,
            "success": int(ok),
            "gt_area": gt_area,
            "pred_area": pred_area,
        })

    total = len(pairs)
    rate = ok_count / total if total else 0.0
    print("\n==============================")
    print(f"OVERALL: success={ok_count}/{total}  rate={rate:.6f}")
    print("==============================")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
