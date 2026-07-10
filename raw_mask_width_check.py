#!/usr/bin/env python3
"""
「深度フィルタより前」、SAM2+OCRで選ばれた直後の生マスクの幅を実測し、
GT・マスク段階(stage1)・最終段階(stage2)と比較するスクリプト。

これまでの stage_width_trace.py は
  Stage1 = 深度フィルタ＋背表紙帯補完後マスクの width_median_px
  Stage2 = 最終手法(M1/M2)の出力
の2点しか比較できず、「深度フィルタより前のSAM/OCR選択そのもの」は
area/lengthの粗い近似でしか見られず信頼できなかった（過去の調査で破棄）。

このスクリプトでは、パイプラインが原因解析用に保存している
  mask{N}_offline_selected_before_depth_prefilter_depth_points.png
（深度フィルタ適用前の選択マスクの有効Depth画素を黒背景に着色したデバッグ画像。
 背景は(0,0,0)、マスク内の有効画素だけ色が付く）
を読み込み、非ゼロ画素から生マスクの2値画像を復元する。
その上で get_book_points.py の analyze_mask_rectangularity と同一のアルゴリズム
（回転長方形フィット→長辺方向スライスごとの幅5-95%パーセンタイル→中央値）
をこのスクリプトに移植して width_median_px を算出し、mm換算する。

get_book_points.py 自体は pyrealsense2/SAM2 等の重い依存とパッケージ相対import を
含み単体importが困難なため、analyze_mask_rectangularity のロジックのみを
そのまま移植している（アルゴリズムはget_book_points.py:3067-3231と同一）。

使い方:
  python3 raw_mask_width_check.py [run_dir]
出力:
  <run_dir>/raw_mask_width_check.csv
"""

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_BASE = BASE_DIR / "captures" / "100test_offline"
MASTER_JSON = BASE_DIR / "master_20260216.json"


def latest_run_dir(offline_base: Path) -> Path:
    dirs = sorted(
        [d for d in offline_base.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )
    if not dirs:
        raise FileNotFoundError(f"実行結果が見つかりません: {offline_base}")
    return dirs[-1]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_glob_first(directory: Path, pattern: str):
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


# ---- get_book_points.py:3067-3231 の analyze_mask_rectangularity を移植 ----
def analyze_mask_rectangularity_width(mask01: np.ndarray):
    mask = (np.asarray(mask01) > 0).astype(np.uint8)
    area = int(mask.sum())
    if area < 200:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) <= 1.0:
        return None

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect
    rw, rh = float(rw), float(rh)

    if rw >= rh:
        a = np.deg2rad(float(angle))
    else:
        a = np.deg2rad(float(angle) + 90.0)
    long_axis = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
    long_axis = long_axis / max(float(np.linalg.norm(long_axis)), 1e-9)
    normal = np.array([-long_axis[1], long_axis[0]], dtype=np.float64)
    center = np.array([float(cx), float(cy)], dtype=np.float64)

    ys, xs = np.where(mask > 0)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    rel = pts - center
    s = rel @ long_axis
    t = rel @ normal

    width_median = None
    try:
        s_low, s_high = np.percentile(s, [10.0, 90.0])
        use = (s >= s_low) & (s <= s_high)
        s_use, t_use = s[use], t[use]
        if s_use.size >= 50:
            n_bins = max(8, min(80, int(round((s_high - s_low) / 10.0))))
            edges = np.linspace(float(s_low), float(s_high), n_bins + 1)
            widths = []
            for i in range(n_bins):
                m = (s_use >= edges[i]) & (s_use < edges[i + 1])
                if np.count_nonzero(m) < 8:
                    continue
                q0, q1 = np.percentile(t_use[m], [5.0, 95.0])
                widths.append(float(q1 - q0))
            if len(widths) >= 4:
                width_median = float(np.median(np.asarray(widths)))
    except Exception:
        pass

    return {
        "width_median_px": width_median,
        "rect_size": [rw, rh],
        "rect_angle_deg": float(angle),
        "normal": [float(normal[0]), float(normal[1])],
        "area_px": area,
    }


def scale_m_per_px(z_median_raw, depth_scale, normal, fx, fy):
    z_m = z_median_raw * depth_scale
    nx, ny = normal
    return z_m * math.sqrt((nx / fx) ** 2 + (ny / fy) ** 2)


def collect_case(case_dir: Path, gt_mm: float, cam: dict):
    png_path = find_glob_first(case_dir, "mask*_offline_selected_before_depth_prefilter_depth_points.png") \
        or find_glob_first(case_dir, "mask*_selected_before_depth_prefilter_depth_points.png")
    if png_path is None:
        return None

    log_path = Path(str(png_path).replace("_depth_points.png", "_depth_points_log.json"))
    log = load_json(log_path)
    if not log or not log.get("used", False):
        return None
    z_median_raw = log.get("depth_raw_median")
    if z_median_raw is None or cam is None:
        return None

    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    mask01 = (img.sum(axis=2) > 0).astype(np.uint8)

    # デバッグ画像には右上にカラーバー(x>=1160)とタイトル文字(y<60)が
    # 白/JET色で描き込まれており(_save_mask_depth_points_debug_image参照)、
    # これをマスクとして誤検出しないよう除外する。
    h, w = mask01.shape[:2]
    mask01[:60, :] = 0
    mask01[:, min(1160, w):] = 0

    # 深度ドロップアウトでマスクが分断されている場合、最大の連結成分だけを使う。
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if n_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask01 = (labels == largest_label).astype(np.uint8)

    shape = analyze_mask_rectangularity_width(mask01)
    if shape is None or shape["width_median_px"] is None:
        return None

    scale = scale_m_per_px(z_median_raw, cam["depth_scale"], shape["normal"], cam["fx"], cam["fy"])
    width_mm = shape["width_median_px"] * scale * 1000
    return {
        "stage0_raw_mask_mm": width_mm,
        "err0_vs_gt": width_mm - gt_mm,
        "raw_area_px": shape["area_px"],
        "raw_width_median_px": shape["width_median_px"],
    }


def _fmt(v, fmt=".1f"):
    return f"{v:{fmt}}" if v is not None else "  N/A"


def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) >= 2 else latest_run_dir(OFFLINE_BASE)
    print(f"対象ディレクトリ: {run_dir}")

    master_data = load_json(MASTER_JSON)
    books = [{"book_name": b["book_name"], "book_width_mm": float(b["book_width"])} for b in master_data]
    n_books = len(books)

    # stage_width_trace.csv があれば誤差1/誤差2と結合する
    trace_csv = run_dir / "stage_width_trace.csv"
    trace_by_idx = {}
    if trace_csv.exists():
        with open(trace_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                trace_by_idx[int(row["test_index"])] = row

    case_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    rows = []
    for case_dir in case_dirs:
        idx = int(case_dir.name)
        book = books[(idx - 1) % n_books]
        cam = load_json(case_dir / "camera_params.json")
        result = collect_case(case_dir, gt_mm=book["book_width_mm"], cam=cam)
        if result is None:
            continue
        t = trace_by_idx.get(idx, {})
        row = {
            "test_index": idx,
            "book_name": book["book_name"],
            "gt_mm": book["book_width_mm"],
            **result,
            "err1_vs_gt": float(t["err1_vs_gt"]) if t.get("err1_vs_gt") else None,
            "err2_vs_gt": float(t["err2_vs_gt"]) if t.get("err2_vs_gt") else None,
            "method": t.get("method"),
        }
        rows.append(row)

    out_csv = run_dir / "raw_mask_width_check.csv"
    fieldnames = ["test_index", "book_name", "gt_mm", "raw_width_median_px", "raw_area_px",
                  "stage0_raw_mask_mm", "err0_vs_gt", "err1_vs_gt", "err2_vs_gt", "method"]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV保存: {out_csv} ({len(rows)}件)")

    print(f"\n{'idx':>4} {'book_name':<28} {'GT':>6} {'誤差0(生mask)':>13} {'誤差1(filt)':>11} {'誤差2(最終)':>11}")
    print("-" * 80)
    for r in sorted(rows, key=lambda r: r["book_name"]):
        print(
            f"{r['test_index']:>4} {r['book_name'][:28]:<28} {r['gt_mm']:>6.1f} "
            f"{_fmt(r['err0_vs_gt']):>13} {_fmt(r['err1_vs_gt']):>11} {_fmt(r['err2_vs_gt']):>11}"
        )

    print("\n" + "=" * 80)
    print("  書籍別 誤差0(生マスク, 深度フィルタ前)の平均")
    print("=" * 80)
    from collections import defaultdict
    by_book = defaultdict(list)
    for r in rows:
        by_book[r["book_name"]].append(r["err0_vs_gt"])
    for name, errs in by_book.items():
        print(f"{name[:40]:<40} n={len(errs):>3}  平均誤差0={sum(errs)/len(errs):+.1f}mm")


if __name__ == "__main__":
    main()
