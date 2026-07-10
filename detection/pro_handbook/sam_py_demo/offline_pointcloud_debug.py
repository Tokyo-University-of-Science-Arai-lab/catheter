import os
import sys

# onnxruntime-gpu が cuDNN を見つけられるよう LD_LIBRARY_PATH を設定して再起動
# （recognition_accuracy_test.py と同じ対処）
_cudnn_lib = (
    "/home/book/pro_book/pro_hand_book_python"
    "/.pro_hand_book_fixed/lib/python3.10/site-packages/nvidia/cudnn/lib"
)
if _cudnn_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _cudnn_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(
        sys.executable,
        [sys.executable, "-m", "detection.pro_handbook.sam_py_demo.offline_pointcloud_debug"]
        + sys.argv[1:],
    )

from .get_book_points import run_capture_and_pca_offline
from pathlib import Path

# Project root: .../pro_hand_book_python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
import json
import csv
import os
import time
import traceback
from datetime import datetime
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr

import cv2
import numpy as np


# =========================
# 設定
# =========================
BASE_DIR = Path(str(PROJECT_ROOT)).resolve()

# 元の入力データ
TEST_BASE_DIR = BASE_DIR / "captures" / "100test"

# offline実行結果の保存先
OFFLINE_BASE_DIR = BASE_DIR / "captures" / "100test_offline"

MASTER_JSON = BASE_DIR / "master_20260216.json"

SAM_DEVICE = "gpu"

START_INDEX = 1
END_INDEX = 100
# 1ラウンドで全書籍を1枚ずつ撮影し、それを N_ROUNDS 回繰り返す構成
# test_index の書籍マッピング:
#   master_index = (test_index - 1) % N_BOOKS   (書籍が1周するたびにリセット)
#   round_index  = (test_index - 1) // N_BOOKS + 1
# 例: N_BOOKS=10 のとき
#   1〜10  → 書籍0〜9 (ラウンド1)
#   11〜20 → 書籍0〜9 (ラウンド2)
N_ROUNDS = 10  # 各書籍を何回繰り返すか（N_BOOKS × N_ROUNDS = END_INDEX）

# 評価しきい値 [mm]
ERROR_THRESHOLDS_MM = [1.0, 1.5, 2.0]

# offline入力としてコピーするファイル
# run_capture_and_pca_offline はこの2つを shot_dir から読む想定
INPUT_FILES = [
    "after_init_rgb.png",
    "after_init_depth.npy",
]

CONTACT_SHEET_STAGES = [
    ("05", "05 before column refine"),
    ("06a", "06a no seed guard"),
    ("06b", "06b seed guard"),
    ("07", "07 selected result"),
    ("08", "08 final_t_width_clip"),
    ("09", "09 one_sided post-final"),
    ("11", "11 RANSAC"),
    ("12", "12 post-RANSAC"),
    ("final", "final"),
]


class Tee:
    """stdout/stderrをターミナルとファイルの両方へ出すための簡易Tee。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def make_unique_run_dir(base_dir: Path, timestamp: str) -> Path:
    """
    captures/100test_offline/<timestamp> を作る。
    同じ秒に複数回実行して重複した場合は _001, _002 ... を付ける。
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir = base_dir / timestamp
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return run_dir

    for i in range(1, 1000):
        candidate = base_dir / f"{timestamp}_{i:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate

    raise RuntimeError(f"unique run dir を作れませんでした: {base_dir}/{timestamp}_XXX")


def load_master(master_json: Path):
    with open(master_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    books = []
    for i, item in enumerate(data):
        try:
            book_name = item["book_name"]
            book_width_mm = float(item["book_width"])
        except Exception as e:
            raise ValueError(f"master json の {i} 番目が不正です: {item}") from e

        books.append({
            "book_name": book_name,
            "book_width_mm": book_width_mm,
            "raw": item,
        })

    return books


def get_book_info_for_test_index(master_books, test_index: int):
    """
    1ラウンドで全書籍を1枚ずつカバーする構成。
    N_BOOKS=10 のとき:
      1〜10  -> master[0]〜master[9]  (ラウンド1)
      11〜20 -> master[0]〜master[9]  (ラウンド2)
      ...
    """
    n_books = len(master_books)
    master_index = (test_index - 1) % n_books
    repeat_index = (test_index - 1) // n_books + 1

    return master_index, repeat_index, master_books[master_index]


def safe_float_or_none(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def copy_offline_inputs(source_shot_dir: Path, run_shot_dir: Path):
    """
    元データ captures/100test/<idx> から、
    今回の実行先 captures/100test_offline/<timestamp>/<idx> に
    offline実行に必要な入力だけをコピーする。

    debug画像や過去のfinal.pngはコピーしない。
    """
    source_shot_dir = Path(source_shot_dir)
    run_shot_dir = Path(run_shot_dir)
    run_shot_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for name in INPUT_FILES:
        src = source_shot_dir / name
        dst = run_shot_dir / name

        if not src.exists():
            missing.append(str(src))
            continue

        shutil.copy2(src, dst)
        copied.append({
            "name": name,
            "source": str(src),
            "destination": str(dst),
            "size_bytes": int(dst.stat().st_size),
        })

    if missing:
        raise FileNotFoundError(
            "offline入力ファイルが不足しています:\n" + "\n".join(missing)
        )

    manifest = {
        "source_shot_dir": str(source_shot_dir),
        "run_shot_dir": str(run_shot_dir),
        "input_files": copied,
        "copy_policy": "copy only after_init_rgb.png and after_init_depth.npy; do not copy previous debug/final outputs",
    }

    save_json(run_shot_dir / "offline_input_manifest.json", manifest)
    return manifest


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_case_indices_from_env():
    raw = os.environ.get("CODEX_CASE_INDICES", "").strip()
    if not raw:
        return None
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def get_codex_result_root():
    raw = os.environ.get("CODEX_RESULT_ROOT", "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = BASE_DIR / p
    p.mkdir(parents=True, exist_ok=True)
    for name in ["contact_sheets", "logs", "representative_cases", "top_error_cases", "random_over_2mm_cases"]:
        (p / name).mkdir(parents=True, exist_ok=True)
    return p


def _find_stage_image(case_dir: Path, stage_id: str):
    if stage_id == "final":
        for p in [
            case_dir / "final.png",
            *sorted(case_dir.glob("mask*_final_valid_depth_region_overlay.png")),
            *sorted((case_dir / "debug_final_pointcloud_rgb").glob("*_final_valid_depth_region_overlay.png")),
            *sorted((case_dir / "debug_final_pointcloud_rgb").glob("*_final_pointcloud_projection_overlay.png")),
        ]:
            if p.exists():
                return p
        return None
    debug_dir = case_dir / "debug_step_by_step_pointcloud_rgb"
    matches = []
    if debug_dir.exists():
        matches = sorted(debug_dir.glob(f"*_{stage_id}_*_projection_overlay.png"))
        if not matches:
            matches = sorted(debug_dir.glob(f"*_{stage_id}*_projection_overlay.png"))
    if matches:
        return matches[0]

    # Some local branches do not emit step-by-step projection images. Fall back to
    # the closest stage overlays produced by the OCR/column debug code.
    band_dir = case_dir / "debug_ocr_band"
    fallback_patterns = {
        "05": ["*_ocr_band_candidate_overlay_kept_removed.png", "*_spine_column_overlay_kept_removed.png"],
        "06a": ["*_candidate_no_seed_guard_spine_column_overlay_kept_removed.png"],
        "06b": ["*_candidate_seed_guard_spine_column_overlay_kept_removed.png"],
        "07": ["*_spine_column_overlay_kept_removed.png"],
        "08": ["*_final_t_width_clip_spine_column_overlay_kept_removed.png"],
        "09": ["*_one_sided_post_final_column_prune_spine_column_overlay_kept_removed.png"],
        "11": ["../debug_ransac_spine_plane/*_overlay_kept_removed.png"],
        "12": ["../debug_post_ransac_plane_a95/*_overlay_kept_removed.png"],
    }
    for pattern in fallback_patterns.get(stage_id, []):
        matches = sorted(band_dir.glob(pattern))
        if matches:
            return matches[0]
    if stage_id in {"09", "11", "12"}:
        return _find_stage_image(case_dir, "final")
    return matches[0] if matches else None


def _valid_pixel_count_for_stage(case_dir: Path, stage_id: str):
    if stage_id == "final":
        candidates = [case_dir / "final_mask.png"]
        candidates.extend(sorted((case_dir / "debug_final_pointcloud_rgb").glob("*_final_valid_depth_region_mask.png")))
        for p in candidates:
            if not p.exists():
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return int(np.count_nonzero(img))
        return None
    debug_dir = case_dir / "debug_step_by_step_pointcloud_rgb"
    logs = sorted(debug_dir.glob(f"*_{stage_id}_*_log.json")) if debug_dir.exists() else []
    if not logs and debug_dir.exists():
        logs = sorted(debug_dir.glob(f"*_{stage_id}*_log.json"))
    for p in logs:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            v = data.get("valid_input_pixel_count")
            if v is None:
                v = data.get("point_count")
            return None if v is None else int(v)
        except Exception:
            continue
    band_dir = case_dir / "debug_ocr_band"
    fallback_logs = {
        "06a": ["*_candidate_no_seed_guard_spine_column_log.json"],
        "06b": ["*_candidate_seed_guard_spine_column_log.json"],
        "07": ["*_spine_column_log.json"],
        "08": ["*_final_t_width_clip_spine_column_log.json"],
        "09": ["*_one_sided_post_final_column_prune_spine_column_log.json"],
    }
    for pattern in fallback_logs.get(stage_id, []):
        for p in sorted(band_dir.glob(pattern)):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                v = data.get("valid_count_after") or data.get("area_after")
                return None if v is None else int(v)
            except Exception:
                continue
    if stage_id in {"11", "12"}:
        return _valid_pixel_count_for_stage(case_dir, "final")
    return None


def make_contact_sheet(case_dir: Path, out_path: Path, *, title: str = ""):
    tiles = []
    tile_w, tile_h = 420, 300
    for stage_id, label in CONTACT_SHEET_STAGES:
        img_path = _find_stage_image(case_dir, stage_id)
        if img_path is not None:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        else:
            img = None
        if img is None:
            tile = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
            cv2.putText(tile, "missing", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2, cv2.LINE_AA)
        else:
            h, w = img.shape[:2]
            scale = min(tile_w / max(w, 1), (tile_h - 48) / max(h, 1))
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            tile = np.full((tile_h, tile_w, 3), 255, dtype=np.uint8)
            y0 = 48 + (tile_h - 48 - nh) // 2
            x0 = (tile_w - nw) // 2
            tile[y0:y0 + nh, x0:x0 + nw] = resized
        count = _valid_pixel_count_for_stage(case_dir, stage_id)
        header = f"{label}"
        sub = f"valid={count}" if count is not None else "valid=n/a"
        cv2.rectangle(tile, (0, 0), (tile_w, 46), (30, 30, 30), -1)
        cv2.putText(tile, header, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(tile, sub, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    rows = []
    for i in range(0, len(tiles), 3):
        row_tiles = tiles[i:i + 3]
        while len(row_tiles) < 3:
            row_tiles.append(np.full((tile_h, tile_w, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(row_tiles))
    sheet = np.vstack(rows)
    if title:
        title_bar = np.full((42, sheet.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(title_bar, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        sheet = np.vstack([title_bar, sheet])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def copy_case_outputs_to_codex(case_dir: Path, codex_result_root: Path, test_index: int, row: dict):
    if codex_result_root is None:
        return {}
    copied = {}
    case_name = f"case_{test_index:03d}"
    log_src = case_dir / "offline_run_console.log"
    if log_src.exists():
        log_dst = codex_result_root / "logs" / f"case_{test_index:03d}_console.log"
        shutil.copy2(log_src, log_dst)
        copied["console_log"] = str(log_dst)
    sheet_path = codex_result_root / "contact_sheets" / f"case_{test_index:03d}_contact_sheet.png"
    try:
        make_contact_sheet(
            case_dir,
            sheet_path,
            title=f"case {test_index:03d} | {row.get('book_name', '')} | status={row.get('status')}",
        )
        copied["contact_sheet"] = str(sheet_path)
    except Exception as e:
        copied["contact_sheet_error"] = str(e)

    if test_index in {6, 7, 15, 26, 29}:
        dst_dir = codex_result_root / "representative_cases" / str(test_index)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ["case_result.json", "final.png", "final_mask.png", "offline_run_console.log"]:
            src = case_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)
        if "contact_sheet" in copied:
            shutil.copy2(copied["contact_sheet"], dst_dir / f"case_{test_index:03d}_contact_sheet.png")
        copied["representative_case_dir"] = str(dst_dir)
    return copied


def main():
    case_indices = parse_case_indices_from_env()
    codex_result_root = get_codex_result_root()
    print("\n===== BOOK WIDTH OFFLINE EVAL START =====")
    print(f"source test dir : {TEST_BASE_DIR}")
    print(f"offline base dir: {OFFLINE_BASE_DIR}")
    print(f"master json     : {MASTER_JSON}")
    print(f"sam_device      : {SAM_DEVICE}")
    print(f"test range      : {START_INDEX} to {END_INDEX}")
    if case_indices is not None:
        print(f"case indices    : {case_indices}")
    if codex_result_root is not None:
        print(f"codex result root: {codex_result_root}")
    print("=========================================\n")

    master_books = load_master(MASTER_JSON)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root_dir = make_unique_run_dir(OFFLINE_BASE_DIR, timestamp)

    out_csv = run_root_dir / f"book_width_eval_results_{run_root_dir.name}.csv"
    out_json = run_root_dir / f"book_width_eval_results_{run_root_dir.name}.json"
    out_summary = run_root_dir / f"book_width_eval_summary_{run_root_dir.name}.json"
    run_config_json = run_root_dir / "eval_run_config.json"

    run_config = {
        "timestamp": timestamp,
        "run_root_dir": str(run_root_dir),
        "source_test_base_dir": str(TEST_BASE_DIR),
        "offline_base_dir": str(OFFLINE_BASE_DIR),
        "master_json": str(MASTER_JSON),
        "sam_device": SAM_DEVICE,
        "start_index": START_INDEX,
        "end_index": END_INDEX,
        "n_rounds": N_ROUNDS,
        "error_thresholds_mm": ERROR_THRESHOLDS_MM,
        "input_files": INPUT_FILES,
        "output_structure": "captures/100test_offline/<timestamp>/<test_index>/",
    }
    save_json(run_config_json, run_config)

    print(f"今回のoffline実行保存先: {run_root_dir}")
    print(f"設定JSON: {run_config_json}")

    results = []
    indices_to_run = case_indices if case_indices is not None else list(range(START_INDEX, END_INDEX + 1))
    total_trials = len(indices_to_run)

    case_notes_csv = codex_result_root / "case_notes.csv" if codex_result_root is not None else None
    if case_notes_csv is not None and not case_notes_csv.exists():
        with open(case_notes_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "test_index", "book_name", "gt_width_mm", "pred_width_mm",
                "abs_error_mm", "status", "collapsed_stage", "visual_note",
                "contact_sheet_path", "run_shot_dir",
            ])
            writer.writeheader()

    for test_index in indices_to_run:
        source_shot_dir = TEST_BASE_DIR / str(test_index)
        run_shot_dir = run_root_dir / str(test_index)
        run_shot_dir.mkdir(parents=True, exist_ok=True)

        case_log_path = run_shot_dir / "offline_run_console.log"

        # デフォルト値。例外時に未定義にならないようにする。
        master_index = None
        repeat_index = None
        book_name = ""
        gt_width_mm = None
        pred_width_mm = None
        abs_error_mm = None
        returned_shot_dir = None
        elapsed = 0.0
        row = None

        with open(case_log_path, "w", encoding="utf-8") as log_f:
            tee_out = Tee(sys.stdout, log_f)
            tee_err = Tee(sys.stderr, log_f)

            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                start = time.perf_counter()

                try:
                    master_index, repeat_index, book_info = get_book_info_for_test_index(
                        master_books,
                        test_index,
                    )

                    book_name = book_info["book_name"]
                    gt_width_mm = book_info["book_width_mm"]

                    print("\n" + "=" * 60)
                    print(f"test_index       : {test_index}")
                    print(f"master_index     : {master_index}")
                    print(f"repeat_index     : {repeat_index}")
                    print(f"book_name        : {book_name}")
                    print(f"gt_width_mm      : {gt_width_mm}")
                    print(f"source_shot_dir  : {source_shot_dir}")
                    print(f"run_shot_dir     : {run_shot_dir}")
                    print(f"case_console_log : {case_log_path}")
                    print("=" * 60)

                    if not source_shot_dir.exists():
                        raise FileNotFoundError(f"source_shot_dir が存在しません: {source_shot_dir}")

                    # 重要: 元の100testを直接使わず、timestamp配下へ入力だけコピーして実行する
                    input_manifest = copy_offline_inputs(
                        source_shot_dir=source_shot_dir,
                        run_shot_dir=run_shot_dir,
                    )
                    print("✔ copied offline inputs")
                    print(json.dumps(input_manifest, ensure_ascii=False, indent=2))

                    theta_rad, p_min, pred_width_mm, returned_shot_dir = run_capture_and_pca_offline(
                        query=book_name,
                        shot_dir=run_shot_dir.resolve(),
                        sam_device=SAM_DEVICE,
                        encoder_path=str(BASE_DIR / "models" / "sam_vit_h_4b8939.encoder.onnx"),
                        decoder_path=str(BASE_DIR / "models" / "sam_vit_h_4b8939.decoder.onnx"),
                        interactive=False,
                        show_pointcloud_gui=False,
                        save_pointcloud_debug=True,
                    )

                    elapsed = time.perf_counter() - start
                    pred_width_mm = safe_float_or_none(pred_width_mm)

                    if pred_width_mm is None:
                        raise ValueError("pred_width_mm が None または float 変換不可です。")

                    abs_error_mm = abs(pred_width_mm - gt_width_mm)

                    row = {
                        "test_index": test_index,
                        "master_index": master_index,
                        "repeat_index": repeat_index,
                        "book_name": book_name,
                        "gt_book_width_mm": gt_width_mm,
                        "pred_book_width_mm": pred_width_mm,
                        "abs_error_mm": abs_error_mm,
                        "elapsed_sec": elapsed,
                        "status": "success",
                        "error": "",
                        "source_shot_dir": str(source_shot_dir),
                        "run_shot_dir": str(run_shot_dir),
                        "returned_shot_dir": str(returned_shot_dir),
                        "case_console_log": str(case_log_path),
                    }

                    print("\n--- RESULT ---")
                    print(f"pred_width_mm : {pred_width_mm:.3f}")
                    print(f"gt_width_mm   : {gt_width_mm:.3f}")
                    print(f"abs_error_mm  : {abs_error_mm:.3f}")
                    print(f"elapsed_sec   : {elapsed:.3f}")
                    print(f"outputs       : {run_shot_dir}")

                except Exception as e:
                    elapsed = time.perf_counter() - start

                    # book_info が取れる前に落ちる可能性もあるので保険
                    if not book_name:
                        try:
                            master_index, repeat_index, book_info = get_book_info_for_test_index(
                                master_books,
                                test_index,
                            )
                            book_name = book_info["book_name"]
                            gt_width_mm = book_info["book_width_mm"]
                        except Exception:
                            book_name = ""
                            gt_width_mm = None

                    row = {
                        "test_index": test_index,
                        "master_index": master_index,
                        "repeat_index": repeat_index,
                        "book_name": book_name,
                        "gt_book_width_mm": gt_width_mm,
                        "pred_book_width_mm": None,
                        "abs_error_mm": None,
                        "elapsed_sec": elapsed,
                        "status": "fail",
                        "error": str(e),
                        "source_shot_dir": str(source_shot_dir),
                        "run_shot_dir": str(run_shot_dir),
                        "returned_shot_dir": None,
                        "case_console_log": str(case_log_path),
                    }

                    print("\n❌ FAILED")
                    print(f"test_index      : {test_index}")
                    print(f"book_name       : {book_name}")
                    print(f"source_shot_dir : {source_shot_dir}")
                    print(f"run_shot_dir    : {run_shot_dir}")
                    print(f"error           : {e}")
                    traceback.print_exc()

                finally:
                    if row is not None:
                        save_json(run_shot_dir / "case_result.json", row)

        results.append(row)

        # 途中で落ちても結果が残るように毎回保存
        save_results(results, out_csv, out_json)

        copied_outputs = {}
        try:
            copied_outputs = copy_case_outputs_to_codex(run_shot_dir, codex_result_root, test_index, row)
        except Exception as e:
            copied_outputs = {"copy_error": str(e)}
            traceback.print_exc()

        if case_notes_csv is not None and row is not None:
            with open(case_notes_csv, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "test_index", "book_name", "gt_width_mm", "pred_width_mm",
                    "abs_error_mm", "status", "collapsed_stage", "visual_note",
                    "contact_sheet_path", "run_shot_dir",
                ])
                writer.writerow({
                    "test_index": row.get("test_index"),
                    "book_name": row.get("book_name"),
                    "gt_width_mm": row.get("gt_book_width_mm"),
                    "pred_width_mm": row.get("pred_book_width_mm"),
                    "abs_error_mm": row.get("abs_error_mm"),
                    "status": row.get("status"),
                    "collapsed_stage": "",
                    "visual_note": "pending manual visual inspection",
                    "contact_sheet_path": copied_outputs.get("contact_sheet", ""),
                    "run_shot_dir": row.get("run_shot_dir"),
                })

    summary = make_summary(results, total_trials)
    save_summary(summary, out_summary)
    if codex_result_root is not None:
        shutil.copy2(out_csv, codex_result_root / "eval_results.csv")
        shutil.copy2(out_summary, codex_result_root / "eval_summary.json")
        shutil.copy2(run_config_json, codex_result_root / "offline_eval_run_config.json")

    print_summary(summary)

    print("\n保存先:")
    print(f"RUN ROOT: {run_root_dir}")
    print(f"CSV     : {out_csv}")
    print(f"JSON    : {out_json}")
    print(f"SUMMARY : {out_summary}")
    print("=========================================\n")


def make_summary(results, total_trials: int):
    success_results = [r for r in results if r["status"] == "success"]
    fail_results = [r for r in results if r["status"] != "success"]

    errors = [
        r["abs_error_mm"]
        for r in success_results
        if r["abs_error_mm"] is not None
    ]

    threshold_counts = {}
    for th in ERROR_THRESHOLDS_MM:
        count = sum(1 for e in errors if 0.0 <= e <= th)
        threshold_counts[f"within_{th:.1f}mm"] = {
            "count": count,
            "denominator": total_trials,
            "rate": count / total_trials if total_trials > 0 else None,
        }

    if errors:
        mean_abs_error = sum(errors) / len(errors)
        max_abs_error = max(errors)
        min_abs_error = min(errors)
    else:
        mean_abs_error = None
        max_abs_error = None
        min_abs_error = None

    summary = {
        "total_trials": total_trials,
        "success_count": len(success_results),
        "fail_count": len(fail_results),
        "threshold_counts": threshold_counts,
        "mean_abs_error_mm_success_only": mean_abs_error,
        "min_abs_error_mm_success_only": min_abs_error,
        "max_abs_error_mm_success_only": max_abs_error,
        "results": results,
    }

    return summary


def save_results(results, out_csv: Path, out_json: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_index",
        "master_index",
        "repeat_index",
        "book_name",
        "gt_book_width_mm",
        "pred_book_width_mm",
        "abs_error_mm",
        "elapsed_sec",
        "status",
        "error",
        "source_shot_dir",
        "run_shot_dir",
        "returned_shot_dir",
        "case_console_log",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_summary(summary, out_summary: Path):
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def print_summary(summary):
    print("\n\n===== BOOK WIDTH OFFLINE EVAL SUMMARY =====")

    total_trials = summary["total_trials"]

    print(f"総試行回数 : {total_trials}")
    print(f"成功回数   : {summary['success_count']} / {total_trials}")
    print(f"失敗回数   : {summary['fail_count']} / {total_trials}")
    print("")

    for th in ERROR_THRESHOLDS_MM:
        key = f"within_{th:.1f}mm"
        item = summary["threshold_counts"][key]
        count = item["count"]
        denom = item["denominator"]
        rate = item["rate"]

        if rate is None:
            rate_text = "None"
        else:
            rate_text = f"{rate * 100:.1f}%"

        print(f"0〜{th:.1f}mm以内: {count} / {denom} ({rate_text})")

    print("")

    mean_abs_error = summary["mean_abs_error_mm_success_only"]
    min_abs_error = summary["min_abs_error_mm_success_only"]
    max_abs_error = summary["max_abs_error_mm_success_only"]

    if mean_abs_error is not None:
        print(f"平均絶対誤差[成功のみ] : {mean_abs_error:.3f} mm")
        print(f"最小絶対誤差[成功のみ] : {min_abs_error:.3f} mm")
        print(f"最大絶対誤差[成功のみ] : {max_abs_error:.3f} mm")
    else:
        print("平均絶対誤差[成功のみ] : None")

    print("===========================================\n")


if __name__ == "__main__":
    main()
