#!/usr/bin/env python3
"""Latest SAM3 book-width offline evaluator, based on offline_pointcloud_debug.

【書籍用】1枚の画像には全品目が並んで写っている想定で、画像1枚に対して master json の
全品目を query として順に認識させる。
  評価数 = 画像枚数(N_SHOTS) × 品目数(len(master_books))
5枚 × 20種 = 100評価。撮影は5回で済む。

2026-08-22: 本ファイルはoffline_pointcloud_debug_SAM3_catheter.py(カテーテル用に
TEST_BASE_DIR/MASTER_JSON等を差し替えた版)から分離した書籍用オリジナル設定の復元版。
TEST_BASE_DIRはcaptures/100test/を指す。captures/100test/の各写真は「1品目につき専用5枚」
という撮影意図(offline_100test_SAM3.py参照)ではあるものの、実際の写真自体は棚全体・
全20品目が写っている(ユーザー確認済み)ため、本スクリプトが前提とする「1枚に全品目」
という条件は満たしている。既定ではN_SHOTS=5枚(1〜5番)だけを使い全20品目でクエリするが、
--shotsで使う写真番号を変更できる(例: --shots 1-100 なら全100枚×20品目=2000評価)。
"""

from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_sam2_width import (
    prepare_offline_shot,
    run_capture_and_pca_offline_sam3_refined_sam2_width,
)
from detection.pro_handbook.sam3_runtime.integration_service_manager import (
    Sam3ServiceSession,
)
from pathlib import Path
import argparse
import json
import csv
import statistics
import time
import traceback
from datetime import datetime
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr


# =========================
# 設定
# =========================
BASE_DIR = Path(__file__).resolve().parent

# 元の入力データ（<N>/after_init_rgb.png ... の連番フォルダ）
# captures/100test/を使う(各写真は棚全体・全20品目が写っているため、
# 「1枚に全品目」という本スクリプトの前提を満たす)。
TEST_BASE_DIR = BASE_DIR / "captures" / "100test"

# offline実行結果の保存先(offline_100test_SAM3.pyのcaptures/100test_offline/とは
# 別物。評価方式(1枚に全品目クエリ)が違うため出力先も分ける)
OFFLINE_BASE_DIR = BASE_DIR / "captures" / "100test_pointcloud_debug_offline"

MASTER_JSON = BASE_DIR / "master_100test.json"

SAM_DEVICE = "gpu"

# 撮影した画像の枚数。各画像に対して master json の全品目を試す
N_SHOTS = 5

# 評価しきい値 [mm]
ERROR_THRESHOLDS_MM = [1.0, 1.5, 2.0]

# offline入力としてコピーするファイル
# run_capture_and_pca_offline はこの2つを shot_dir から読む想定
INPUT_FILES = [
    "after_init_rgb.png",
    "after_init_depth.npy",
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
    captures/offline_POINTCLOUD_DEBUG_SAM3_<timestamp> を作る。
    同じ秒に複数回実行して重複した場合は _001, _002 ... を付ける。
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    run_dir = base_dir / f"{timestamp}"
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
            "master_index": i,          # 絞り込んでも元の通し番号を保てるよう持たせる
            "book_name": book_name,
            "book_width_mm": book_width_mm,
            "raw": item,
        })

    return books


def safe_dir_name(name: str) -> str:
    """book_name をフォルダ名に使えるよう、記号を _ に置き換える。"""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def build_trials(books, shot_ids):
    """
    (shot_id, book_info) の全組み合わせを作る。

    shot_ids=[1,2] / 20種 なら 40件。
    同じ画像を使い回して query だけ変える構成。
    """
    trials = []
    for shot_id in shot_ids:
        for book_info in books:
            trials.append((shot_id, book_info))
    return trials


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
        "copy_policy": "copy only saved RGB-D inputs; rerun SAM3, offline OCR, refinement, Depth/RANSAC/PCA, and SAM2-compatible width",
    }

    save_json(run_shot_dir / "offline_input_manifest.json", manifest)
    return manifest


def prepare_shot_once(source_shot_dir: Path, prepared_shot_dir: Path):
    """
    画像1枚につき1回だけ、query に依存しない処理（OCR・SAM3マスク生成）を済ませる。

    同じ画像を20種の query で処理するため、これを試行ごとにやり直すと
    OCRとSAM3を19回ぶん無駄に計算することになる。生成物は各試行のフォルダへ
    コピーされるので、出力の中身は毎回計算した場合と同じになる。
    """
    copy_offline_inputs(
        source_shot_dir=source_shot_dir,
        run_shot_dir=prepared_shot_dir,
    )
    started = time.perf_counter()
    prepared = prepare_offline_shot(
        prepared_shot_dir.resolve(),
        sam_device=SAM_DEVICE,
    )
    prepared["wall_seconds"] = float(time.perf_counter() - started)
    return prepared


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the latest SAM3 recognition on saved RGB-D shots "
                    "(all master items per shot)."
    )
    parser.add_argument(
        "--shots", type=str,
        help="使う画像番号。例: --shots 1,3 / --shots 1-3 （既定: 1〜N_SHOTS）",
    )
    parser.add_argument(
        "--book-index", type=int, action="append",
        help="master json の何番目だけを試すか（0始まり、複数指定可）",
    )
    parser.add_argument(
        "--external-service",
        action="store_true",
        help="require an already-ready service; do not start one",
    )
    return parser.parse_args()


def parse_shot_ids(spec: str | None):
    """'1,3' や '1-3' を [1,3] / [1,2,3] に変換する。未指定なら 1〜N_SHOTS。"""
    if spec is None:
        return list(range(1, N_SHOTS + 1))

    shot_ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            shot_ids.extend(range(int(lo), int(hi) + 1))
        else:
            shot_ids.append(int(part))

    if not shot_ids:
        raise ValueError(f"--shots の指定が空です: {spec!r}")
    if any(s < 1 for s in shot_ids):
        raise ValueError(f"--shots は1以上を指定してください: {spec!r}")
    return sorted(set(shot_ids))


def filter_books(master_books, book_indices):
    """--book-index が指定されていればその品目だけに絞る。"""
    if not book_indices:
        return master_books
    for i in book_indices:
        if not 0 <= i < len(master_books):
            raise IndexError(
                f"--book-index {i} は範囲外です（0〜{len(master_books) - 1}）"
            )
    return [master_books[i] for i in sorted(set(book_indices))]


def main():
    args = parse_args()
    shot_ids = parse_shot_ids(args.shots)

    master_books = load_master(MASTER_JSON)
    books = filter_books(master_books, args.book_index)
    trials = build_trials(books, shot_ids)
    total_trials = len(trials)

    print("\n===== BOOK WIDTH OFFLINE EVAL START =====")
    print(f"source test dir : {TEST_BASE_DIR}")
    print(f"offline base dir: {OFFLINE_BASE_DIR}")
    print(f"master json     : {MASTER_JSON}")
    print(f"sam_device      : {SAM_DEVICE}")
    print(f"shots           : {shot_ids}")
    print(f"books           : {len(books)} / {len(master_books)} 種")
    print(f"total trials    : {len(shot_ids)} 枚 × {len(books)} 種 = {total_trials}")
    print("=========================================\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root_dir = make_unique_run_dir(OFFLINE_BASE_DIR, timestamp)

    out_csv = run_root_dir / "results.csv"
    out_json = run_root_dir / "results.json"
    out_summary = run_root_dir / "summary.json"
    run_log = run_root_dir / "run.log"
    run_config_json = run_root_dir / "eval_run_config.json"

    run_config = {
        "timestamp": timestamp,
        "run_root_dir": str(run_root_dir),
        "source_test_base_dir": str(TEST_BASE_DIR),
        "offline_base_dir": str(OFFLINE_BASE_DIR),
        "master_json": str(MASTER_JSON),
        "sam_device": SAM_DEVICE,
        "shot_ids": shot_ids,
        "n_books": len(books),
        "book_names": [b["book_name"] for b in books],
        "total_trials": total_trials,
        "error_thresholds_mm": ERROR_THRESHOLDS_MM,
        "input_files": INPUT_FILES,
        "output_structure": "<timestamp>/shot<shot_id>/<master_index>_<book_name>/",
        "recognition_api": "run_capture_and_pca_offline_sam3_refined_sam2_width",
        "query_independent_stage": (
            "OCR と SAM3 マスク生成は query に依存しないため画像ごとに1回だけ実行し、"
            "生成物を各試行のフォルダへコピーして使い回す"
        ),
        "service_policy": "external only" if args.external_service else "reuse ready service, otherwise start and stop owned service",
    }
    save_json(run_config_json, run_config)

    print(f"今回のoffline実行保存先: {run_root_dir}")
    print(f"設定JSON: {run_config_json}")

    results = []
    # 画像ごとの前処理（OCR・SAM3）の置き場。試行フォルダとは分けておく。
    prepared_root = run_root_dir / "_prepared"
    prepared_by_shot: dict[int, dict] = {}
    prepare_records = []
    service_session = Sam3ServiceSession()
    service_info = {
        "endpoint": service_session.endpoint,
        "external_service_required": bool(args.external_service),
    }
    if args.external_service:
        from detection.pro_handbook.sam3_runtime.integration_service_manager import _health
        reachable, ready, payload = _health(service_session.endpoint)
        if not (reachable and ready):
            raise RuntimeError(
                f"SAM3 service is not ready at {service_session.endpoint}: {payload}"
            )
        service_info.update({"borrowed": True, "health": payload})
    else:
        payload = service_session.ensure_ready()
        service_info.update(
            {
                "borrowed": not service_session.started_by_this_process,
                "started_by_script": service_session.started_by_this_process,
                "owned_pid": service_session.owned_pid,
                "health": payload,
            }
        )
    save_json(run_root_dir / "sam3_service_session.json", service_info)
    run_log.write_text(
        json.dumps(
            {"run_root": str(run_root_dir), "config": run_config, "service": service_info},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    for trial_no, (shot_id, book_info) in enumerate(trials, start=1):
        master_index = book_info["master_index"]
        book_name = book_info["book_name"]
        gt_width_mm = book_info["book_width_mm"]

        source_shot_dir = TEST_BASE_DIR / str(shot_id)
        # 同じ画像を複数queryで処理するので、組み合わせごとに別フォルダへ出力する
        # （認識関数が shot_dir へ中間ファイルを書き込むため、共有すると上書きされる）
        run_shot_dir = (
            run_root_dir / f"shot{shot_id}" / f"{master_index:02d}_{safe_dir_name(book_name)}"
        )
        run_shot_dir.mkdir(parents=True, exist_ok=True)

        case_log_path = run_shot_dir / "offline_run_console.log"

        # デフォルト値。例外時に未定義にならないようにする。
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
                    print("\n" + "=" * 60)
                    print(f"trial            : {trial_no} / {total_trials}")
                    print(f"shot_id          : {shot_id}")
                    print(f"master_index     : {master_index}")
                    print(f"book_name        : {book_name}")
                    print(f"gt_width_mm      : {gt_width_mm}")
                    print(f"source_shot_dir  : {source_shot_dir}")
                    print(f"run_shot_dir     : {run_shot_dir}")
                    print(f"case_console_log : {case_log_path}")
                    print("=" * 60)

                    if not source_shot_dir.exists():
                        raise FileNotFoundError(f"source_shot_dir が存在しません: {source_shot_dir}")

                    # 重要: 元データを直接使わず、timestamp配下へ入力だけコピーして実行する
                    input_manifest = copy_offline_inputs(
                        source_shot_dir=source_shot_dir,
                        run_shot_dir=run_shot_dir,
                    )
                    print("✔ copied offline inputs")
                    print(json.dumps(input_manifest, ensure_ascii=False, indent=2))

                    # query に依存しない処理は、この画像で最初の試行のときだけ実行する
                    prepared = prepared_by_shot.get(shot_id)
                    if prepared is None:
                        print(f"\n--- shot{shot_id}: query非依存の前処理（OCR + SAM3）---")
                        prepared = prepare_shot_once(
                            source_shot_dir=source_shot_dir,
                            prepared_shot_dir=prepared_root / f"shot{shot_id}",
                        )
                        prepared_by_shot[shot_id] = prepared
                        prepare_records.append({
                            "shot_id": shot_id,
                            "source_shot_dir": str(source_shot_dir),
                            "prepared_shot_dir": prepared["prepared_shot_dir"],
                            "mask_count": prepared["mask_count"],
                            "prepare_seconds": prepared["wall_seconds"],
                            "first_trial_no": trial_no,
                        })
                        save_json(prepared_root / "prepare_timing.json", prepare_records)
                        print(
                            f"--- 前処理完了: {prepared['wall_seconds']:.3f} 秒 / "
                            f"マスク {prepared['mask_count']} 個 ---\n"
                        )

                    # 前処理の時間は elapsed_sec に含めない（試行ごとの認識時間を見るため）
                    start = time.perf_counter()

                    recognition = run_capture_and_pca_offline_sam3_refined_sam2_width(
                        query=book_name,
                        shot_dir=run_shot_dir.resolve(),
                        sam_device=SAM_DEVICE,
                        prepared=prepared,
                    )
                    theta_rad = float(recognition["roll_rad"])
                    p_min = recognition["point_3d"]
                    pred_width_mm = recognition["pred_book_width_mm"]
                    returned_shot_dir = recognition["returned_shot_dir"]

                    elapsed = time.perf_counter() - start
                    pred_width_mm = safe_float_or_none(pred_width_mm)

                    if pred_width_mm is None:
                        raise ValueError("pred_width_mm が None または float 変換不可です。")

                    abs_error_mm = abs(pred_width_mm - gt_width_mm)
                    signed_error_mm = pred_width_mm - gt_width_mm

                    row = {
                        "trial_no": trial_no,
                        "shot_id": shot_id,
                        "master_index": master_index,
                        "book_name": book_name,
                        "gt_book_width_mm": gt_width_mm,
                        "pred_book_width_mm": pred_width_mm,
                        "abs_error_mm": abs_error_mm,
                        "signed_error_mm": signed_error_mm,
                        "roll_rad": theta_rad,
                        "target_point_m": p_min,
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

                    row = {
                        "trial_no": trial_no,
                        "shot_id": shot_id,
                        "master_index": master_index,
                        "book_name": book_name,
                        "gt_book_width_mm": gt_width_mm,
                        "pred_book_width_mm": None,
                        "abs_error_mm": None,
                        "signed_error_mm": None,
                        "roll_rad": None,
                        "target_point_m": None,
                        "elapsed_sec": elapsed,
                        "status": "fail",
                        "error": str(e),
                        "source_shot_dir": str(source_shot_dir),
                        "run_shot_dir": str(run_shot_dir),
                        "returned_shot_dir": None,
                        "case_console_log": str(case_log_path),
                    }

                    print("\n❌ FAILED")
                    print(f"trial           : {trial_no} / {total_trials}")
                    print(f"shot_id         : {shot_id}")
                    print(f"book_name       : {book_name}")
                    print(f"source_shot_dir : {source_shot_dir}")
                    print(f"run_shot_dir    : {run_shot_dir}")
                    print(f"error           : {e}")
                    traceback.print_exc()

                finally:
                    if row is not None:
                        save_json(run_shot_dir / "case_result.json", row)

        results.append(row)
        with run_log.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 途中で落ちても結果が残るように毎回保存
        save_results(results, out_csv, out_json)

    summary = make_summary(results, total_trials)
    save_summary(summary, out_summary)

    print_summary(summary)

    if prepare_records:
        prepare_total = sum(r["prepare_seconds"] for r in prepare_records)
        trial_total = sum(r["elapsed_sec"] for r in results)
        print("\n--- query非依存の前処理（OCR + SAM3）---")
        print(f"実行回数   : {len(prepare_records)} 回（画像の枚数と同じ）")
        print(f"合計時間   : {prepare_total:.1f} 秒")
        print(f"試行の合計 : {trial_total:.1f} 秒")
        print(f"総計       : {prepare_total + trial_total:.1f} 秒")

    print("\n保存先:")
    print(f"RUN ROOT: {run_root_dir}")
    print(f"CSV     : {out_csv}")
    print(f"JSON    : {out_json}")
    print(f"SUMMARY : {out_summary}")
    print("=========================================\n")
    if not args.external_service:
        service_session.stop_if_owned()


def make_summary(results, total_trials: int):
    success_results = [r for r in results if r["status"] == "success"]
    fail_results = [r for r in results if r["status"] != "success"]

    errors = [
        r["abs_error_mm"]
        for r in success_results
        if r["abs_error_mm"] is not None
    ]
    signed_errors = [
        r["signed_error_mm"]
        for r in success_results
        if r["signed_error_mm"] is not None
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
        median_abs_error = statistics.median(errors)
        max_abs_error = max(errors)
        min_abs_error = min(errors)
        mean_signed_error = sum(signed_errors) / len(signed_errors)
    else:
        mean_abs_error = None
        median_abs_error = None
        max_abs_error = None
        min_abs_error = None
        mean_signed_error = None

    summary = {
        "total_trials": total_trials,
        "success_count": len(success_results),
        "fail_count": len(fail_results),
        "threshold_counts": threshold_counts,
        "mean_abs_error_mm_success_only": mean_abs_error,
        "median_abs_error_mm_success_only": median_abs_error,
        "min_abs_error_mm_success_only": min_abs_error,
        "max_abs_error_mm_success_only": max_abs_error,
        "mean_signed_error_mm_success_only": mean_signed_error,
        "underestimate_count": sum(e < 0 for e in signed_errors),
        "overestimate_count": sum(e > 0 for e in signed_errors),
        "per_book": make_per_book_summary(results),
        "failure_cases": [
            {
                "trial_no": r["trial_no"],
                "shot_id": r["shot_id"],
                "book_name": r["book_name"],
                "error": r["error"],
                "source_shot_dir": r["source_shot_dir"],
            }
            for r in fail_results
        ],
        "results": results,
    }

    return summary


def make_per_book_summary(results):
    """品目ごとに集計する。どの品目が苦手かを見るため。"""
    per_book = {}

    for r in results:
        key = r["book_name"]
        entry = per_book.setdefault(key, {
            "master_index": r["master_index"],
            "book_name": key,
            "gt_book_width_mm": r["gt_book_width_mm"],
            "n_trials": 0,
            "n_success": 0,
            "abs_errors_mm": [],
        })
        entry["n_trials"] += 1
        if r["status"] == "success" and r["abs_error_mm"] is not None:
            entry["n_success"] += 1
            entry["abs_errors_mm"].append(r["abs_error_mm"])

    for entry in per_book.values():
        errors = entry["abs_errors_mm"]
        entry["mean_abs_error_mm"] = sum(errors) / len(errors) if errors else None
        entry["max_abs_error_mm"] = max(errors) if errors else None
        # 同じ品目を複数枚で測ったときのばらつき（再現性の指標）
        entry["stdev_abs_error_mm"] = (
            statistics.stdev(errors) if len(errors) >= 2 else None
        )

    # master json の並び順で返す
    return sorted(per_book.values(), key=lambda e: e["master_index"])


def save_results(results, out_csv: Path, out_json: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial_no",
        "shot_id",
        "master_index",
        "book_name",
        "gt_book_width_mm",
        "pred_book_width_mm",
        "abs_error_mm",
        "signed_error_mm",
        "roll_rad",
        "target_point_m",
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
            row = dict(r)
            row["target_point_m"] = json.dumps(
                row.get("target_point_m"), ensure_ascii=False
            )
            writer.writerow(row)

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
    median_abs_error = summary["median_abs_error_mm_success_only"]
    min_abs_error = summary["min_abs_error_mm_success_only"]
    max_abs_error = summary["max_abs_error_mm_success_only"]

    if mean_abs_error is not None:
        print(f"平均絶対誤差[成功のみ] : {mean_abs_error:.3f} mm")
        print(f"中央値絶対誤差[成功のみ]: {median_abs_error:.3f} mm")
        print(f"最小絶対誤差[成功のみ] : {min_abs_error:.3f} mm")
        print(f"最大絶対誤差[成功のみ] : {max_abs_error:.3f} mm")
        print(
            f"平均符号付き誤差[成功のみ]: "
            f"{summary['mean_signed_error_mm_success_only']:.3f} mm"
        )
        print(f"過小推定: {summary['underestimate_count']}")
        print(f"過大推定: {summary['overestimate_count']}")
    else:
        print("平均絶対誤差[成功のみ] : None")

    print("\n--- 品目ごと ---")
    print(f"{'idx':>3} {'book_name':22} {'正解':>6} {'成功':>7} {'平均誤差':>9} {'最大誤差':>9} {'ばらつき':>9}")
    for e in summary["per_book"]:
        def fmt(v):
            return f"{v:9.3f}" if v is not None else f"{'-':>9}"

        print(
            f"{e['master_index']:3d} {e['book_name'][:22]:22} "
            f"{e['gt_book_width_mm']:6.1f} "
            f"{e['n_success']:3d}/{e['n_trials']:<3d} "
            f"{fmt(e['mean_abs_error_mm'])} {fmt(e['max_abs_error_mm'])} "
            f"{fmt(e['stdev_abs_error_mm'])}"
        )

    print("===========================================\n")


if __name__ == "__main__":
    main()
