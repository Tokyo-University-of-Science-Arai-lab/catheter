#!/usr/bin/env python3
"""SAM2時代の100回試験（captures/100test）を、現在のSAM3で再実行する。

2026-07-08 に SAM2 で行った試験
（captures/100test_offline/20260708_203730）と直接比較するためのもの。
入力画像・品目の並び・正解幅を当時と揃えてあるので、
結果CSVの test_index どうしがそのまま対応する。

構成は 5shot 版（offline_pointcloud_debug_SAM3.py）とは違う:
  画像1枚 = 1品目 = 1試行。10品目を1周として、それを10ラウンド繰り返す。
    master_index = (test_index - 1) % 品目数
    round_index  = (test_index - 1) // 品目数 + 1
  例: 1〜10 が品目0〜9（ラウンド1）、11〜20 が品目0〜9（ラウンド2）…

実行はリポジトリルートから:
    python offline_100test_SAM3.py --external-service
"""

from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_sam2_width import (
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
TEST_BASE_DIR = BASE_DIR / "captures" / "100test"

# offline実行結果の保存先（SAM2時代と同じ場所・同じ階層構造）
OFFLINE_BASE_DIR = BASE_DIR / "captures" / "100test_offline"

# 100test 専用のマスタ。master_20260216.json は品目が20種に増えており、
# 幅の値も当時と変わっているので流用できない。
MASTER_JSON = BASE_DIR / "master_100test.json"

SAM_DEVICE = "gpu"

START_INDEX = 1
END_INDEX = 100

# 評価しきい値 [mm]
ERROR_THRESHOLDS_MM = [1.0, 1.5, 2.0]

# offline入力としてコピーするファイル
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
    captures/100test_offline/<timestamp> を作る。
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
    """book_name（=query）と book_width（=正解幅）だけを読む。

    `_` で始まるキーは編集時のメモ用なので無視される。
    """
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
            "master_index": i,
            "book_name": book_name,
            "book_width_mm": book_width_mm,
            "raw": item,
        })

    return books


def get_book_info_for_test_index(master_books, test_index: int):
    """
    1ラウンドで全品目を1枚ずつカバーする構成。
    品目数=10 のとき:
      1〜10  -> master[0]〜master[9]  (ラウンド1)
      11〜20 -> master[0]〜master[9]  (ラウンド2)
    """
    n_books = len(master_books)
    master_index = (test_index - 1) % n_books
    round_index = (test_index - 1) // n_books + 1
    return master_index, round_index, master_books[master_index]


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


def save_json(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_case_spec(spec: str | None):
    """'5' や '1-20' や '1-10,51-60' を test_index のリストに変換する。"""
    if spec is None:
        return list(range(START_INDEX, END_INDEX + 1))

    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))

    if not indices:
        raise ValueError(f"--cases の指定が空です: {spec!r}")
    for i in indices:
        if not START_INDEX <= i <= END_INDEX:
            raise ValueError(f"--cases は {START_INDEX}〜{END_INDEX} の範囲です: {i}")
    return sorted(set(indices))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-run the SAM2-era 100-case test (captures/100test) with the current SAM3 pipeline."
    )
    parser.add_argument(
        "--cases", type=str,
        help="実行するケース番号。例: --cases 1-10 / --cases 1,5,7 （既定: 1〜100）",
    )
    parser.add_argument(
        "--master", type=str, default=None,
        help=f"使うマスタJSON（既定: {MASTER_JSON.name}）",
    )
    parser.add_argument(
        "--external-service",
        action="store_true",
        help="require an already-ready service; do not start one",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    master_json = Path(args.master).resolve() if args.master else MASTER_JSON
    test_indices = parse_case_spec(args.cases)

    master_books = load_master(master_json)
    n_books = len(master_books)

    if END_INDEX % n_books != 0:
        print(
            f"⚠ 警告: 品目数 {n_books} で {END_INDEX} ケースを割り切れません。"
            f"ラウンドの端数が出るのでマスタの品目数を確認してください。"
        )

    print("\n===== BOOK WIDTH OFFLINE EVAL START (100test / SAM3) =====")
    print(f"source test dir : {TEST_BASE_DIR}")
    print(f"offline base dir: {OFFLINE_BASE_DIR}")
    print(f"master json     : {master_json}")
    print(f"sam_device      : {SAM_DEVICE}")
    print(f"books           : {n_books} 種 × {END_INDEX // n_books} ラウンド")
    print(f"total cases     : {len(test_indices)}")
    print("==========================================================\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root_dir = make_unique_run_dir(OFFLINE_BASE_DIR, timestamp)

    out_csv = run_root_dir / f"book_width_eval_results_{timestamp}.csv"
    out_json = run_root_dir / f"book_width_eval_results_{timestamp}.json"
    out_summary = run_root_dir / f"book_width_eval_summary_{timestamp}.json"
    run_log = run_root_dir / "run.log"
    run_config_json = run_root_dir / "eval_run_config.json"

    run_config = {
        "timestamp": timestamp,
        "run_root_dir": str(run_root_dir),
        "source_test_base_dir": str(TEST_BASE_DIR),
        "offline_base_dir": str(OFFLINE_BASE_DIR),
        "master_json": str(master_json),
        "sam_device": SAM_DEVICE,
        "start_index": START_INDEX,
        "end_index": END_INDEX,
        "test_indices": test_indices,
        "n_books": n_books,
        "n_rounds": END_INDEX // n_books,
        "book_names": [b["book_name"] for b in master_books],
        "gt_widths_mm": [b["book_width_mm"] for b in master_books],
        "error_thresholds_mm": ERROR_THRESHOLDS_MM,
        "input_files": INPUT_FILES,
        "output_structure": "captures/100test_offline/<timestamp>/<test_index>/",
        "recognition_api": "run_capture_and_pca_offline_sam3_refined_sam2_width",
        "index_mapping": "master_index = (test_index - 1) % n_books, round_index = (test_index - 1) // n_books + 1",
        "baseline_for_comparison": "captures/100test_offline/20260708_203730 (SAM2, 旧 ~/pro_book)",
        "service_policy": "external only" if args.external_service else "reuse ready service, otherwise start and stop owned service",
    }
    save_json(run_config_json, run_config)

    print(f"今回のoffline実行保存先: {run_root_dir}")
    print(f"設定JSON: {run_config_json}")

    results = []
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

    total_cases = len(test_indices)

    for case_no, test_index in enumerate(test_indices, start=1):
        master_index, round_index, book_info = get_book_info_for_test_index(
            master_books, test_index
        )
        book_name = book_info["book_name"]
        gt_width_mm = book_info["book_width_mm"]

        source_shot_dir = TEST_BASE_DIR / str(test_index)
        # SAM2時代と同じ <timestamp>/<test_index>/ 構造にして、突き合わせを楽にする
        run_shot_dir = run_root_dir / str(test_index)
        run_shot_dir.mkdir(parents=True, exist_ok=True)

        case_log_path = run_shot_dir / "offline_run_console.log"

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
                    print(f"case             : {case_no} / {total_cases}")
                    print(f"test_index       : {test_index}")
                    print(f"master_index     : {master_index}")
                    print(f"round_index      : {round_index}")
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

                    start = time.perf_counter()

                    recognition = run_capture_and_pca_offline_sam3_refined_sam2_width(
                        query=book_name,
                        shot_dir=run_shot_dir.resolve(),
                        sam_device=SAM_DEVICE,
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
                        "test_index": test_index,
                        "master_index": master_index,
                        "round_index": round_index,
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
                        "test_index": test_index,
                        "master_index": master_index,
                        "round_index": round_index,
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
                    print(f"case            : {case_no} / {total_cases}")
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
        with run_log.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")

        # 途中で落ちても結果が残るように毎回保存
        save_results(results, out_csv, out_json)

    summary = make_summary(results, total_cases)
    save_summary(summary, out_summary)

    print_summary(summary)

    print("\n保存先:")
    print(f"RUN ROOT: {run_root_dir}")
    print(f"CSV     : {out_csv}")
    print(f"JSON    : {out_json}")
    print(f"SUMMARY : {out_summary}")
    print("==========================================================\n")
    if not args.external_service:
        service_session.stop_if_owned()


def make_summary(results, total_cases: int):
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
            "denominator": total_cases,
            "rate": count / total_cases if total_cases > 0 else None,
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
        "total_cases": total_cases,
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
                "test_index": r["test_index"],
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
        # 同じ品目を10ラウンド測ったときのばらつき（再現性の指標）
        entry["stdev_abs_error_mm"] = (
            statistics.stdev(errors) if len(errors) >= 2 else None
        )

    return sorted(per_book.values(), key=lambda e: e["master_index"])


def save_results(results, out_csv: Path, out_json: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_index",
        "master_index",
        "round_index",
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
    print("\n\n===== BOOK WIDTH OFFLINE EVAL SUMMARY (100test / SAM3) =====")

    total_cases = summary["total_cases"]

    print(f"総試行回数 : {total_cases}")
    print(f"成功回数   : {summary['success_count']} / {total_cases}")
    print(f"失敗回数   : {summary['fail_count']} / {total_cases}")
    print("")

    for th in ERROR_THRESHOLDS_MM:
        key = f"within_{th:.1f}mm"
        item = summary["threshold_counts"][key]
        count = item["count"]
        denom = item["denominator"]
        rate = item["rate"]

        rate_text = "None" if rate is None else f"{rate * 100:.1f}%"
        print(f"0〜{th:.1f}mm以内: {count} / {denom} ({rate_text})")

    print("")

    mean_abs_error = summary["mean_abs_error_mm_success_only"]

    if mean_abs_error is not None:
        print(f"平均絶対誤差[成功のみ] : {mean_abs_error:.3f} mm")
        print(f"中央値絶対誤差[成功のみ]: {summary['median_abs_error_mm_success_only']:.3f} mm")
        print(f"最小絶対誤差[成功のみ] : {summary['min_abs_error_mm_success_only']:.3f} mm")
        print(f"最大絶対誤差[成功のみ] : {summary['max_abs_error_mm_success_only']:.3f} mm")
        print(
            f"平均符号付き誤差[成功のみ]: "
            f"{summary['mean_signed_error_mm_success_only']:.3f} mm"
        )
        print(f"過小推定: {summary['underestimate_count']}")
        print(f"過大推定: {summary['overestimate_count']}")
    else:
        print("平均絶対誤差[成功のみ] : None")

    print("\n--- 品目ごと ---")
    print(f"{'idx':>3} {'book_name':30} {'正解':>6} {'成功':>7} {'平均誤差':>9} {'最大誤差':>9} {'ばらつき':>9}")
    for e in summary["per_book"]:
        def fmt(v):
            return f"{v:9.3f}" if v is not None else f"{'-':>9}"

        print(
            f"{e['master_index']:3d} {e['book_name'][:30]:30} "
            f"{e['gt_book_width_mm']:6.1f} "
            f"{e['n_success']:3d}/{e['n_trials']:<3d} "
            f"{fmt(e['mean_abs_error_mm'])} {fmt(e['max_abs_error_mm'])} "
            f"{fmt(e['stdev_abs_error_mm'])}"
        )

    print("=============================================================\n")


if __name__ == "__main__":
    main()
