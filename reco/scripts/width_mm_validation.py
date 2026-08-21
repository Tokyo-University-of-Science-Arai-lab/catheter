#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reco/*/depth_shots/ の実撮影(RGB+depth+camera_params.json)に対し、get_book_points.py
本番の3D計測パイプライン(_run_recognition_core_like_offline、SAM→OCR→PCA→実mm幅)を
実際に呼び出し、推定した実mm幅をマスタJSON(reco/master_catheter_reco.json)の
book_width(mm、正解値)と比較する。

depth_shots画像はOCR都合で撮影時のセンサー向きから180度回転させて保存されている
(reco/README.md参照)。RGB/depthを同時にnp.rot90(arr, 2)で撮影時の向きに戻した上で、
スクラッチ作業フォルダにコピーしてから処理する(OCRサブプロセスがshot_dir直下の
after_init_rgb.pngをディスクから直接読むため、in-memory配列だけでなく実ファイルも
揃えて回転を一致させる必要がある)。camera_params.jsonの内部パラメータはget_book_points.py
のオフライン版が使う固定値と完全一致することを確認済みなので、そのまま使う。

reco/*/内のオリジナルdepth_shots/は一切変更しない(読み取り専用)。

実行:
    cd ~/pro_book/pro_hand_book_python
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset diagonal-40
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset stand-100
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset diagonal-40 --only Target_R --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# onnxruntime-gpu がCUDA/cuDNN等を見つけられるよう、venv内nvidia-*パッケージの
# libディレクトリをLD_LIBRARY_PATHに積んでから自分自身を再起動する
# (run_sam_eval.py・recognition_accuracy_test.pyと同じ手法)。
# ---------------------------------------------------------------------------
def _ensure_cuda_ld_library_path() -> None:
    site_packages = Path(sys.executable).resolve().parents[1] / "lib" / "python3.10" / "site-packages"
    nvidia_dir = site_packages / "nvidia"
    if not nvidia_dir.is_dir():
        return
    lib_dirs = [str(p) for p in sorted(nvidia_dir.glob("*/lib")) if p.is_dir()]
    if not lib_dirs:
        return
    joined = ":".join(lib_dirs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if all(d not in cur for d in lib_dirs):
        os.environ["LD_LIBRARY_PATH"] = joined + (":" + cur if cur else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_ld_library_path()

import numpy as np  # noqa: E402
import cv2  # noqa: E402

RECO_ROOT = Path(__file__).resolve().parents[1]  # .../pro_hand_book_python/reco
REPO_ROOT = RECO_ROOT.parent  # .../pro_hand_book_python
sys.path.insert(0, str(REPO_ROOT))

MASTER_JSON = RECO_ROOT / "master_catheter_reco.json"

SAM_ENCODER = REPO_ROOT / "models" / "sam_vit_h_4b8939.encoder.onnx"
SAM_DECODER = REPO_ROOT / "models" / "sam_vit_h_4b8939.decoder.onnx"

# diagonal-40/depth_shots/<name> のフォルダ名(末尾の _L/_R を除いた品目部分) ->
# マスタJSONのbook_name(multikey_matcherのqueryとして渡すキー)。
DIAGONAL_PRODUCT_TO_BOOKNAME = {
    "Target": "612104",
    "Transform": "ESC0407",
    "Transform305": "ESC0305",
    "Synchro": "SSTD215STR",
    "Excelsior XT-27": "XT275081",
    "Surpass": "FD45017",
    "AXS_DAC": "INC-11814-125",
    "AXS": "INC-11814-146",
    "pNOVUS": "MC1715000",
    "Phenom": "FG13160-0615-1S",
    "Ripride": "MAT-110-110",
    "orange": "M00345100950",
    "Excelsior-XT-17": "C1775ST",
    "Excelsior SL-10": "1681189",
    "neuroform": "EZAS3021",
    "Wallaby": "10CSW12612",
    "CHIKAI": "AIN-CHI-200R",
    "SHOURYU2": "SS2-40-070H",
    "Guidepost": "KMDA044178",
    "Esperance": "DAC6F115",
}


def load_master() -> dict[str, dict]:
    with open(MASTER_JSON, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {r["book_name"]: r for r in rows}


def product_base_name(shot_folder_name: str) -> str:
    for suffix in ("_L", "_R"):
        if shot_folder_name.endswith(suffix):
            return shot_folder_name[: -len(suffix)]
    return shot_folder_name


def prepare_derotated_shot(src_dir: Path, work_dir: Path) -> Path:
    """depth_shots/<name>/ の after_init_rgb.png + after_init_depth.npy を
    work_dir にコピーする。

    【2026-08-21 訂正】以前はここで180度回転(rot90x2)をかけていたが、これは誤りだった。
    depth_shots自体が本来のパイプラインが得る正しい向きのデータであり(ユーザー確認済み、
    生画像を直接目視しても正立・可読)、回転が必要なのは逆にannotations側(images/*.png、
    人間のアノテーション用)の方だった。回転をかけると本来正しい画像を180度反転させて
    OCR/認識に渡してしまうバグになっていたため、単純コピーに変更した。
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    rgb_candidates = sorted(src_dir.glob("*.png"))
    if not rgb_candidates:
        raise FileNotFoundError(f"{src_dir} にpng画像がありません")
    rgb_path = rgb_candidates[0]
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise FileNotFoundError(f"{rgb_path} を読み込めません")
    depth = np.load(src_dir / "after_init_depth.npy")

    cv2.imwrite(str(work_dir / "after_init_rgb.png"), rgb)
    np.save(work_dir / "after_init_depth.npy", depth)

    cam_params_src = src_dir / "camera_params.json"
    if cam_params_src.exists():
        shutil.copyfile(cam_params_src, work_dir / "camera_params_captured.json")

    return work_dir


def run_one(shot_name: str, src_dir: Path, query_book_name: str, work_root: Path,
            runner_kwargs: dict) -> dict:
    from detection.pro_handbook.sam_py_demo.get_book_points import run_capture_and_pca_offline

    work_dir = work_root / shot_name
    prepare_derotated_shot(src_dir, work_dir)

    t0 = time.time()
    theta_rad, target_point, book_width_mm, out_dir = run_capture_and_pca_offline(
        query=query_book_name,
        shot_dir=work_dir,
        encoder_path=str(SAM_ENCODER),
        decoder_path=str(SAM_DECODER),
        **runner_kwargs,
    )
    elapsed = time.time() - t0
    return {
        "shot": shot_name,
        "query": query_book_name,
        "book_width_mm_pred": book_width_mm,
        "elapsed_sec": round(elapsed, 2),
        "ok": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["diagonal-40", "stand-100"], required=True)
    ap.add_argument("--only", default=None, help="このshot名だけ処理する(スモークテスト用)")
    ap.add_argument("--limit", type=int, default=None, help="先頭N件だけ処理する")
    ap.add_argument("--sam-device", default="gpu")
    ap.add_argument("--sam-pts-side", default=None,
                    help="get_book_points.pyのsam_pts_sideを上書きする(例: 64,16)。"
                         "省略時はパイプライン既定値(32,8)のまま。")
    ap.add_argument("--work-suffix", default=None,
                    help="work_root/out_csvのファイル名に付与する接尾辞(例: _ptsdense)。"
                         "省略時は実行日時(例: _20260821_143000)を自動で付与し、"
                         "実行のたびにフォルダ/ファイルが区別できるようにする"
                         "(2026-08-21、ユーザー要望により変更)。")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.work_suffix is None:
        args.work_suffix = "_" + time.strftime("%Y%m%d_%H%M%S")

    master_by_bookname = load_master()

    runner_kwargs = dict(
        sam_device=args.sam_device,
        interactive=False,
        use_persistent_runtime=True,
        show_pointcloud_gui=False,
        save_pointcloud_debug=False,
        save_step_by_step_pointcloud_debug=False,
    )
    if args.sam_pts_side:
        a, b = args.sam_pts_side.split(",")
        runner_kwargs["sam_pts_side"] = (int(a), int(b))

    dataset_dir = RECO_ROOT / args.dataset
    depth_shots_dir = dataset_dir / "depth_shots"
    work_root = dataset_dir / f"width_eval_work{args.work_suffix}"
    out_csv = dataset_dir / f"width_eval_result{args.work_suffix}.csv"

    jobs: list[tuple[str, Path, str]] = []  # (shot_name, src_dir, query_book_name)

    if args.dataset == "diagonal-40":
        for shot_dir in sorted(depth_shots_dir.iterdir()):
            if not shot_dir.is_dir():
                continue
            base = product_base_name(shot_dir.name)
            book_name = DIAGONAL_PRODUCT_TO_BOOKNAME.get(base)
            if book_name is None:
                print(f"⚠ 品目マッピング未登録のためスキップ: {shot_dir.name} (base={base})")
                continue
            jobs.append((shot_dir.name, shot_dir, book_name))
    else:  # stand-100: 各棚配置に全20品目のqueryを試す
        for shot_dir in sorted(depth_shots_dir.iterdir()):
            if not shot_dir.is_dir():
                continue
            for book_name in master_by_bookname:
                jobs.append((f"{shot_dir.name}__{book_name}", shot_dir, book_name))

    if args.only:
        jobs = [j for j in jobs if j[0] == args.only or Path(j[1]).name == args.only]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"対象ジョブ数: {len(jobs)}")

    done_shots: set[str] = set()
    if out_csv.exists():
        with open(out_csv, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                done_shots.add(row["shot"])
        print(f"既存結果 {len(done_shots)}件をスキップ(resume)")

    fieldnames = ["shot", "query", "book_name_true", "display_name", "book_width_mm_true",
                  "book_width_mm_pred", "abs_error_mm", "elapsed_sec", "error"]
    write_header = not out_csv.exists()
    with open(out_csv, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, (shot_name, src_dir, book_name) in enumerate(jobs):
            if shot_name in done_shots:
                continue
            master_row = master_by_bookname.get(book_name, {})
            true_mm = master_row.get("book_width")
            print(f"[{i+1}/{len(jobs)}] {shot_name} (query={book_name}) ...", flush=True)
            row = {
                "shot": shot_name,
                "query": book_name,
                "book_name_true": book_name,
                "display_name": master_row.get("display_name", ""),
                "book_width_mm_true": true_mm,
                "book_width_mm_pred": "",
                "abs_error_mm": "",
                "elapsed_sec": "",
                "error": "",
            }
            try:
                result = run_one(shot_name, src_dir, book_name, work_root, runner_kwargs)
                row["book_width_mm_pred"] = round(float(result["book_width_mm_pred"]), 2)
                row["elapsed_sec"] = result["elapsed_sec"]
                if true_mm is not None:
                    row["abs_error_mm"] = round(abs(row["book_width_mm_pred"] - float(true_mm)), 2)
                print(f"  -> pred={row['book_width_mm_pred']}mm true={true_mm}mm "
                      f"err={row['abs_error_mm']} ({result['elapsed_sec']}s)")
            except Exception as e:  # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"  ✗ 失敗: {row['error']}")
                if args.verbose:
                    traceback.print_exc()
            writer.writerow(row)
            f.flush()

    print(f"\n✔ 結果 -> {out_csv}")


if __name__ == "__main__":
    main()
