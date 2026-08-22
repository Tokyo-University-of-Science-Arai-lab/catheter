#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reco/*/depth_shots/ の実撮影(RGB+depth+camera_params.json)に対し、get_book_points.py
本番の3D計測パイプライン(_run_recognition_core_like_offline、SAM→OCR→PCA→実mm幅)を
実際に呼び出し、推定した実mm幅をマスタJSON(reco/master_catheter_reco.json)の
book_width(mm、正解値)と比較する。

【2026-08-21 訂正】以前ここには「depth_shotsは180度回転させて保存されているので
撮影時の向きに戻す」と書かれ、実際にnp.rot90(arr, 2)をかけていたが、これは誤りだった。
depth_shots自体が本来のパイプラインが得る正しい向きのデータであり(ユーザー確認済み、
生画像を直接目視しても正立・可読)、回転が必要なのは逆にannotations側(images/*.png、
人間のアノテーション用)の方だった。この誤った回転により、OCR/識別が長時間にわたり
誤った向きの画像で動作していた(識別正答率が10〜22%まで悪化する原因になっていた)。
現在はdepth_shotsのRGB/depthをそのままコピーするだけで、回転処理は一切行わない。

reco/*/内のオリジナルdepth_shots/は一切変更しない(読み取り専用)。

作業フォルダ・結果CSVの名前には実行日時を必ず含める(2026-08-21、ユーザー要望)。
`--label`で意味のある名前も併用できる。

stand-100の各アイテムの作業フォルダ名は`{処理順}-{使用した画像番号}-{display_name}`
(例: `23-2-Target_XL`)。処理順(i+1)はこの実行内でのjobs一覧上の通し番号で、
以前の「認識番号」(=実行そのものの世代番号、例:7)とは別物(2026-08-21、ユーザー要望:
「認識した順番-使用した画像-display_name」で命名してほしい、との追加依頼への対応)。
CSVの`shot`列自体は従来通り`{画像番号}__{REF}`のまま(一意性・resume判定用)。
diagonal-40は1画像=1アイテムで曖昧さがないため、work_root直下のフォルダ名は
従来通りshot_dir.name(例: `Target_R`)のまま変更していない。

実行:
    cd ~/pro_book/pro_hand_book_python
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset diagonal-40
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset stand-100
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset stand-100 --label rotfix
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset diagonal-40 --only Target_R --verbose

【2026-08-22追加】--pipeline {simplified,legacy}
    本番採用した簡易版パイプライン(get_book_points_sam3_refined_sam2_width.py、SAM3マスク選択
    →depth外れ値除去→RANSAC平面フィット1回→SAM2互換幅算出の4段階、詳細はHANDOFF 8.2節参照)と、
    従来の複雑なパイプライン(get_book_points.py、07column_refine等9段階の後処理)を切り替えられる。
    既定は"simplified"(=現在Retrieval_integration_SAM3.py・R_I_SAM3_C.pyが実際に使っているものと
    同じ)。旧パイプラインでA/B比較したい場合は--pipeline legacyを指定する。
    --sam-pts-side・--depth-outlier-method・--no-fragmentation-handlingは旧パイプライン専用の
    オプションで、--pipeline simplifiedと併用すると警告を出して無視される(簡易版にはこれらに
    相当する後処理ステージ自体が無いため)。リトライ機構のバリエーション(RETRY_VARIANTS)も
    パイプラインごとに別内容(下記コード参照)。

    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset stand-100
        (既定で--pipeline simplified、本番と同じ挙動を評価)
    .pro_hand_book_fixed/bin/python3.10 reco/scripts/width_mm_validation.py --dataset stand-100 --pipeline legacy --label legacy_ab
        (旧パイプラインとのA/B比較用)
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


# ---------------------------------------------------------------------------
# 2026-08-21発見: PYTHONHASHSEEDが未固定だと(既定でPython起動ごとにランダム化される)、
# 識別スコアが完全にタイになる品目(OCR手がかりが皆無なオレンジ箱等)で、Hungarian法割当の
# タイブレークがdict/set反復順に依存し、プロセス起動のたびに選ばれるマスクが変わる実例を
# 確認した(同一shot・同一パラメータでも"selected id=1"と"selected id=21"のように結果が
# 変わった)。SAM3の生マスク・OCR認識テキスト自体は完全に決定的なことを確認済みで、
# 揺れの原因はハッシュランダム化のみ。1回のスクリプト実行内(リトライ含む)は同一プロセスの
# ため揺れないが、スクリプトを再実行して比較する場合(デバッグ・A/B比較)に結果が変わり
# うるため、再現性のためPYTHONHASHSEEDを固定して自分自身を再起動する。
# -----------------------------------------------------------------------------
def _ensure_fixed_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_fixed_hash_seed()

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


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def product_base_name(shot_folder_name: str) -> str:
    for suffix in ("_L", "_R"):
        if shot_folder_name.endswith(suffix):
            return shot_folder_name[: -len(suffix)]
    return shot_folder_name


def prepare_shot_files(src_dir: Path, work_dir: Path) -> Path:
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
            runner_kwargs: dict, folder_name: str | None = None, pipeline: str = "simplified") -> dict:
    """pipeline="legacy" は get_book_points.py(9段階の複雑な後処理、A/B比較用)、
    pipeline="simplified" は get_book_points_sam3_refined_sam2_width.py(本番採用済み、
    4段階、HANDOFF 8.2節参照)を呼ぶ。呼び出し先の関数シグネチャ・戻り値形式が異なるため
    ここで吸収し、呼び出し元には統一した辞書を返す。
    """
    work_dir = work_root / (folder_name or shot_name)
    prepare_shot_files(src_dir, work_dir)

    t0 = time.time()
    if pipeline == "legacy":
        from detection.pro_handbook.sam_py_demo.get_book_points import run_capture_and_pca_offline
        theta_rad, target_point, book_width_mm, out_dir = run_capture_and_pca_offline(
            query=query_book_name,
            shot_dir=work_dir,
            encoder_path=str(SAM_ENCODER),
            decoder_path=str(SAM_DECODER),
            **runner_kwargs,
        )
    else:
        from detection.pro_handbook.sam_py_demo.get_book_points_sam3_refined_sam2_width import (
            run_capture_and_pca_offline_sam3_refined_sam2_width,
        )
        result = run_capture_and_pca_offline_sam3_refined_sam2_width(
            query=query_book_name,
            shot_dir=work_dir,
            **runner_kwargs,
        )
        book_width_mm = result["pred_book_width_mm"]
    elapsed = time.time() - t0
    return {
        "shot": shot_name,
        "query": query_book_name,
        "book_width_mm_pred": book_width_mm,
        "elapsed_sec": round(elapsed, 2),
        "ok": True,
    }


# 2026-08-21、ユーザー要望: 「推定幅誤差が5mm以上ある場合は認識をやり直す」の試験導入。
# 【重要な設計上の前提】このスクリプトは正解幅(book_width、マスタJSON)が分かっている
# 評価専用の文脈でのみ動く。同じ静止画像(after_init_rgb.png/depth.npy)に対して全く
# 同じパラメータで再実行しても、パイプラインは決定的なので結果は変わらない
# (SAM3はテキストプロンプト固定・ランダム性なし)。そのため「やり直す」とは、
# depth外れ値除去の方式・許容幅という、②(depth prefilter)に関わるパラメータを
# 変えながら複数バリエーションを試し、正解に最も近い結果を採用することを指す
# (実機の再撮影のような新しい入力データを得る手段が無いオフライン評価での代替策)。
# 本番(get_book_points.pyのオンライン経路)へ同じ仕組みを入れるかどうかは、実機が
# 物理的に再撮影を伴う重い判断のため、いったんこの評価スクリプト側のみに留める。
#
# 【各試行で何がどう変わるか(2026-08-21、ユーザー確認済み・要明記)】
# 試行0(初回、通常の実行): depth_outlier_method="absolute"、tolerance=±3cm(30 raw、
#   depth_scale=0.001m/rawなので30 raw=30mm)。パイプラインの既定挙動そのまま。
# 試行1(retry 1回目): depth_outlier_method="ransac_residual"。②の絶対閾値をやめ、
#   OCR参照領域からRANSAC平面フィット→残差8mm(ransac_distance_threshold_m既定値)で
#   外れ値除去する方式に切り替える(優先度2、ed実装)。点数不足/フィット失敗時は
#   絶対閾値±3cm(tolerance変更なし)へ自動フォールバックする。
# 試行2(retry 2回目): depth_outlier_method="absolute"に戻し、tolerance=±4.5cm
#   (45 raw)に拡大。傾いた背表紙表面のdepth勾配が±3cmでは狭すぎる可能性を検証する。
# 試行3(retry 3回目・最終): depth_outlier_method="ransac_residual" かつ
#   tolerance=±4.5cm。RANSACが成功すればtoleranceの値自体は使われないが、
#   RANSACが失敗して絶対閾値にフォールバックした場合は±4.5cmが使われる
#   (試行1と試行3の違いは「RANSAC失敗時のフォールバック幅」のみ)。
# 各試行後、正解幅(book_width)との誤差が最も小さかったものを採用する
# (誤差<5mmになった時点で以降の試行は打ち切る)。
WIDTH_ERROR_RETRY_THRESHOLD_MM = 5.0
MAX_RETRIES = 3

# legacyパイプライン用: depth外れ値除去(②)の方式・許容幅を変えながら試す
# (詳細は上のコメント・HANDOFFを参照)。
RETRY_VARIANTS_LEGACY: list[dict] = [
    {"depth_outlier_method": "ransac_residual"},
    {"depth_outlier_method": "absolute", "depth_merge_tolerance_raw": 45},
    {"depth_outlier_method": "ransac_residual", "depth_merge_tolerance_raw": 45},
]

# simplifiedパイプライン用(2026-08-22追加): 簡易版にはdepth_outlier_methodという
# 選択肢自体が無く(depth中央値フィルタ→RANSAC平面フィット1回、の固定構成)、
# 唯一の調整可能パラメータはdepth_merge_tolerance_raw(depth中央値±Xmmの許容幅、
# 既定30=±3cm)のみ。そのため単純にこの許容幅を段階的に広げるだけの3段階とする。
RETRY_VARIANTS_SIMPLIFIED: list[dict] = [
    {"depth_merge_tolerance_raw": 45},
    {"depth_merge_tolerance_raw": 60},
    {"depth_merge_tolerance_raw": 90},
]


def run_one_with_retry(shot_name: str, src_dir: Path, query_book_name: str, work_root: Path,
                        runner_kwargs: dict, true_mm, folder_name: str | None = None,
                        enable_retry: bool = True, pipeline: str = "simplified") -> dict:
    """abs_error_mmがWIDTH_ERROR_RETRY_THRESHOLD_MM以上なら、pipelineに応じた
    RETRY_VARIANTS_*を順に試し、正解幅に最も近かった結果を採用する。true_mmが無い場合は
    リトライせず1回だけ実行する(正解が分からないと「改善したか」を判定できないため)。

    各試行は同じfolder_name配下を上書きするのではなく、末尾に_attempt{N}を付けた
    別フォルダに保存する(どの試行がどんな結果だったか後から確認できるようにするため)。
    """
    retry_variants = RETRY_VARIANTS_LEGACY if pipeline == "legacy" else RETRY_VARIANTS_SIMPLIFIED

    def _try(kwargs: dict, attempt_idx: int) -> tuple[dict, float | None]:
        suffix = "" if attempt_idx == 0 else f"_attempt{attempt_idx}"
        fname = f"{folder_name}{suffix}" if folder_name else f"{shot_name}{suffix}"
        result = run_one(shot_name, src_dir, query_book_name, work_root, kwargs, folder_name=fname, pipeline=pipeline)
        pred = float(result["book_width_mm_pred"])
        err = abs(pred - float(true_mm)) if true_mm is not None else None
        return result, err

    best_result, best_err = _try(runner_kwargs, 0)
    retry_count = 0

    if enable_retry and true_mm is not None and best_err is not None and best_err >= WIDTH_ERROR_RETRY_THRESHOLD_MM:
        for variant in retry_variants[:MAX_RETRIES]:
            retry_count += 1
            merged = {**runner_kwargs, **variant}
            result, err = _try(merged, retry_count)
            if err is not None and (best_err is None or err < best_err):
                best_result, best_err = result, err
            if err is not None and err < WIDTH_ERROR_RETRY_THRESHOLD_MM:
                break

    best_result["retry_count"] = retry_count
    return best_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["diagonal-40", "stand-100"], required=True)
    ap.add_argument("--only", default=None, help="このshot名だけ処理する(スモークテスト用)")
    ap.add_argument("--limit", type=int, default=None, help="先頭N件だけ処理する")
    ap.add_argument("--sam-device", default="gpu")
    ap.add_argument("--pipeline", default="simplified", choices=["simplified", "legacy"],
                    help="simplified(既定): 本番採用済みの簡易版パイプライン"
                         "(get_book_points_sam3_refined_sam2_width.py)。"
                         "legacy: 従来の複雑なパイプライン(get_book_points.py、A/B比較用)。")
    ap.add_argument("--sam-pts-side", default=None,
                    help="get_book_points.pyのsam_pts_sideを上書きする(例: 64,16)。"
                         "省略時はパイプライン既定値(32,8)のまま。")
    ap.add_argument("--depth-outlier-method", default=None, choices=["absolute", "ransac_residual"],
                    help="get_book_points.pyのdepth_outlier_methodを上書きする"
                         "(A/B比較用、2026-08-21追加)。省略時はパイプライン既定値(absolute)のまま。")
    ap.add_argument("--label", default=None,
                    help="work_root/out_csvのファイル名に付与する追加ラベル(例: rotfix)。"
                         "実行日時(必須、常に付与)の後ろに追加される"
                         "(例: --label rotfix -> _20260821_143000_rotfix)。")
    ap.add_argument("--work-suffix", default=None,
                    help="【非推奨、後方互換用】接尾辞を直接指定する。指定すると実行日時は"
                         "付与されない。通常は--labelを使うこと。")
    ap.add_argument("--no-retry", action="store_true",
                    help="推定幅誤差5mm以上でのリトライを無効化する(A/B比較で単一要因だけを"
                         "見たい場合用)。省略時はリトライ有効。")
    ap.add_argument("--no-fragmentation-handling", action="store_true",
                    help="②(depth prefilter)直後のマスク分裂対応ステージを無効化する"
                         "(A/B比較用、2026-08-21追加、優先度1)。省略時は有効。")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.work_suffix is None:
        # 2026-08-21、ユーザー要望: 実行日時は必ず含める(フォルダ/ファイルの区別のため)。
        # --labelで意味のある名前も併用できる(例: _20260821_143000_rotfix)。
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.work_suffix = f"_{timestamp}" + (f"_{args.label}" if args.label else "")

    master_by_bookname = load_master()

    legacy_only_flags_used = args.sam_pts_side or args.depth_outlier_method or args.no_fragmentation_handling
    if args.pipeline == "simplified" and legacy_only_flags_used:
        print("⚠ --sam-pts-side/--depth-outlier-method/--no-fragmentation-handlingは"
              "旧パイプライン(--pipeline legacy)専用のオプションです。"
              "簡易版パイプラインにはこれらに相当する後処理ステージが無いため無視します。")

    if args.pipeline == "legacy":
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
        if args.depth_outlier_method:
            runner_kwargs["depth_outlier_method"] = args.depth_outlier_method
        if args.no_fragmentation_handling:
            runner_kwargs["enable_fragmentation_handling"] = False
    else:
        # simplifiedパイプライン(get_book_points_sam3_refined_sam2_width.py)は
        # query/shot_dir以外にsam_device・depth_merge_tolerance_raw等ごく僅かな
        # キーワード引数しか受け付けない(encoder_path/interactive等は存在しない)。
        runner_kwargs = dict(sam_device=args.sam_device)

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
                  "book_width_mm_pred", "abs_error_mm", "retry_count", "elapsed_sec", "error"]
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
                "retry_count": 0,
                "elapsed_sec": "",
                "error": "",
            }
            folder_name = None
            if args.dataset == "stand-100":
                image_id = shot_name.split("__", 1)[0]
                folder_name = f"{i + 1}-{image_id}-{safe_name(row['display_name'] or shot_name)}"
            else:
                # diagonal-40は1画像=1アイテムで画像番号の概念が無いため、
                # {処理順}-{shot名}とする(2026-08-22、ユーザー要望:
                # stand-100と同様に処理順をフォルダ名の先頭に付けてほしい)。
                folder_name = f"{i + 1}-{safe_name(shot_name)}"
            try:
                result = run_one_with_retry(shot_name, src_dir, book_name, work_root, runner_kwargs,
                                             true_mm, folder_name=folder_name,
                                             enable_retry=not args.no_retry, pipeline=args.pipeline)
                row["book_width_mm_pred"] = round(float(result["book_width_mm_pred"]), 2)
                row["elapsed_sec"] = result["elapsed_sec"]
                row["retry_count"] = result["retry_count"]
                if true_mm is not None:
                    row["abs_error_mm"] = round(abs(row["book_width_mm_pred"] - float(true_mm)), 2)
                print(f"  -> pred={row['book_width_mm_pred']}mm true={true_mm}mm "
                      f"err={row['abs_error_mm']} retry={row['retry_count']} ({result['elapsed_sec']}s)")
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
