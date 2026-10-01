#!/usr/bin/env python3
"""
画像1枚に対して、マスタ全品目の対応付けを1回で行う短時間テストスクリプト。

recognition_accuracy_test.py のようにマスタの品目ごとにSAM3推論・OCRを
やり直すのではなく、SAM3推論とOCRを画像につき1回だけ実行し、
multikey_matcher.match_text_to_mask_main() が内部で行っている
「マスタ全品目 x 全マスクの一括割当」を1回呼ぶだけで、
どの品目がどのマスクに対応付けられたか／対応付けられなかったかを一覧できるようにする。

使い方:
    python -m catheter.scripts.single_image_match_test --shot-dir reco/OPTIMA-2/depth_shots/1
    python -m catheter.scripts.single_image_match_test --shot-dir <dir> --master-json <path>

    # 2026-09-29追加: --datasetでreco/<名前>/depth_shots/の全画像をまとめて処理する
    # (width_mm_validation.pyのdepth_shots_dir.iterdir()と同じ、枚数のハードコード無し)
    python -m catheter.scripts.single_image_match_test --dataset 0911 --master-json <path>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
RECO_ROOT = REPO_ROOT / "reco"
sys.path.insert(0, str(REPO_ROOT))

from catheter.scripts.multikey_matcher import match_text_to_mask_main, _load_master, DEFAULT_MASTER_JSON
from detection.pro_handbook.sam_py_demo.get_book_points import (
    _get_sam_runner_compat,
    _make_stage_save_cfg_compat,
    _infer_masks_compat,
    start_ocr_subprocess,
    wait_ocr_subprocess,
)


def _default_out_root(src_dir: Path) -> Path:
    """width_mm_validation.pyのreco_result_<日時>/ と同じ流儀。src_dirが
    reco/<データセット>/depth_shots/<n> の形なら、reco_result_<日時>/ は
    depth_shots/ ではなく <データセット>/ 直下に作る。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root = src_dir.parent.parent if src_dir.parent.name == "depth_shots" else src_dir.parent
    return dataset_root / f"single_image_match_reco_result_{ts}"


def _make_result_dir(src_dir: Path, out_root: Path) -> Path:
    """src_dir(元データ)には一切書き込まず、out_root/work/<src_dir名>/ へ入力ファイルだけ
    コピーする(2026-09-29追加、ユーザー要望)。"""
    work_dir = Path(out_root) / "work" / src_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("after_init_rgb.png", "after_init_depth.npy", "camera_params.json"):
        src = src_dir / fn
        if src.exists():
            shutil.copy2(src, work_dir / fn)
    return work_dir


def _build_xlsx_report(out_root: Path) -> None:
    """結果フォルダ(out_root)から、目視確認用Excel(single_image_match_report.xlsx)と
    品目ごとの画像をまとめたimages/を自動生成する(2026-09-29追加、ユーザー要望:
    「スクリプトを実行したら自動生成されるようにしてほしい」)。
    reco/scripts/build_single_image_match_report.py相当。失敗しても認識結果自体は
    既に保存済みなので、処理は継続する(width_mm_validation.pyの自動レポート生成と同じ方針)。
    """
    try:
        sys.path.insert(0, str(RECO_ROOT / "scripts"))
        from build_single_image_match_report import build_report as _build_report_fn
        _build_report_fn(out_root)
    except Exception as e:
        print(f"⚠ Excelレポートの自動生成に失敗しました(認識結果自体は保存済みです): "
              f"{type(e).__name__}: {e}")


def _export_mask_shapes(shot_dir: Path) -> None:
    """撮影1枚ぶんのマスクを、形だけ切り出した画像(mask_images/mask_<番号>.png)として
    保存する(2026-10-01追加、ユーザー要望:「認識を回すときに自動で実行されるように
    してほしい」)。reco/scripts/export_mask_shape_images.py相当。失敗しても認識結果
    自体は既に保存済みなので、処理は継続する(_build_xlsx_reportと同じ方針)。
    """
    try:
        sys.path.insert(0, str(RECO_ROOT / "scripts"))
        from export_mask_shape_images import export_shot as _export_shot_fn
        _export_shot_fn(shot_dir)
    except Exception as e:
        print(f"⚠ マスク形状画像の自動生成に失敗しました(認識結果自体は保存済みです): "
              f"{type(e).__name__}: {e}")


def _write_run_info(out_root: Path, *, target_desc: str, master_json: str | None,
                     sam_device: str) -> None:
    """実行条件(環境変数・SAM3モデル・マスタ・gitのHEAD等)を、結果フォルダに
    run_info.mdとして記録する(2026-09-29追加、ユーザー要望: 「実行した際の条件が
    記録されたmdファイルが作られるようにしてほしい」)。width_mm_validation.pyを
    まとめて回す際に手作業で作っていたrun_info_<日時>.mdと同じ内容。
    失敗しても認識処理は継続する。
    """
    lines = ["# single_image_match_test.py 実行条件の記録", "",
             f"- 実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"- 対象: {target_desc}",
             f"- コマンド: {' '.join(sys.argv)}"]

    master_path = Path(master_json) if master_json else Path(DEFAULT_MASTER_JSON)
    try:
        n_items = len(_load_master(master_path))
        lines.append(f"- マスタ: {master_path} ({n_items}件)")
    except Exception as e:
        lines.append(f"- マスタ: {master_path} (読み込み失敗: {type(e).__name__}: {e})")
    lines.append(f"- sam_device: {sam_device}")

    endpoint = os.environ.get("SAM3_ENDPOINT", "http://127.0.0.1:8765")
    try:
        with urllib.request.urlopen(f"{endpoint}/model-info", timeout=5) as resp:
            info = resp.read().decode("utf-8")
        lines.append(f"- SAM3サービス({endpoint}): {info}")
    except Exception as e:
        lines.append(f"- SAM3サービス({endpoint}): 取得失敗({type(e).__name__}: {e})")

    multikey_env = {k: v for k, v in os.environ.items() if k.startswith("MULTIKEY_")}
    if multikey_env:
        for k, v in sorted(multikey_env.items()):
            lines.append(f"- 環境変数 {k}={v}")
    else:
        lines.append("- 環境変数: MULTIKEY_*系は未設定(該当なし足切りはOFF)")

    try:
        head = subprocess.run(["git", "log", "-1", "--format=%h %ad", "--date=short"],
                               cwd=REPO_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        n_dirty = len(subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT,
                                      capture_output=True, text=True, timeout=5).stdout.splitlines())
        lines.append(f"- gitのHEAD: {head} / 未コミットの変更ファイル数: {n_dirty}")
    except Exception as e:
        lines.append(f"- git情報: 取得失敗({type(e).__name__}: {e})")

    try:
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "run_info.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"⚠ 実行条件の記録(run_info.md)に失敗しました(認識結果自体は保存済みです): "
              f"{type(e).__name__}: {e}")


def run_single_image_match(shot_dir: Path, master_json: str | None, sam_device: str = "gpu",
                            out_dir: Path | None = None, build_report: bool = True,
                            write_run_info: bool = True) -> dict:
    src_dir = Path(shot_dir).expanduser().resolve()
    if not (src_dir / "after_init_rgb.png").exists():
        raise FileNotFoundError(f"{src_dir / 'after_init_rgb.png'} がありません")

    # 元データ(src_dir)には一切書き込まない。入力ファイルだけコピーしたwork_dirを作り、
    # SAM3・OCR・照合の結果はすべてそちらに保存する。
    out_root = Path(out_dir).expanduser().resolve() if out_dir else _default_out_root(src_dir)
    shot_dir = _make_result_dir(src_dir, out_root)
    print(f"[single_image_match_test] 元データ: {src_dir}")
    print(f"[single_image_match_test] 結果の保存先: {shot_dir}")
    if write_run_info:
        _write_run_info(out_root, target_desc=f"1枚 (--shot-dir {src_dir})",
                         master_json=master_json, sam_device=sam_device)

    rgb_path = shot_dir / "after_init_rgb.png"
    depth_path = shot_dir / "after_init_depth.npy"
    if not rgb_path.exists():
        raise FileNotFoundError(f"{rgb_path} がありません")

    color_np = cv2.imread(str(rgb_path))
    if color_np is None:
        raise FileNotFoundError(f"{rgb_path} を読み込めませんでした")
    depth_np_u16 = np.load(depth_path) if depth_path.exists() else None

    # ===== OCRを非同期開始 =====
    ocr_proc = start_ocr_subprocess(shot_dir)

    # ===== SAM3推論(画像につき1回だけ) =====
    sam_runner = _get_sam_runner_compat(
        encoder_path="", decoder_path="", sam_device=sam_device,
    )
    rgb_pil = Image.fromarray(cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB))
    stage_cfg = _make_stage_save_cfg_compat(shot_dir)
    masks, _sam_data = _infer_masks_compat(
        sam_runner, rgb_pil, stage_cfg, depth_np_u16=depth_np_u16,
    )
    print(f"[single_image_match_test] SAM3マスク数: {len(masks)}")
    _export_mask_shapes(shot_dir)

    # ===== OCR終了待ち =====
    ocr_stdout = wait_ocr_subprocess(ocr_proc, timeout=120.0)
    if ocr_stdout.strip():
        print(ocr_stdout, end="" if ocr_stdout.endswith("\n") else "\n")

    # ===== マスタ全品目 x 全マスクの一括照合を1回だけ実行 =====
    master_path = Path(master_json) if master_json else Path(DEFAULT_MASTER_JSON)
    master = _load_master(master_path)
    if not master:
        raise RuntimeError("マスタが空です")
    dummy_query = master[0]["book_name"]  # target_j選択専用。all_assignmentsの内容には影響しない

    match_text_to_mask_main(
        dummy_query, masks, shot_dir,
        master_json=master_json,
        save_all_assignments=True,
        save_all_perkey_overlays=True,
    )

    debug = json.loads((shot_dir / "multikey_match_debug.json").read_text(encoding="utf-8"))
    all_assignments = debug["all_assignments"]

    matched = [a for a in all_assignments if a["assigned_mask"] is not None]
    unmatched = [a for a in all_assignments if a["assigned_mask"] is None]

    print(f"\n=== 対応付け結果 ({shot_dir}) ===")
    print(f"マスタ品目数: {len(all_assignments)} / マスク数: {len(masks)}")
    print(f"\n--- 対応付けされた品目 ({len(matched)}) ---")
    for a in matched:
        name = a["display_name"] or a["book_name"]
        print(f"  {name:<30} -> {a['assigned_mask']:<10} score={a['score']:>6.1f}  key={a['winning_key']}")

    print(f"\n--- 対応付けされなかった品目 ({len(unmatched)}) ---")
    for a in unmatched:
        name = a["display_name"] or a["book_name"]
        print(f"  {name}")

    print(f"\n一覧画像: {shot_dir / 'ocr_overlay_all_assignments.png'}")
    print(f"品目ごとの画像: {shot_dir / 'images'}/<品目名>.png"
          "（対応付けされた品目=選択箇所の拡大、対応付けされなかった品目=真っ暗な画像）")
    print(f"詳細JSON: {shot_dir / 'multikey_match_debug.json'}")

    if build_report:
        _build_xlsx_report(out_root)

    return {"matched": matched, "unmatched": unmatched, "n_masks": len(masks)}


def run_dataset_match(dataset: str, master_json: str | None, sam_device: str = "gpu",
                       out_dir: Path | None = None, build_report: bool = True) -> dict:
    """reco/<dataset>/depth_shots/ の下にある画像フォルダを全部処理する
    (2026-09-29追加、ユーザー要望)。

    width_mm_validation.pyのmain()と同じく、depth_shots_dir.iterdir()で存在するフォルダを
    そのまま数えるだけなので、枚数のハードコードは無く何枚でも対応する。
    結果は全画像ぶんまとめて1つのreco_result_<日時>/work/<画像フォルダ名>/以下に保存する
    (画像ごとに新しいreco_resultを作らない)。
    """
    depth_shots_dir = RECO_ROOT / dataset / "depth_shots"
    if not depth_shots_dir.is_dir():
        raise FileNotFoundError(f"{depth_shots_dir} がありません")

    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = RECO_ROOT / dataset / f"single_image_match_reco_result_{ts}"
    out_dir = Path(out_dir).expanduser().resolve()

    shot_dirs = sorted((p for p in depth_shots_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not shot_dirs:
        raise FileNotFoundError(f"{depth_shots_dir} に画像フォルダがありません")
    print(f"[single_image_match_test] {dataset}: {len(shot_dirs)}枚を処理します"
          f"({', '.join(p.name for p in shot_dirs)})")
    print(f"[single_image_match_test] 結果の保存先(まとめて): {out_dir}")
    # run_info.mdも全画像で共有(画像ごとに書き直すと最後の1枚の情報で上書きされて
    # しまうため、対象説明に全画像分をまとめて1回だけ書く)。
    _write_run_info(out_dir, target_desc=f"{dataset}: depth_shots/ 全{len(shot_dirs)}枚 "
                                          f"({', '.join(p.name for p in shot_dirs)})",
                     master_json=master_json, sam_device=sam_device)

    per_shot = {}
    for shot_dir in shot_dirs:
        print(f"\n########## {shot_dir.name} ##########")
        # レポートは全画像処理後に1回だけ作る(画像ごとに作ると、後の画像の分だけ
        # 何度も上書き生成することになり無駄なため)。
        per_shot[shot_dir.name] = run_single_image_match(
            shot_dir, master_json, sam_device, out_dir=out_dir,
            build_report=False, write_run_info=False,
        )

    total_matched = sum(len(r["matched"]) for r in per_shot.values())
    total_unmatched = sum(len(r["unmatched"]) for r in per_shot.values())
    print(f"\n=== {dataset} 全{len(shot_dirs)}枚のまとめ ===")
    for name, r in per_shot.items():
        print(f"  {name}: 対応付け {len(r['matched'])}件 / 該当なし {len(r['unmatched'])}件"
              f" (マスク数{r['n_masks']})")
    print(f"合計: 対応付け {total_matched}件 / 該当なし {total_unmatched}件")

    if build_report:
        _build_xlsx_report(out_dir)

    return {"per_shot": per_shot, "out_dir": out_dir}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--shot-dir", help="after_init_rgb.pngのある元データのディレクトリ1つだけを処理する(読み込み専用)")
    target.add_argument("--dataset", help="reco/<名前>/depth_shots/ の下の画像を全部処理する(2026-09-29追加)")
    ap.add_argument("--master-json", default=None, help="省略時はmultikey_matcher.pyの既定マスタ")
    ap.add_argument("--sam-device", default="gpu")
    ap.add_argument("--out-dir", default=None,
                    help="結果の保存先。省略時は<データセット直下>/"
                         "single_image_match_reco_result_<日時>/work/<画像フォルダ名>/ を自動作成")
    ap.add_argument("--no-report", action="store_true",
                    help="完了後の目視確認用Excel・画像フォルダの自動生成(build_single_image_match_report.py"
                         "相当)を無効化する。省略時は自動生成される(2026-09-29追加)")
    args = ap.parse_args()
    build_report = not args.no_report
    if args.dataset:
        run_dataset_match(args.dataset, args.master_json, args.sam_device,
                           out_dir=args.out_dir, build_report=build_report)
    else:
        run_single_image_match(Path(args.shot_dir), args.master_json, args.sam_device,
                                out_dir=args.out_dir, build_report=build_report)


if __name__ == "__main__":
    main()
