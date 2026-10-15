#!/usr/bin/env python3
"""
depth_shots形式(after_init_rgb.png + after_init_depth.npy + camera_params.json)
ではなく、ただのRGB画像フォルダ(*.png)に対して認識(SAM3+OCR+multikey_matcher)を
バッチ実行するスクリプト(2026-10-06追加、解像度ごとのOCR精度比較実験用)。

single_image_match_test.pyのrun_single_image_match()をそのまま再利用し、
入力ファイル名の要件(after_init_rgb.pngという決め打ちの名前。OCRサブプロセスが
この名前を前提にしている)だけ、一時フォルダへのコピーで満たす。

SAM3サービスは、このスクリプトが自動で起動し、終了時に停止する(2026-10-15追加)。
事前に手で立ち上げておく必要はない。起動中のサービスをそのまま使いたい場合は
--keep-service を付ける。

使い方:
    python -m catheter.scripts.no_depth_recognition \
        --images-dir captures/resolution_test/4/1920x1080 \
        --master-json Master_JSON/master_catheter_0911.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from catheter.scripts.single_image_match_test import (  # noqa: E402
    run_single_image_match,
    _write_run_info,
    _build_xlsx_report,
)

# ===== SAM3サービスの起動・停止 =====
# 2026-10-15追加。これまでは事前に手でサービスを立ち上げていたため、前の試験の
# 状態(キャッシュ・GPUメモリの断片化等)を引きずったまま計測していた。1回の試験ごとに
# 起動し直すことで、解像度ごとの計算時間・VRAMを同じ条件で測れるようにする。
# 設計は HANDOFF_20261013_sam3_service_lifecycle_todo.md の方針に沿う
# (既存の start_service.sh / stop_service.sh をそのまま呼び、本番の
#  Retrieval_integration.py 等、常時稼働が前提のスクリプトには手を入れない)。
SERVICE_SCRIPTS = REPO_ROOT / "detection/pro_handbook/sam3_runtime/scripts"
SERVICE_URL = "http://{}:{}".format(
    os.getenv("SAM3_HOST", "127.0.0.1"), os.getenv("SAM3_PORT", "8765"))


def _service_ready(timeout: float = 2.0) -> bool:
    """/health が ready を返すか。起動していなければ False。"""
    try:
        with urllib.request.urlopen(f"{SERVICE_URL}/health", timeout=timeout) as res:
            return bool(json.loads(res.read().decode()).get("ready"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _stop_service() -> None:
    subprocess.run([str(SERVICE_SCRIPTS / "stop_service.sh")],
                    check=False, capture_output=True, text=True)


def _start_service(wait_sec: float = 300.0) -> None:
    """サービスを起動し、モデルの読み込みが終わる(=/healthがready)まで待つ。"""
    proc = subprocess.run([str(SERVICE_SCRIPTS / "start_service.sh")],
                           check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"start_service.sh が失敗しました:\n{proc.stderr.strip()}")
    print(f"[service] {proc.stdout.strip()}  モデル読み込みを待っています...")
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if _service_ready():
            print(f"[service] 起動しました ({SERVICE_URL})")
            return
        time.sleep(1.0)
    _stop_service()
    raise RuntimeError(
        f"SAM3サービスが{wait_sec:.0f}秒以内に ready になりませんでした。"
        f" detection/pro_handbook/sam3_runtime/logs/service.stdout.log を確認してください")


def _sort_key(p: Path):
    return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)


def run_images_dir(images_dir: Path, master_json: str | None, sam_device: str = "gpu",
                    out_dir: Path | None = None) -> dict:
    images_dir = Path(images_dir).expanduser().resolve()
    image_files = sorted(images_dir.glob("*.png"), key=_sort_key)
    if not image_files:
        raise FileNotFoundError(f"{images_dir} に*.pngがありません")

    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = images_dir.parent / f"no_depth_reco_{images_dir.name}_{ts}"
    out_dir = Path(out_dir).expanduser().resolve()

    print(f"[no_depth_recognition] {images_dir}: {len(image_files)}枚を処理します")
    print(f"[no_depth_recognition] 結果の保存先(まとめて): {out_dir}")
    _write_run_info(
        out_dir,
        target_desc=f"{images_dir}: *.png 全{len(image_files)}枚 "
                    f"({', '.join(p.name for p in image_files)})",
        master_json=master_json,
        sam_device=sam_device,
    )

    per_shot = {}
    with tempfile.TemporaryDirectory(prefix="no_depth_recognition_") as tmp_root:
        tmp_root = Path(tmp_root)
        for img_path in image_files:
            shot_name = img_path.stem
            # run_single_image_match()はshot_dir内の"after_init_rgb.png"という
            # 決まったファイル名を前提にしている(OCRサブプロセスの入力名と合わせる
            # 必要があるため)ので、一時フォルダへその名前でコピーしてから渡す。
            src_dir = tmp_root / shot_name
            src_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, src_dir / "after_init_rgb.png")

            print(f"\n########## {shot_name} ({img_path.name}) ##########")
            per_shot[shot_name] = run_single_image_match(
                src_dir, master_json, sam_device, out_dir=out_dir,
                build_report=False, write_run_info=False,
            )

    total_matched = sum(len(r["matched"]) for r in per_shot.values())
    total_unmatched = sum(len(r["unmatched"]) for r in per_shot.values())
    print(f"\n=== {images_dir} 全{len(image_files)}枚のまとめ ===")
    for name, r in per_shot.items():
        print(f"  {name}: 対応付け {len(r['matched'])}件 / 該当なし {len(r['unmatched'])}件"
              f" (マスク数{r['n_masks']})")
    print(f"合計: 対応付け {total_matched}件 / 該当なし {total_unmatched}件")

    _build_xlsx_report(out_dir)
    return {"per_shot": per_shot, "out_dir": out_dir}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True,
                     help="*.pngが並んだフォルダ(depth_shots形式でなくてよい)")
    ap.add_argument("--master-json", default=None, help="省略時はmultikey_matcher.pyの既定マスタ")
    ap.add_argument("--sam-device", default="gpu")
    ap.add_argument("--out-dir", default=None,
                     help="省略時は--images-dirの親に ocr_test_reco_result_<フォルダ名>_<日時>/ を作る")
    ap.add_argument("--keep-service", action="store_true",
                     help="起動中のSAM3サービスをそのまま使い、終了時も停止しない。"
                          "既定は試験ごとに起動し直し、終了時に停止する(計測条件をそろえるため)")
    args = ap.parse_args()

    started_by_us = False
    if args.keep_service:
        if not _service_ready():
            raise SystemExit("--keep-service が指定されましたが、SAM3サービスが起動していません")
        print("[service] 起動中のサービスをそのまま使います (--keep-service)")
    else:
        if _service_ready():
            print("[service] 既に起動中のサービスを停止します(計測条件をそろえるため)")
            _stop_service()
            for _ in range(15):
                if not _service_ready(timeout=1.0):
                    break
                time.sleep(1.0)
        if _service_ready():
            # stop_service.sh は logs/service.pid のPIDしか止めないので、別の方法で
            # 起動されたサービスは残る。その場合は新たに起動せず(ポートが衝突する)、
            # 既存のものをそのまま使う。
            print("[service] 停止できなかったため、起動中のサービスをそのまま使います")
        else:
            _start_service()
            started_by_us = True

    try:
        run_images_dir(
            Path(args.images_dir), args.master_json, args.sam_device,
            Path(args.out_dir) if args.out_dir else None,
        )
    finally:
        # 異常終了(Ctrl-C・例外)でもサービスを残さない
        if started_by_us:
            print("[service] SAM3サービスを停止します")
            _stop_service()


if __name__ == "__main__":
    main()
