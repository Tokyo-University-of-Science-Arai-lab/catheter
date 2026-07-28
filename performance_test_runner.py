#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能試験（認識＋実機での取り出し）用の実行スクリプト。

Retrieval_integration.py の main_sequence()（実際にxArm・ハンドを動かす、
動作実績のある関数）をそのまま呼び出し、以下を追加する。

  1. 認識開始前に、どの書籍を取り出すか選択・スキップ・試験終了を選べる
     （main_sequence() 内の Enter/Ctrl+D/Ctrl+C による
      挿入する/次の書籍へ/中断する、という選択はそのまま生きている）
  2. main_sequence() は (book_width, result, shot_dir) を直接返すので、
     ログファイルを読み返さずにその場で3分類へ振り分ける
       - equipment_fault       : 給電不足・モーターエラー等 → やり直し要
       - recognition_incomplete: 認識自体が完了しなかった   → やり直し要
       - retrieval_outcome     : 取り出し結果（成功/断念/引っかかり等） → やり直し不要
  3. 保存先を captures/<run_id>/<n>/ と performance_tests/<run_id>/<n>/ の
     両方にする
  4. run_info.json / progress.json による記録と --resume-dir での再開

重要な前提（xarm_monitor.py の実装より）:
  XArmMonitor は異常を検知すると os._exit(1) でプロセスそのものを
  即時強制終了する。これは例外処理では捕まえられない。「ケースの途中で
  プロセスが死ぬ」ことは正常に起こりうる前提として扱い、再起動後の
  --resume-dir 再開時にだけ、直前のケースがどう終わったかをオペレーターに
  1問確認する（このケースに限りログファイルを参考情報として読む）。

使い方:
  新規実行:
    python performance_test_runner.py --repeat-per-book 5 --order block

  中断後の再開:
    python performance_test_runner.py --resume-dir performance_tests/20260714_153000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from Retrieval_integration import (
    load_config,
    setup_hardware,
    teardown_hardware,
    main_sequence,
)

try:
    from detection.pro_handbook.sam_py_demo.recognition_100test import (
        find_project_root,
        resolve_path,
        load_master_books,
        build_test_plan,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "detection/pro_handbook/sam_py_demo"))
    from recognition_100test import (  # type: ignore
        find_project_root,
        resolve_path,
        load_master_books,
        build_test_plan,
    )


REDO_CATEGORIES = {"equipment_fault", "recognition_incomplete"}

# main_sequence() が返す result 文字列 → (category, detail)
RESULT_MAP: dict[str, tuple[str, str]] = {
    "success": ("retrieval_outcome", "success"),
    "ctrl+d": ("retrieval_outcome", "aborted_before_insert"),
    "barcode_mismatch": ("retrieval_outcome", "barcode_mismatch"),
    "recognition_fail": ("recognition_incomplete", "recognition_fail"),
    "exception": ("equipment_fault", "exception"),
}

QUIT = object()
SKIP = object()


# ── JSON入出力（クラッシュに強い書き込み） ──────────────────────────

def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── retrieval_log.csv の最終行を読む（プロセス死亡後の再開時のみ使用） ──

LOG_FIELDS = ["timestamp", "book_name", "shelf_id", "roll_deg", "book_width",
              "side", "height", "result", "shot_dir", "memo"]


def read_last_log_row(log_path: Path) -> dict[str, str] | None:
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    with log_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    last = rows[-1]
    if len(last) < len(LOG_FIELDS):
        last = last + [""] * (len(LOG_FIELDS) - len(last))
    return dict(zip(LOG_FIELDS, last))


# ── run_info.json / progress.json ───────────────────────────────────

def build_run_info(*, run_id: str, args: argparse.Namespace, plan_len: int) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",  # running / interrupted / completed
        "purpose": args.purpose or "",
        "params": {
            "master_json": args.master_json,
            "repeat_per_book": args.repeat_per_book,
            "order": args.order,
        },
        "plan_len": plan_len,
    }


def case_needs_redo(case_entry: dict[str, Any] | None) -> bool:
    if case_entry is None:
        return True
    if case_entry.get("state") != "done":
        return True  # in_progressのまま = 途中でプロセスが落ちた
    if case_entry.get("category") in REDO_CATEGORIES:
        return True
    return False


def mark_in_progress(progress_path: Path, case_dict: dict, test_index: int, book_name: str) -> None:
    case_dict[str(test_index)] = {
        "state": "in_progress",
        "book_name": book_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_atomic(progress_path, {"cases": case_dict})


def mark_done(
    progress_path: Path,
    case_dict: dict,
    test_index: int,
    *,
    category: str,
    detail: str | None,
    book_width_mm: float | None,
    shot_dir_captures: str | None,
    shot_dir_perf: str | None,
) -> None:
    case_dict[str(test_index)] = {
        "state": "done",
        "category": category,
        "detail": detail,
        "book_width_mm": book_width_mm,
        "shot_dir_captures": shot_dir_captures,
        "shot_dir_perf": shot_dir_perf,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_atomic(progress_path, {"cases": case_dict})


# ── ディレクトリの整理・複製 ─────────────────────────────────────────

def organize_case_dir(shot_dir: str | Path | None, captures_run_dir: Path, test_index: int) -> Path | None:
    if not shot_dir:
        return None
    src = Path(shot_dir)
    if not src.exists():
        return None
    dst = captures_run_dir / str(test_index)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def mirror_case_dir(src_captures_dir: Path, perf_run_dir: Path, test_index: int) -> Path:
    dst = perf_run_dir / str(test_index)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_captures_dir, dst)
    return dst


# ── 認識開始前：書籍を選ぶ／スキップ／終了 ────────────────────────────

def display_name_of(books: list[dict[str, Any]], book_name: str) -> str:
    """book_name に対応する表示名を返す。見つからなければ book_name のまま。"""
    for b in books:
        if b["book_name"] == book_name:
            return b.get("display_name", book_name)
    return book_name


def prompt_book_selection(
    books: list[dict[str, Any]],
    rec: dict[str, Any],
    case_idx: int,
    total: int,
    repeat_per_book: int,
) -> dict[str, Any] | object:
    """認識を始める前に、対象書籍を選ばせる。
    戻り値: 選ばれた書籍dict、または SKIP / QUIT。"""
    print(f"\n========== CASE {case_idx}/{total} ==========")
    print(f"予定: {display_name_of(books, rec['book_name'])}（繰り返し {rec['repeat_index']}/{repeat_per_book}）")
    for idx, b in enumerate(books):
        mark = " ←予定" if b["book_name"] == rec["book_name"] else ""
        print(f"  {idx + 1:2d}. {b.get('display_name', b['book_name'])}{mark}")
    print("[Enter] 予定の書籍で実行  /  [番号] 別の書籍を選ぶ  /  [s] このケースをスキップ  /  [q] 試験を終了")

    while True:
        raw = input("> ").strip()
        if raw == "":
            return rec
        if raw.lower() == "q":
            return QUIT
        if raw.lower() == "s":
            return SKIP
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(books):
                b = books[n - 1]
                return {
                    "book_name": b["book_name"],
                    "ISBN_number": b["ISBN_number"],
                    "bookshelf_ID": b["bookshelf_ID"],
                }
        print(f"  Enter / 1〜{len(books)} / s / q のいずれかを入力してください。")


def classify_after_process_death(log_path: Path, rec: dict[str, Any], books: list[dict[str, Any]]) -> tuple[str, str]:
    """前回プロセスが緊急停止等で死んだ場合、再開時にオペレーターへ確認する。
    ログはあくまで参考情報として表示するだけで、判断自体はオペレーターが行う。"""
    row = read_last_log_row(log_path)
    hint = ""
    if row is not None and row.get("book_name") == rec["book_name"]:
        hint = f"（参考: ログの最終記録 result={row.get('result')!r}, 時刻={row.get('timestamp')}）"

    print(f"\n前回、ケース{rec['test_index']}（{display_name_of(books, rec['book_name'])}）の途中でプログラムが停止しました。{hint}")
    print("この停止の原因を教えてください:")
    print("  1) 設備トラブル（給電不足・モーターエラー等）→ このケースはやり直します")
    print("  2) 差し込み時に引っかかったことによる緊急停止（幅不足）→ 結果として記録し、次へ進みます")
    while True:
        choice = input("番号を入力 [1-2]: ").strip()
        if choice == "1":
            return "equipment_fault", "process_died_equipment"
        if choice == "2":
            return "retrieval_outcome", "caught_on_insert"
        print("  1か2を入力してください。")


# ── メイン ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--master-json", type=str, default="master_20260216.json")
    parser.add_argument("--captures-dir", type=str, default="captures")
    parser.add_argument("--perf-dir", type=str, default="performance_tests")
    parser.add_argument("--repeat-per-book", type=int, default=5)
    parser.add_argument("--order", choices=["block", "cycle"], default="block")
    parser.add_argument("--purpose", type=str, default="")
    parser.add_argument("--resume-dir", type=str, default=None)

    args = parser.parse_args()

    project_root = find_project_root()
    captures_base = resolve_path(args.captures_dir, base_dir=project_root)
    perf_base = resolve_path(args.perf_dir, base_dir=project_root)

    config = load_config(str(project_root / "Retrieval_integration.yaml"))
    retrieval_log_path = Path(config["paths"]["log"]["retrieval"])

    pending_death_classification = False

    if args.resume_dir:
        perf_run_dir = resolve_path(args.resume_dir, base_dir=project_root)
        run_info_path = perf_run_dir / "run_info.json"
        progress_path = perf_run_dir / "progress.json"
        if not run_info_path.exists() or not progress_path.exists():
            raise FileNotFoundError(f"再開に必要なファイルが見つかりません: {perf_run_dir}")

        run_info = read_json(run_info_path)
        progress = read_json(progress_path)
        run_id = run_info["run_id"]
        p = run_info["params"]

        books = load_master_books(p["master_json"], project_root=project_root)
        plan = build_test_plan(books, repeat_per_book=p["repeat_per_book"], order=p["order"])
        args.master_json = p["master_json"]
        args.repeat_per_book = p["repeat_per_book"]
        args.order = p["order"]

        captures_run_dir = captures_base / run_id
        captures_run_dir.mkdir(parents=True, exist_ok=True)

        run_info["status"] = "running"
        write_json_atomic(run_info_path, run_info)

        case_dict: dict[str, Any] = progress["cases"]
        pending_death_classification = any(
            c.get("state") == "in_progress" for c in case_dict.values()
        )

        print(f"===== 再開: {run_id} =====")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        captures_run_dir = captures_base / run_id
        perf_run_dir = perf_base / run_id
        captures_run_dir.mkdir(parents=True, exist_ok=True)
        perf_run_dir.mkdir(parents=True, exist_ok=True)

        books = load_master_books(args.master_json, project_root=project_root)
        plan = build_test_plan(books, repeat_per_book=args.repeat_per_book, order=args.order)

        run_info = build_run_info(run_id=run_id, args=args, plan_len=len(plan))
        run_info_path = perf_run_dir / "run_info.json"
        write_json_atomic(run_info_path, run_info)
        write_json_atomic(captures_run_dir / "run_info.json", run_info)

        progress_path = perf_run_dir / "progress.json"
        case_dict = {}
        write_json_atomic(progress_path, {"cases": case_dict})

        print(f"===== 新規実行: {run_id} =====")

    print(f"project root  : {project_root}")
    print(f"captures保存先 : {captures_run_dir}")
    print(f"perf保存先     : {perf_run_dir}")
    print(f"総ケース数     : {len(plan)}")
    print("緊急停止でプロセスが強制終了した場合は、同じコマンドに")
    print(f"  --resume-dir {perf_run_dir}")
    print("を付けて再実行してください。")
    print("=========================================")

    hw = setup_hardware(config)
    node = hw["node"]
    arm = hw["arm"]
    executor = hw["executor"]
    monitor = hw["monitor"]
    tp = hw["tp"]
    waypoint_node = hw["waypoint_node"]

    retrieved_book_width_list = [0.0]
    interrupted = False

    try:
        for rec in plan:
            i = int(rec["test_index"])
            existing = case_dict.get(str(i))

            if not case_needs_redo(existing):
                continue

            # 前回プロセスが死んだ直後の、そのケースについてだけ1問確認する。
            # 注意: 緊急停止時のログはshot_dirを記録しないため、このケースの
            # 撮影データは captures/ 直下のタイムスタンプフォルダに残ったままとなり、
            # 自動での整理・複製はできない（要手動確認）。
            if pending_death_classification and existing is not None and existing.get("state") == "in_progress":
                category, detail = classify_after_process_death(retrieval_log_path, rec, books)
                pending_death_classification = False
                mark_done(
                    progress_path, case_dict, i,
                    category=category, detail=detail,
                    book_width_mm=None,
                    shot_dir_captures=None, shot_dir_perf=None,
                )
                if category in REDO_CATEGORIES:
                    print("→ やり直します。\n")
                else:
                    print("→ 結果として記録しました。次のケースへ進みます。\n")
                    continue

            if existing is not None and existing.get("state") == "done":
                print(f"\n(ケース{i}の前回の記録: category={existing.get('category')}, detail={existing.get('detail')} のためやり直します)")

            # ===== 認識開始前：書籍を選ぶ／スキップ／試験終了 =====
            choice = prompt_book_selection(books, rec, i, len(plan), args.repeat_per_book)
            if choice is QUIT:
                print("試験を終了します。")
                break
            if choice is SKIP:
                mark_done(
                    progress_path, case_dict, i,
                    category="retrieval_outcome", detail="skipped_before_recognition",
                    book_width_mm=None, shot_dir_captures=None, shot_dir_perf=None,
                )
                print("→ スキップしました。次のケースへ進みます。")
                continue

            chosen_book = choice  # dict: book_name / ISBN_number / bookshelf_ID
            book_width_offset = sum(retrieved_book_width_list)

            # 開始時点で in_progress として記録（緊急停止対策）
            mark_in_progress(progress_path, case_dict, i, chosen_book["book_name"])

            # main_sequence() の中で、実際の認識・xArm移動・ハンド開閉・
            # 差し込み動作が行われる。差し込み直前に Enter(挿入)/Ctrl+D(次へ)/
            # Ctrl+C(中断) をオペレーターが選ぶプロンプトが出る（変更なし）。
            ret, result, shot_dir = main_sequence(
                config=config,
                book_name=chosen_book["book_name"],
                barcode_number=chosen_book["ISBN_number"],
                bookshelf_ID=chosen_book["bookshelf_ID"],
                book_width_offset=book_width_offset,
                tp=tp,
                node=node,
                arm=arm,
                executor=executor,
                waypoint_node=waypoint_node,
                shelf_manager=waypoint_node.shelf_manager,
                monitor=monitor,
            )
            # ここに到達した = プロセスは生きている = main_sequence()が正常return

            category, detail = RESULT_MAP.get(result, ("recognition_incomplete", f"unknown_result:{result}"))

            final_dir = organize_case_dir(shot_dir, captures_run_dir, i)
            perf_case_dir = mirror_case_dir(final_dir, perf_run_dir, i) if final_dir is not None else None

            mark_done(
                progress_path, case_dict, i,
                category=category, detail=detail,
                book_width_mm=(ret if isinstance(ret, float) and ret != 0.0 else None),
                shot_dir_captures=str(final_dir) if final_dir else None,
                shot_dir_perf=str(perf_case_dir) if perf_case_dir else None,
            )

            if category in REDO_CATEGORIES:
                print(f"⚠ {category}({detail}) として記録しました。次回このケースはやり直します。")
            else:
                print(f"記録しました（{detail}）。次のケースへ進みます。")

            if isinstance(ret, float) and ret != 0.0:
                retrieved_book_width_list.append(ret)

    except KeyboardInterrupt:
        interrupted = True
        print("\n中断しました。progress.json に途中経過が保存されています。")
        print(f"再開するには: python {Path(__file__).name} --resume-dir {perf_run_dir}")

    finally:
        teardown_hardware(hw)

    run_info["status"] = "interrupted" if interrupted else "completed"
    run_info["finished_at"] = datetime.now().isoformat(timespec="seconds")
    write_json_atomic(perf_run_dir / "run_info.json", run_info)
    write_json_atomic(captures_run_dir / "run_info.json", run_info)

    n_done = sum(
        1 for c in case_dict.values()
        if c.get("state") == "done" and c.get("category") == "retrieval_outcome"
    )
    print(f"\n===== 終了（status={run_info['status']}） =====")
    print(f"有効な結果: {n_done} / {len(plan)}")


if __name__ == "__main__":
    main()
