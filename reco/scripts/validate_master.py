#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_catheter_*.json の内容を、実際に棚のカテーテル箱と突き合わせる前にチェックする。

2026-09-11、master_catheter_0910.json で以下のような問題が手作業で発覚したことを
きっかけに作成した:
  - book_name(REF)が複数行で重複している(うち一部は全フィールド完全一致のコピペミス)
  - book_name が重複していると、multikey_matcher.py はクエリ時に「最初に見つかった
    1件」だけを返す(_load_master→next()検索、EZAS3021の例では2028-12-09の行が常に
    選ばれ、2031-01-06の行は永久にクエリ対象として選ばれない)ため、気づかないまま
    運用すると特定の行が「存在しないのと同じ」になってしまう。

このスクリプトは棚とJSONの対応そのもの(欠け・余り)は見てくれない(それは別問題、
実際に何が棚にあるかを知っているのは人間だけ)。ここでやるのは、JSON単体で
機械的に判定できる不整合を洗い出すこと:
  1. book_name(REF)の重複(完全一致/一部フィールドのみ相違の両方を区別して表示)
  2. book_name が空 / 4文字未満(multikey_matcher.MIN_KEY_TEXT_LEN_FOR_MATCHにより
     _key_scoreが常に0点を返し、実質そのキーが機能しない)
  3. display_name が空(同上、identificationの主要キーが1本減る)
  4. expiration date が短すぎる/空(_digits_only後6桁未満は_key_scoreが常に0点)
  5. SPEC_1・SPEC_2 が両方空(数値専用比較なので0点になる。片方だけ空は正常運用として許容)

実行:
    python3 reco/scripts/validate_master.py --master-json ../catheter-100/master_catheter_0910.json
    python3 reco/scripts/validate_master.py --master-json catheter-100/master_catheter_0910.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RECO_ROOT = SCRIPT_DIR.parent
REPO_ROOT = RECO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT / "catheter" / "scripts"))
import multikey_matcher as mm  # noqa: E402


def _row_signature(row: dict) -> str:
    """行全体の完全一致判定用キー(フィールド順に依存しない)。"""
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def validate(master: list[dict]) -> list[str]:
    """検出した問題を人間可読な文字列のリストで返す(問題が無ければ空リスト)。"""
    problems: list[str] = []

    # ---- 1. book_name(REF)の重複 ----
    by_ref: dict[str, list[int]] = {}
    for idx, row in enumerate(master):
        ref = row.get("book_name", "")
        by_ref.setdefault(ref, []).append(idx)

    for ref, idxs in by_ref.items():
        if len(idxs) <= 1:
            continue
        rows = [master[i] for i in idxs]
        sigs = {_row_signature(r) for r in rows}
        if len(sigs) == 1:
            problems.append(
                f"[重複・完全一致] book_name={ref!r} が {len(idxs)}件、全フィールド完全一致 "
                f"(行番号 {[i + 1 for i in idxs]})。コピペミスの可能性が高い。"
            )
        else:
            problems.append(
                f"[重複・REFのみ一致] book_name={ref!r} が {len(idxs)}件あるが内容が異なる "
                f"(行番号 {[i + 1 for i in idxs]})。multikey_matcher.pyはクエリ時に最初の1件"
                f"(行{idxs[0] + 1})しか返さないため、他の{len(idxs) - 1}件はクエリ対象として"
                f"選ばれない。意図的な複数ロット登録なら別REFに分けるか、対応方針の検討が必要。"
            )
            for i in idxs:
                r = master[i]
                problems.append(
                    f"    行{i + 1}: LOT={r.get('LOT', '')!r} "
                    f"expiration={r.get('expiration date', '')!r} "
                    f"ISBN={r.get('ISBN_number', '')!r} "
                    f"width={r.get('book_width', '')!r}"
                )

    # ---- 2. book_name が空 / 短すぎる ----
    for idx, row in enumerate(master):
        ref = row.get("book_name", "") or ""
        if not ref:
            problems.append(f"[REF空] 行{idx + 1}: book_nameが空。クエリできない行になっている。")
        elif len(mm._normalize(ref)) < mm.MIN_KEY_TEXT_LEN_FOR_MATCH:
            problems.append(
                f"[REF短すぎ] 行{idx + 1}: book_name={ref!r} は{mm.MIN_KEY_TEXT_LEN_FOR_MATCH}文字"
                f"未満。_key_scoreの最小文字数ガードにより常に0点になり、refキーが機能しない。"
            )

    # ---- 3. display_name が空 ----
    for idx, row in enumerate(master):
        if not (row.get("display_name", "") or "").strip():
            problems.append(f"[display_name空] 行{idx + 1} (book_name={row.get('book_name', '')!r})")

    # ---- 4. expiration date が短すぎる/空 ----
    for idx, row in enumerate(master):
        date_val = row.get("expiration date", "") or ""
        digits = mm._digits_only(date_val)
        if len(digits) < 6:
            problems.append(
                f"[使用期限が不完全] 行{idx + 1} (book_name={row.get('book_name', '')!r}): "
                f"expiration date={date_val!r} → 数字{len(digits)}桁(6桁未満)。"
                f"dateキーが常に0点になる。"
            )

    # ---- 5. SPEC_1・SPEC_2 が両方空 ----
    for idx, row in enumerate(master):
        s1 = (row.get("SPEC_1", "") or "").strip()
        s2 = (row.get("SPEC_2", "") or "").strip()
        if not s1 and not s2:
            problems.append(
                f"[SPEC両方空] 行{idx + 1} (book_name={row.get('book_name', '')!r}): "
                f"SPEC_1・SPEC_2がどちらも空。識別に使えるキーがref/display_name/dateの3本のみになる。"
            )

    return problems


def print_checklist(master: list[dict]) -> None:
    """棚を見ながら目視で突き合わせるための一覧を、display_name順に並べて表示する。

    生のJSONを上から読むより、製品名(display_name)のアルファベット順に並んだ
    表のほうが「これはもう確認した/していない」を目で追いやすい。"""
    rows = sorted(enumerate(master, start=1), key=lambda p: (p[1].get("display_name", ""), p[1].get("book_name", "")))

    name_w = max((len(r.get("display_name", "")) for _, r in rows), default=12)
    ref_w = max((len(r.get("book_name", "")) for _, r in rows), default=6)

    print(f"{'#':>3}  {'display_name':<{name_w}}  {'book_name(REF)':<{ref_w}}  {'expiration':<12}  SPEC_1 / SPEC_2")
    print("-" * (3 + 2 + name_w + 2 + ref_w + 2 + 12 + 20))
    for line_no, row in rows:
        spec = f"{row.get('SPEC_1', '') or '-'} / {row.get('SPEC_2', '') or '-'}"
        print(
            f"{line_no:>3}  {row.get('display_name', ''):<{name_w}}  "
            f"{row.get('book_name', ''):<{ref_w}}  {row.get('expiration date', ''):<12}  {spec}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-json", required=True, help="チェック対象のマスタJSONパス")
    ap.add_argument(
        "--checklist", action="store_true",
        help="display_name順の一覧を表示する(棚を見ながらの目視突き合わせ用)"
    )
    args = ap.parse_args()

    master_path = Path(args.master_json)
    if not master_path.is_absolute():
        # カレントディレクトリ相対、それが無ければ reco/ 相対でも試す
        candidates = [Path.cwd() / master_path, RECO_ROOT / master_path]
        master_path = next((c for c in candidates if c.exists()), candidates[0])

    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)
    if not isinstance(master, list):
        raise SystemExit(f"マスタはリスト形式である必要があります: {master_path}")

    print(f"チェック対象: {master_path.resolve()}")
    print(f"件数: {len(master)}")
    print()

    if args.checklist:
        print_checklist(master)
        print()

    problems = validate(master)
    if not problems:
        print("問題は見つかりませんでした。")
        return

    print(f"{len(problems)}件の指摘:")
    for p in problems:
        print(f"  {p}")

    # book_name基準でのユニーク件数(重複を除いた実質的なクエリ可能種類数)も表示
    unique_refs = {row.get("book_name", "") for row in master}
    print()
    print(f"ユニークなbook_name(REF)数: {len(unique_refs)} / 全{len(master)}行")


if __name__ == "__main__":
    main()
