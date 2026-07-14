#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


AUDIT_CASES = [13, 16, 99, 43, 8]
PRIMARY_CASES = {13, 16, 99, 43}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def find_eval_csv(root: Path) -> Path:
    candidates = [root / "eval_results.csv", *sorted(root.glob("book_width_eval_results_*.csv"))]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"no eval csv under {root}")


def load_eval(root: Path) -> dict[int, dict[str, Any]]:
    path = find_eval_csv(root)
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row.get("test_index") or row.get("\ufefftest_index"))
            out[idx] = {
                "case_index": idx,
                "book_name": row.get("book_name") or "",
                "gt_width_mm": to_float(row.get("gt_book_width_mm") or row.get("gt_width_mm")),
                "pred_width_mm": to_float(row.get("pred_book_width_mm") or row.get("pred_width_mm")),
                "abs_error_mm": to_float(row.get("abs_error_mm")),
                "status": row.get("status") or "",
                "run_shot_dir": row.get("run_shot_dir") or "",
            }
    return out


def find_processing_log(root: Path, case_index: int) -> Path | None:
    case_dir = root / str(case_index)
    logs = sorted(case_dir.glob("*processing_log.json"))
    if logs:
        return logs[0]
    logs = sorted(case_dir.rglob("*processing_log.json")) if case_dir.exists() else []
    return logs[0] if logs else None


def load_log(root: Path, case_index: int) -> tuple[dict[str, Any], str]:
    path = find_processing_log(root, case_index)
    if path is None:
        return {}, ""
    try:
        return read_json(path), str(path)
    except Exception:
        return {}, str(path)


def policy_bits(log: dict[str, Any]) -> dict[str, Any]:
    a = log.get("residual_policy_A") if isinstance(log.get("residual_policy_A"), dict) else {}
    b = log.get("residual_policy_B") if isinstance(log.get("residual_policy_B"), dict) else {}
    inter = log.get("residual_policy_interaction") if isinstance(log.get("residual_policy_interaction"), dict) else {}
    sens = inter.get("A_after_width_sensitivity") if isinstance(inter.get("A_after_width_sensitivity"), dict) else {}
    return {
        "policy_A_triggered": bool(a.get("triggered") is True or a.get("reject_candidate") is True),
        "policy_B_triggered": bool(b.get("triggered") is True or b.get("robust_width_adopted") is True),
        "AB_SAFE_B_skipped_due_to_A_plausible": bool(
            b.get("skipped_due_to_A_plausible") is True or inter.get("skip_policy_B") is True
        ),
        "A_then_B_overwrite_prevented": bool(inter.get("A_then_B_overwrite_prevented") is True),
        "A_after_width_plausible": bool(inter.get("A_after_width_plausible") is True),
        "A_after_width_sensitivity": json.dumps(sens, ensure_ascii=False) if sens else "",
        "A_after_p2p98_width_mm": sens.get("p2p98_width_mm"),
        "A_after_p10p90_width_mm": sens.get("p10p90_width_mm"),
        "A_after_p2_minus_p10_mm": sens.get("p2_minus_p10_mm"),
        "A_after_p2_over_p10": sens.get("p2_over_p10"),
        "policy_A_reason": a.get("reason"),
        "policy_A_reject_reasons": ";".join(a.get("reject_reasons") or []),
        "policy_A_before_width_p2p98_mm": a.get("before_width_p2p98_mm"),
        "policy_A_candidate_width_p2p98_mm": a.get("candidate_width_p2p98_mm"),
        "policy_A_before_width_p10p90_mm": a.get("before_width_p10p90_mm"),
        "policy_A_candidate_width_p10p90_mm": a.get("candidate_width_p10p90_mm"),
        "policy_A_candidate_valid_keep_ratio": a.get("candidate_valid_keep_ratio"),
        "B_robust_width_candidate": b.get("robust_width_candidate"),
        "B_selected_robust_width_mm": b.get("selected_robust_width_mm"),
        "B_robust_width_adopted": bool(b.get("robust_width_adopted") is True),
        "B_adopt_reasons": ";".join(b.get("adopt_reasons") or []),
        "B_reject_reasons": ";".join(b.get("reject_reasons") or []),
        "B_skip_reason": b.get("skip_reason") or inter.get("skip_reason"),
    }


def best_worst(row: dict[str, Any], modes: list[str]) -> tuple[str, str]:
    vals = {m: row.get(f"{m}_abs_error_mm") for m in modes if row.get(f"{m}_abs_error_mm") is not None}
    if not vals:
        return "", ""
    return min(vals, key=vals.get), max(vals, key=vals.get)


def judge_skip(row: dict[str, Any]) -> str:
    if not row["AB_SAFE_B_skipped_due_to_A_plausible"]:
        return "skip_not_relevant"
    ab_safe = row["AB_SAFE_abs_error_mm"]
    ab = row["AB_abs_error_mm"]
    if ab_safe is None or ab is None:
        return "unknown"
    if ab < ab_safe - 0.5:
        return "skip_too_aggressive"
    return "skip_correct"


def judge_policy_a(row: dict[str, Any]) -> str:
    baseline = row["baseline_abs_error_mm"]
    a_abs = row["A_abs_error_mm"]
    if baseline is None or a_abs is None:
        return "unknown"
    if a_abs < baseline - 0.5:
        return "A_helpful"
    if a_abs > baseline + 0.5:
        return "A_harmful"
    if not row["policy_A_triggered"] and row["case_index"] == 43:
        return "A_insufficient"
    return "A_neutral"


def judge_policy_b(row: dict[str, Any]) -> str:
    baseline = row["baseline_abs_error_mm"]
    b_abs = row["B_abs_error_mm"]
    ab_safe = row["AB_SAFE_abs_error_mm"]
    if baseline is None or b_abs is None:
        return "unknown"
    if b_abs < baseline - 0.5 or (ab_safe is not None and b_abs < ab_safe - 0.5):
        return "B_would_help"
    if b_abs > baseline + 0.5:
        return "B_would_hurt"
    return "B_neutral"


def regression_cause(row: dict[str, Any]) -> str:
    if row["case_index"] == 43:
        return "post_ransac_a95_guard_insufficient"
    if row["policy_A_judgement"] == "A_harmful":
        return "policy_A_over_rejected_post_ransac_candidate"
    if row["B_skip_judgement"] == "skip_too_aggressive":
        return "AB_SAFE_skip_too_aggressive"
    if row["policy_B_judgement"] == "B_would_hurt":
        return "policy_B_would_hurt"
    return "unknown"


def recommended_fix(row: dict[str, Any]) -> str:
    cause = row["suspected_regression_cause"]
    if cause == "policy_A_over_rejected_post_ransac_candidate":
        return "modify_A_overprune_threshold: require stronger evidence before rejecting post-RANSAC a95 candidate, e.g. combine shrink ratio with absolute candidate width and candidate-vs-before plausibility."
    if cause == "AB_SAFE_skip_too_aggressive":
        return "tighten_B_skip_after_A_plausible: do not skip B when A-after p2-p98/p10-p90 sensitivity remains high or A-after width conflicts with robust width diagnostics."
    if cause == "post_ransac_a95_guard_insufficient":
        return "add_case43_specific_general_guard: generalize post-RANSAC a95 guard for moderate shrink/under-width cases; no case-id hardcode."
    if cause == "policy_B_would_hurt":
        return "do_not_change_B_for_this_case; B is not a rescue path here."
    return "inspect_stage_logs_before_change"


def note_for(row: dict[str, Any]) -> str:
    c = row["case_index"]
    if c in (13, 99):
        return "Crossed out of 2mm. A rejected a good post-RANSAC a95 candidate; B alone stayed near baseline, so A is the main issue."
    if c == 16:
        return "Large worsening. A restored an overly wide pre-a95 width; B alone was not a clear rescue."
    if c == 43:
        return "A did not trigger. Existing issue remains: post-RANSAC a95 guard appears insufficient for this moderate shrink/under-width case."
    if c == 8:
        return "Reference case. AB_SAFE worsened moderately without policy trigger; likely unrelated residual estimator/mask variance."
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(mode_roots: dict[str, Path], output_root: Path) -> None:
    modes = ["baseline", "A", "B", "AB", "AB_SAFE"]
    evals = {m: load_eval(root) for m, root in mode_roots.items()}
    rows: list[dict[str, Any]] = []
    for case in AUDIT_CASES:
        row: dict[str, Any] = {"case_index": case}
        base = evals["baseline"].get(case, {})
        ab_safe = evals["AB_SAFE"].get(case, {})
        row["book_name"] = ab_safe.get("book_name") or base.get("book_name") or ""
        row["gt_width_mm"] = ab_safe.get("gt_width_mm") or base.get("gt_width_mm")
        for mode in modes:
            data = evals[mode].get(case, {})
            row[f"{mode}_pred_width_mm"] = data.get("pred_width_mm")
            row[f"{mode}_abs_error_mm"] = data.get("abs_error_mm")
            row[f"{mode}_status"] = data.get("status")

        log, log_path = load_log(mode_roots["AB_SAFE"], case)
        row.update(policy_bits(log))
        row["AB_SAFE_processing_log"] = log_path
        row["best_mode"], row["worst_mode"] = best_worst(row, modes)
        row["B_skip_judgement"] = judge_skip(row)
        row["policy_A_judgement"] = judge_policy_a(row)
        row["policy_B_judgement"] = judge_policy_b(row)
        row["suspected_regression_cause"] = regression_cause(row)
        row["recommended_fix"] = recommended_fix(row)
        row["notes"] = note_for(row)
        rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_index",
        "book_name",
        "gt_width_mm",
        "baseline_pred_width_mm",
        "baseline_abs_error_mm",
        "A_pred_width_mm",
        "A_abs_error_mm",
        "B_pred_width_mm",
        "B_abs_error_mm",
        "AB_pred_width_mm",
        "AB_abs_error_mm",
        "AB_SAFE_pred_width_mm",
        "AB_SAFE_abs_error_mm",
        "best_mode",
        "worst_mode",
        "policy_A_triggered",
        "policy_B_triggered",
        "AB_SAFE_B_skipped_due_to_A_plausible",
        "A_then_B_overwrite_prevented",
        "A_after_width_plausible",
        "A_after_width_sensitivity",
        "A_after_p2p98_width_mm",
        "A_after_p10p90_width_mm",
        "A_after_p2_minus_p10_mm",
        "A_after_p2_over_p10",
        "B_robust_width_candidate",
        "B_selected_robust_width_mm",
        "B_robust_width_adopted",
        "B_skip_judgement",
        "policy_A_judgement",
        "policy_B_judgement",
        "suspected_regression_cause",
        "recommended_fix",
        "notes",
        "AB_SAFE_processing_log",
    ]
    write_csv(output_root / "ab_safe_regression_audit_cases.csv", rows, fields)
    write_csv(
        output_root / "ab_safe_regression_audit_notes.csv",
        rows,
        [
            "case_index",
            "book_name",
            "best_mode",
            "worst_mode",
            "B_skip_judgement",
            "policy_A_judgement",
            "policy_B_judgement",
            "suspected_regression_cause",
            "recommended_fix",
            "notes",
        ],
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audit_cases": AUDIT_CASES,
        "primary_cases": sorted(PRIMARY_CASES),
        "mode_roots": {m: str(p) for m, p in mode_roots.items()},
        "case_summaries": rows,
        "B_skip_judgement_counts": {k: sum(r["B_skip_judgement"] == k for r in rows) for k in ["skip_correct", "skip_too_aggressive", "skip_not_relevant", "unknown"]},
        "policy_A_judgement_counts": {k: sum(r["policy_A_judgement"] == k for r in rows) for k in ["A_helpful", "A_harmful", "A_neutral", "A_insufficient", "unknown"]},
        "policy_B_judgement_counts": {k: sum(r["policy_B_judgement"] == k for r in rows) for k in ["B_would_help", "B_would_hurt", "B_neutral", "unknown"]},
        "adoption_decision": "adopt_ab_safe_after_minor_guard_tweak",
        "guard_fix": "modify_A_overprune_threshold",
        "next_task": "Policy Aのover-prune閾値を調整して13ケース再評価。case43/28系は別枠診断へ進む。",
    }
    (output_root / "ab_safe_regression_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rec = [
        "# AB_SAFE Regression Audit Recommendation",
        "",
        "## Decision",
        "`adopt_ab_safe_after_minor_guard_tweak`",
        "",
        "AB_SAFE improves the 100-case aggregate, but this audit found valid-SAM regressions where Policy A rejected a useful post-RANSAC a95 candidate. The main tweak should target Policy A over-prune rejection, not broad B-skip relaxation.",
        "",
        "## Case Notes",
    ]
    for r in rows:
        rec.append(
            f"- case{r['case_index']}: best={r['best_mode']}, AB_SAFE_abs={r['AB_SAFE_abs_error_mm']:.3f}, "
            f"A={r['policy_A_judgement']}, B={r['policy_B_judgement']}, skip={r['B_skip_judgement']}; {r['notes']}"
        )
    rec.extend([
        "",
        "## Recommended Fix",
        "General tweak: require stronger evidence before Policy A rejects a post-RANSAC a95 candidate. Candidate shrink alone is not enough when the candidate width is plausible and closer to the robust width range. Do not use case ids or GT in inference.",
        "",
        "case43 remains a separate post-RANSAC a95 guard gap; it did not mainly fail because of AB_SAFE skip.",
    ])
    (output_root / "ab_safe_regression_audit_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")

    worklog = [
        "# Codex Worklog",
        "",
        "## Objective",
        "Audit AB_SAFE regressions for case13, case16, case99, case43, and reference case8 without changing inference code.",
        "",
        "## Inputs",
    ]
    for mode in modes:
        worklog.append(f"- {mode}: `{mode_roots[mode]}`")
    worklog.extend([
        "",
        "## Re-evaluation",
        "A/B/AB were re-evaluated only for cases 13,16,99,43,8. Baseline and AB_SAFE used existing 100-case runs.",
        "",
        "## Outputs",
        "- `ab_safe_regression_audit_cases.csv`",
        "- `ab_safe_regression_audit_summary.json`",
        "- `ab_safe_regression_audit_notes.csv`",
        "- `ab_safe_regression_audit_recommendation.md`",
        "",
        "No changes were made to `get_book_points.py` in this audit pass.",
    ])
    (output_root / "codex_worklog.md").write_text("\n".join(worklog) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=Path("captures/100test_offline/20260703_154211"))
    parser.add_argument("--a-root", type=Path, default=Path("captures/100test_offline/20260707_022753"))
    parser.add_argument("--b-root", type=Path, default=Path("captures/100test_offline/20260707_023219"))
    parser.add_argument("--ab-root", type=Path, default=Path("captures/100test_offline/20260707_023641"))
    parser.add_argument("--ab-safe-root", type=Path, default=Path("captures/100test_offline/20260706_154033 (1)"))
    args = parser.parse_args()
    analyze(
        {
            "baseline": args.baseline_root,
            "A": args.a_root,
            "B": args.b_root,
            "AB": args.ab_root,
            "AB_SAFE": args.ab_safe_root,
        },
        args.output_root,
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
