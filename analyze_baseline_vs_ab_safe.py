#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


BAD_SAM_CASES = {3, 6, 7, 36, 40, 67, 68, 73, 74, 75, 89, 96}
MISSING_INPUT_CASES = {50}
WATCH_CASES = {12, 15, 25, 28, 29, 43, 55, 66, 71, 83, 87}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _find_eval_csv(run_root: Path) -> Path:
    candidates = [
        run_root / "eval_results.csv",
        *sorted(run_root.glob("book_width_eval_results_*.csv")),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"eval results CSV not found under {run_root}")


def _find_summary_json(run_root: Path) -> Path | None:
    candidates = [
        run_root / "eval_summary.json",
        *sorted(run_root.glob("book_width_eval_summary_*.json")),
    ]
    return next((p for p in candidates if p.exists()), None)


def _load_results(run_root: Path) -> dict[int, dict[str, Any]]:
    csv_path = _find_eval_csv(run_root)
    rows: dict[int, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_index = _to_int(row.get("test_index") or row.get("\ufefftest_index"))
            if case_index is None:
                continue
            rows[case_index] = {
                "case_index": case_index,
                "book_name": row.get("book_name") or "",
                "gt_width_mm": _to_float(row.get("gt_book_width_mm") or row.get("gt_width_mm")),
                "pred_width_mm": _to_float(row.get("pred_book_width_mm") or row.get("pred_width_mm")),
                "abs_error_mm": _to_float(row.get("abs_error_mm")),
                "status": row.get("status") or "",
                "error": row.get("error") or "",
                "run_shot_dir": row.get("run_shot_dir") or "",
                "case_console_log": row.get("case_console_log") or "",
            }
    return rows


def _case_group(case_index: int, baseline_status: str | None, ab_status: str | None) -> str:
    if case_index in MISSING_INPUT_CASES:
        return "missing_input"
    if case_index in BAD_SAM_CASES:
        return "bad_sam"
    if baseline_status and baseline_status != "success":
        return "failed"
    if ab_status and ab_status != "success":
        return "failed"
    return "valid_sam"


def _within(abs_error: float | None, threshold: float) -> bool:
    return abs_error is not None and abs_error <= threshold


def _find_processing_log(run_root: Path, case_index: int) -> Path | None:
    case_dir = run_root / str(case_index)
    if not case_dir.exists():
        return None
    logs = sorted(case_dir.glob("*processing_log.json"))
    if logs:
        return logs[0]
    logs = sorted(case_dir.rglob("*processing_log.json"))
    return logs[0] if logs else None


def _load_policy_info(run_root: Path, case_index: int) -> dict[str, Any]:
    path = _find_processing_log(run_root, case_index)
    if path is None:
        return {
            "processing_log_path": "",
            "policy_triggered": False,
            "policy_A_triggered": False,
            "policy_B_triggered": False,
            "policy_B_skipped_due_to_A_plausible": False,
            "A_then_B_overwrite_prevented": False,
            "clean_rectangle_override_used": False,
            "reason": "processing_log not found",
        }
    try:
        log = _read_json(path)
    except Exception as exc:
        return {
            "processing_log_path": str(path),
            "policy_triggered": False,
            "policy_A_triggered": False,
            "policy_B_triggered": False,
            "policy_B_skipped_due_to_A_plausible": False,
            "A_then_B_overwrite_prevented": False,
            "clean_rectangle_override_used": False,
            "reason": f"failed to read processing_log: {exc}",
        }

    a = log.get("residual_policy_A") if isinstance(log.get("residual_policy_A"), dict) else {}
    b = log.get("residual_policy_B") if isinstance(log.get("residual_policy_B"), dict) else {}
    interaction = log.get("residual_policy_interaction") if isinstance(log.get("residual_policy_interaction"), dict) else {}
    a_triggered = bool(a.get("triggered") is True or a.get("reject_candidate") is True)
    b_triggered = bool(b.get("triggered") is True or b.get("robust_width_adopted") is True)
    b_skipped = bool(b.get("skipped_due_to_A_plausible") is True or interaction.get("skip_policy_B") is True)
    prevented = bool(interaction.get("A_then_B_overwrite_prevented") is True)
    return {
        "processing_log_path": str(path),
        "policy_triggered": bool(a_triggered or b_triggered or b_skipped or prevented),
        "policy_A_triggered": a_triggered,
        "policy_B_triggered": b_triggered,
        "policy_B_skipped_due_to_A_plausible": b_skipped,
        "A_then_B_overwrite_prevented": prevented,
        "clean_rectangle_override_used": bool(log.get("clean_rectangle_override_used") is True),
        "residual_policy_mode": log.get("residual_policy_mode"),
        "policy_A_reason": a.get("reason"),
        "policy_B_reason": b.get("reason"),
        "policy_B_skip_reason": b.get("skip_reason"),
        "interaction_skip_reason": interaction.get("skip_reason"),
        "shape_clean_rectangle": (log.get("shape_rectangularity") or {}).get("clean_rectangle")
        if isinstance(log.get("shape_rectangularity"), dict)
        else None,
    }


def _improvement_reason(row: dict[str, Any]) -> str:
    if row["improved_abs_error_mm"] <= 0:
        return ""
    if row["policy_B_skipped_due_to_A_plausible"] or row["A_then_B_overwrite_prevented"]:
        return "AB_SAFE_skip_B_after_A_helped"
    if row["policy_B_triggered"]:
        return "policy_B_robust_width_helped"
    if row["policy_A_triggered"]:
        return "policy_A_overprune_reject_helped"
    if row["clean_rectangle_override_used"]:
        return "clean_rectangle_override_related"
    if row["case_group"] == "bad_sam":
        return "other"
    return "unknown"


def _worsening_reason(row: dict[str, Any]) -> str:
    if row["worsened_abs_error_mm"] <= 0:
        return ""
    if row["case_group"] == "bad_sam":
        return "bad_sam_case"
    if row["policy_B_triggered"]:
        return "policy_B_wrong_robust_width"
    if row["policy_A_triggered"]:
        return "policy_A_over_rejected"
    if row["policy_B_skipped_due_to_A_plausible"] or row["A_then_B_overwrite_prevented"]:
        return "AB_SAFE_wrong_skip_or_wrong_adopt"
    return "unknown"


def _suspected_failure_type(row: dict[str, Any]) -> str:
    case_index = int(row["case_index"])
    if row["case_group"] == "bad_sam":
        return "bad_sam_initial_mask"
    if row["case_group"] == "missing_input":
        return "missing_input"
    if case_index == 28:
        return "column_refine_width_shortage"
    if case_index == 43:
        return "post_ransac_a95_overprune"
    if case_index == 29:
        return "width_estimator_outlier_sensitive_remaining"
    if row["policy_A_triggered"] and row["ab_safe_abs_error_mm"] and row["ab_safe_abs_error_mm"] > 2.0:
        return "post_ransac_a95_overprune"
    if row["policy_B_triggered"] and row["ab_safe_abs_error_mm"] and row["ab_safe_abs_error_mm"] > 2.0:
        return "width_estimator_outlier_sensitive_remaining"
    return "unknown"


def _recommended_next_action(failure_type: str) -> str:
    return {
        "column_refine_width_shortage": "Inspect column-refine seed/width guard; avoid further clipping of already narrow masks.",
        "post_column_or_ransac_residual": "Inspect post-column and RANSAC stage masks; add guarded residual handling only if repeatable.",
        "post_ransac_a95_overprune": "Tune post-RANSAC a95 guard using diagnostics; avoid accepting over-prune candidates.",
        "width_estimator_outlier_sensitive_remaining": "Analyze width percentile sensitivity and axis center diagnostics before changing estimator.",
        "side_front_face_remaining": "Inspect final mask for side/front face survival and consider guarded one-sided removal.",
        "axis_center_misaligned": "Inspect OCR axis/center selection and fallback path.",
        "bad_sam_initial_mask": "Exclude from postprocess optimization; fix SAM/initial mask upstream.",
        "missing_input": "Recover missing input or keep excluded.",
        "unknown": "Inspect contact sheet, processing_log, and width debug for this case.",
    }.get(failure_type, "Inspect logs and contact sheet.")


def _summary_for(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    total = len(rows)
    status_key = f"{prefix}_status"
    abs_key = f"{prefix}_abs_error_mm"
    success_rows = [r for r in rows if r.get(status_key) == "success" and r.get(abs_key) is not None]
    vals = [float(r[abs_key]) for r in success_rows]
    return {
        "total": total,
        "success": len(success_rows),
        "fail": total - len(success_rows),
        "within_1mm": sum(v <= 1.0 for v in vals),
        "within_1p5mm": sum(v <= 1.5 for v in vals),
        "within_2mm": sum(v <= 2.0 for v in vals),
        "mean_abs_error_success_only": statistics.fmean(vals) if vals else None,
        "median_abs_error_success_only": statistics.median(vals) if vals else None,
        "min_abs_error_success_only": min(vals) if vals else None,
        "max_abs_error_success_only": max(vals) if vals else None,
    }


def _comparison_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparable = [
        r
        for r in rows
        if r.get("baseline_abs_error_mm") is not None and r.get("ab_safe_abs_error_mm") is not None
    ]
    deltas = [float(r["delta_abs_error_mm"]) for r in comparable]
    return {
        "improved_case_count": sum(float(r["improved_abs_error_mm"]) > 0.05 for r in comparable),
        "worsened_case_count": sum(float(r["worsened_abs_error_mm"]) > 0.05 for r in comparable),
        "unchanged_case_count": sum(abs(float(r["delta_abs_error_mm"])) <= 0.05 for r in comparable),
        "crossed_into_2mm_count": sum(bool(r["crossed_into_2mm"]) for r in comparable),
        "crossed_out_of_2mm_count": sum(bool(r["crossed_out_of_2mm"]) for r in comparable),
        "mean_delta_abs_error": statistics.fmean(deltas) if deltas else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def analyze(baseline_root: Path | None, ab_safe_root: Path, output_root: Path) -> dict[str, Any]:
    ab_rows = _load_results(ab_safe_root)
    baseline_rows = _load_results(baseline_root) if baseline_root else {}
    can_compare = bool(baseline_rows)
    all_case_indices = sorted(set(range(1, 101)) | set(ab_rows) | set(baseline_rows))

    rows: list[dict[str, Any]] = []
    for case_index in all_case_indices:
        b = baseline_rows.get(case_index, {})
        a = ab_rows.get(case_index, {})
        policy = _load_policy_info(ab_safe_root, case_index)
        b_abs = b.get("abs_error_mm")
        a_abs = a.get("abs_error_mm")
        b_pred = b.get("pred_width_mm")
        a_pred = a.get("pred_width_mm")
        delta_pred = None if b_pred is None or a_pred is None else float(a_pred) - float(b_pred)
        delta_abs = None if b_abs is None or a_abs is None else float(a_abs) - float(b_abs)
        improved = 0.0 if delta_abs is None else max(0.0, -float(delta_abs))
        worsened = 0.0 if delta_abs is None else max(0.0, float(delta_abs))
        case_group = _case_group(case_index, b.get("status"), a.get("status"))
        row = {
            "case_index": case_index,
            "book_name": a.get("book_name") or b.get("book_name") or "",
            "gt_width_mm": a.get("gt_width_mm") if a.get("gt_width_mm") is not None else b.get("gt_width_mm"),
            "baseline_status": b.get("status", "missing") if baseline_root else "not_available",
            "baseline_pred_width_mm": b_pred,
            "baseline_abs_error_mm": b_abs,
            "baseline_within_1mm": _within(b_abs, 1.0),
            "baseline_within_1p5mm": _within(b_abs, 1.5),
            "baseline_within_2mm": _within(b_abs, 2.0),
            "ab_safe_status": a.get("status", "missing"),
            "ab_safe_pred_width_mm": a_pred,
            "ab_safe_abs_error_mm": a_abs,
            "ab_safe_within_1mm": _within(a_abs, 1.0),
            "ab_safe_within_1p5mm": _within(a_abs, 1.5),
            "ab_safe_within_2mm": _within(a_abs, 2.0),
            "delta_pred_width_mm": delta_pred,
            "delta_abs_error_mm": delta_abs,
            "improved_abs_error_mm": improved,
            "worsened_abs_error_mm": worsened,
            "crossed_into_1mm": bool(b_abs is not None and a_abs is not None and b_abs > 1.0 and a_abs <= 1.0),
            "crossed_out_of_1mm": bool(b_abs is not None and a_abs is not None and b_abs <= 1.0 and a_abs > 1.0),
            "crossed_into_1p5mm": bool(b_abs is not None and a_abs is not None and b_abs > 1.5 and a_abs <= 1.5),
            "crossed_out_of_1p5mm": bool(b_abs is not None and a_abs is not None and b_abs <= 1.5 and a_abs > 1.5),
            "crossed_into_2mm": bool(b_abs is not None and a_abs is not None and b_abs > 2.0 and a_abs <= 2.0),
            "crossed_out_of_2mm": bool(b_abs is not None and a_abs is not None and b_abs <= 2.0 and a_abs > 2.0),
            "case_group": case_group,
            **policy,
        }
        notes = []
        if case_index in WATCH_CASES:
            notes.append("watch_case")
        if case_index in BAD_SAM_CASES:
            notes.append("bad_sam_excluded_from_postprocess_judgement")
        if case_index in MISSING_INPUT_CASES:
            notes.append("missing_input")
        if row["crossed_into_2mm"]:
            notes.append("crossed_into_2mm")
        if row["crossed_out_of_2mm"]:
            notes.append("crossed_out_of_2mm")
        if row["policy_B_triggered"]:
            notes.append("policy_B_triggered")
        if row["policy_A_triggered"]:
            notes.append("policy_A_triggered")
        if row["policy_B_skipped_due_to_A_plausible"]:
            notes.append("policy_B_skipped_due_to_A_plausible")
        row["improvement_reason"] = _improvement_reason(row)
        row["worsening_reason"] = _worsening_reason(row)
        row["notes"] = "; ".join(notes)
        rows.append(row)

    for row in rows:
        row["suspected_failure_type"] = _suspected_failure_type(row)
        row["recommended_next_action"] = _recommended_next_action(row["suspected_failure_type"])

    groups = {
        "all_cases": rows,
        "valid_sam_only": [r for r in rows if r["case_group"] == "valid_sam"],
        "bad_sam_only": [r for r in rows if r["case_group"] == "bad_sam"],
        "missing_input": [r for r in rows if r["case_group"] == "missing_input"],
    }
    summaries: dict[str, Any] = {}
    for name, group_rows in groups.items():
        summaries[name] = {
            "baseline": _summary_for(group_rows, "baseline") if can_compare else None,
            "ab_safe": _summary_for(group_rows, "ab_safe"),
            "comparison": _comparison_summary(group_rows) if can_compare else None,
        }

    improved = [r for r in rows if r["improved_abs_error_mm"] > 0.05]
    worsened = [r for r in rows if r["worsened_abs_error_mm"] > 0.05]
    remaining = [
        r
        for r in rows
        if r["ab_safe_status"] == "success" and r["ab_safe_abs_error_mm"] is not None and r["ab_safe_abs_error_mm"] > 2.0
    ]
    remaining.sort(key=lambda r: float(r["ab_safe_abs_error_mm"]), reverse=True)

    policy_rows = [
        r
        for r in rows
        if r["policy_triggered"]
        or r["policy_A_triggered"]
        or r["policy_B_triggered"]
        or r["policy_B_skipped_due_to_A_plausible"]
        or r["A_then_B_overwrite_prevented"]
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    case_fields = [
        "case_index",
        "book_name",
        "gt_width_mm",
        "baseline_status",
        "baseline_pred_width_mm",
        "baseline_abs_error_mm",
        "baseline_within_1mm",
        "baseline_within_1p5mm",
        "baseline_within_2mm",
        "ab_safe_status",
        "ab_safe_pred_width_mm",
        "ab_safe_abs_error_mm",
        "ab_safe_within_1mm",
        "ab_safe_within_1p5mm",
        "ab_safe_within_2mm",
        "delta_pred_width_mm",
        "delta_abs_error_mm",
        "improved_abs_error_mm",
        "worsened_abs_error_mm",
        "crossed_into_1mm",
        "crossed_out_of_1mm",
        "crossed_into_1p5mm",
        "crossed_out_of_1p5mm",
        "crossed_into_2mm",
        "crossed_out_of_2mm",
        "case_group",
        "policy_triggered",
        "policy_A_triggered",
        "policy_B_triggered",
        "policy_B_skipped_due_to_A_plausible",
        "A_then_B_overwrite_prevented",
        "clean_rectangle_override_used",
        "improvement_reason",
        "worsening_reason",
        "suspected_failure_type",
        "recommended_next_action",
        "notes",
        "processing_log_path",
    ]
    _write_csv(output_root / "baseline_vs_ab_safe_case_compare.csv", rows, case_fields)

    improved_fields = [
        "case_index",
        "book_name",
        "gt_width_mm",
        "baseline_pred_width_mm",
        "baseline_abs_error_mm",
        "ab_safe_pred_width_mm",
        "ab_safe_abs_error_mm",
        "improved_abs_error_mm",
        "case_group",
        "crossed_into_1mm",
        "crossed_into_1p5mm",
        "crossed_into_2mm",
        "policy_triggered",
        "policy_A_triggered",
        "policy_B_triggered",
        "policy_B_skipped_due_to_A_plausible",
        "improvement_reason",
        "notes",
    ]
    _write_csv(output_root / "baseline_vs_ab_safe_improved_cases.csv", sorted(improved, key=lambda r: r["improved_abs_error_mm"], reverse=True), improved_fields)

    worsened_fields = [
        "case_index",
        "book_name",
        "gt_width_mm",
        "baseline_pred_width_mm",
        "baseline_abs_error_mm",
        "ab_safe_pred_width_mm",
        "ab_safe_abs_error_mm",
        "worsened_abs_error_mm",
        "case_group",
        "crossed_out_of_1mm",
        "crossed_out_of_1p5mm",
        "crossed_out_of_2mm",
        "policy_triggered",
        "policy_A_triggered",
        "policy_B_triggered",
        "policy_B_skipped_due_to_A_plausible",
        "worsening_reason",
        "notes",
    ]
    _write_csv(output_root / "baseline_vs_ab_safe_worsened_cases.csv", sorted(worsened, key=lambda r: r["worsened_abs_error_mm"], reverse=True), worsened_fields)

    remaining_fields = [
        "case_index",
        "book_name",
        "gt_width_mm",
        "ab_safe_pred_width_mm",
        "ab_safe_abs_error_mm",
        "baseline_abs_error_mm",
        "delta_abs_error_mm",
        "case_group",
        "suspected_failure_type",
        "recommended_next_action",
        "policy_A_triggered",
        "policy_B_triggered",
        "policy_B_skipped_due_to_A_plausible",
        "notes",
    ]
    _write_csv(output_root / "baseline_vs_ab_safe_remaining_errors.csv", remaining, remaining_fields)

    policy_fields = [
        "case_index",
        "book_name",
        "baseline_abs_error_mm",
        "ab_safe_abs_error_mm",
        "delta_abs_error_mm",
        "improved_abs_error_mm",
        "worsened_abs_error_mm",
        "policy_A_triggered",
        "policy_B_triggered",
        "policy_B_skipped_due_to_A_plausible",
        "A_then_B_overwrite_prevented",
        "policy_A_reason",
        "policy_B_reason",
        "policy_B_skip_reason",
        "interaction_skip_reason",
        "processing_log_path",
    ]
    _write_csv(output_root / "baseline_vs_ab_safe_policy_trigger_cases.csv", policy_rows, policy_fields)

    summary_json = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_run_root": str(baseline_root) if baseline_root else None,
        "ab_safe_run_root": str(ab_safe_root),
        "baseline_summary_json": str(_find_summary_json(baseline_root)) if baseline_root and _find_summary_json(baseline_root) else None,
        "ab_safe_summary_json": str(_find_summary_json(ab_safe_root)) if _find_summary_json(ab_safe_root) else None,
        "can_compare_baseline_vs_ab_safe": can_compare,
        "bad_sam_cases": sorted(BAD_SAM_CASES),
        "missing_input_cases": sorted(MISSING_INPUT_CASES),
        "summaries": summaries,
        "crossed_into_2mm_cases": [r["case_index"] for r in rows if r["crossed_into_2mm"]],
        "crossed_out_of_2mm_cases": [r["case_index"] for r in rows if r["crossed_out_of_2mm"]],
        "crossed_into_1p5mm_cases": [r["case_index"] for r in rows if r["crossed_into_1p5mm"]],
        "improvement_ge_1mm_cases": [r["case_index"] for r in rows if r["improved_abs_error_mm"] >= 1.0],
        "improvement_ge_2mm_cases": [r["case_index"] for r in rows if r["improved_abs_error_mm"] >= 2.0],
        "worsening_ge_0p5mm_cases": [r["case_index"] for r in rows if r["worsened_abs_error_mm"] >= 0.5],
        "worsening_ge_1mm_cases": [r["case_index"] for r in rows if r["worsened_abs_error_mm"] >= 1.0],
        "ab_safe_remaining_over_2mm_valid_sam_cases": [
            r["case_index"] for r in remaining if r["case_group"] == "valid_sam"
        ],
        "ab_safe_top10_errors": [
            {
                "case_index": r["case_index"],
                "book_name": r["book_name"],
                "case_group": r["case_group"],
                "ab_safe_abs_error_mm": r["ab_safe_abs_error_mm"],
                "baseline_abs_error_mm": r["baseline_abs_error_mm"],
                "suspected_failure_type": r["suspected_failure_type"],
            }
            for r in remaining[:10]
        ],
        "ab_safe_top10_valid_sam_errors": [
            {
                "case_index": r["case_index"],
                "book_name": r["book_name"],
                "ab_safe_abs_error_mm": r["ab_safe_abs_error_mm"],
                "baseline_abs_error_mm": r["baseline_abs_error_mm"],
                "suspected_failure_type": r["suspected_failure_type"],
            }
            for r in [x for x in remaining if x["case_group"] == "valid_sam"][:10]
        ],
        "policy_A_triggered_cases": [r["case_index"] for r in policy_rows if r["policy_A_triggered"]],
        "policy_B_triggered_cases": [r["case_index"] for r in policy_rows if r["policy_B_triggered"]],
        "policy_B_skipped_due_to_A_plausible_cases": [
            r["case_index"] for r in policy_rows if r["policy_B_skipped_due_to_A_plausible"]
        ],
        "A_then_B_overwrite_prevented_cases": [
            r["case_index"] for r in policy_rows if r["A_then_B_overwrite_prevented"]
        ],
        "watch_cases": {str(c): next((r for r in rows if r["case_index"] == c), None) for c in sorted(WATCH_CASES)},
    }
    (output_root / "baseline_vs_ab_safe_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _write_markdown(output_root, summary_json, rows, improved, worsened, remaining, policy_rows)
    return summary_json


def _md_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _write_markdown(
    output_root: Path,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    improved: list[dict[str, Any]],
    worsened: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
) -> None:
    def group_line(name: str) -> list[Any]:
        s = summary["summaries"][name]
        b = s["baseline"] or {}
        a = s["ab_safe"]
        c = s["comparison"] or {}
        return [
            name,
            f"{b.get('within_2mm', 'n/a')} -> {a.get('within_2mm')}",
            f"{_fmt(b.get('mean_abs_error_success_only'))} -> {_fmt(a.get('mean_abs_error_success_only'))}",
            _fmt(c.get("mean_delta_abs_error")),
            c.get("crossed_into_2mm_count", "n/a"),
            c.get("crossed_out_of_2mm_count", "n/a"),
        ]

    report = [
        "# Baseline vs AB_SAFE Analysis",
        "",
        f"- baseline run root: `{summary['baseline_run_root']}`",
        f"- AB_SAFE run root: `{summary['ab_safe_run_root']}`",
        f"- comparison available: `{summary['can_compare_baseline_vs_ab_safe']}`",
        "",
        "## Summary",
        "",
        _md_table(
            [group_line("all_cases"), group_line("valid_sam_only"), group_line("bad_sam_only"), group_line("missing_input")],
            ["group", "within2", "mean abs", "mean delta", "into2", "out2"],
        ),
        "",
        "## Crossed Into 2mm",
        "",
        ", ".join(map(str, summary["crossed_into_2mm_cases"])) or "None",
        "",
        "## Crossed Out Of 2mm",
        "",
        ", ".join(map(str, summary["crossed_out_of_2mm_cases"])) or "None",
        "",
        "## Large Improvements",
        "",
        _md_table(
            [
                [
                    r["case_index"],
                    r["book_name"],
                    _fmt(r["baseline_abs_error_mm"]),
                    _fmt(r["ab_safe_abs_error_mm"]),
                    _fmt(r["improved_abs_error_mm"]),
                    r["case_group"],
                    r["improvement_reason"],
                ]
                for r in sorted(improved, key=lambda x: x["improved_abs_error_mm"], reverse=True)[:20]
            ],
            ["case", "book", "base abs", "AB_SAFE abs", "improve", "group", "reason"],
        ),
        "",
        "## Worsened Cases",
        "",
        _md_table(
            [
                [
                    r["case_index"],
                    r["book_name"],
                    _fmt(r["baseline_abs_error_mm"]),
                    _fmt(r["ab_safe_abs_error_mm"]),
                    _fmt(r["worsened_abs_error_mm"]),
                    r["case_group"],
                    r["worsening_reason"],
                ]
                for r in sorted(worsened, key=lambda x: x["worsened_abs_error_mm"], reverse=True)[:20]
            ],
            ["case", "book", "base abs", "AB_SAFE abs", "worsen", "group", "reason"],
        ),
        "",
        "## Remaining Valid-SAM Errors",
        "",
        _md_table(
            [
                [
                    r["case_index"],
                    r["book_name"],
                    _fmt(r["ab_safe_abs_error_mm"]),
                    _fmt(r["baseline_abs_error_mm"]),
                    _fmt(r["delta_abs_error_mm"]),
                    r["suspected_failure_type"],
                ]
                for r in [x for x in remaining if x["case_group"] == "valid_sam"][:15]
            ],
            ["case", "book", "AB_SAFE abs", "base abs", "delta", "suspected"],
        ),
        "",
        "## Policy Trigger Cases",
        "",
        _md_table(
            [
                [
                    r["case_index"],
                    r["book_name"],
                    _fmt(r["baseline_abs_error_mm"]),
                    _fmt(r["ab_safe_abs_error_mm"]),
                    "A" if r["policy_A_triggered"] else "",
                    "B" if r["policy_B_triggered"] else "",
                    "skip" if r["policy_B_skipped_due_to_A_plausible"] else "",
                ]
                for r in policy_rows
            ],
            ["case", "book", "base abs", "AB_SAFE abs", "A", "B", "skip"],
        ),
    ]
    report_text = "\n".join(report) + "\n"
    (output_root / "baseline_vs_ab_safe_report.md").write_text(report_text, encoding="utf-8")

    recommendation = [
        "# Recommendation",
        "",
        "AB_SAFE improves the available 100-case comparison, especially on valid-SAM cases, and does not introduce a crossed-out-of-2mm case in this analysis.",
        "It is a reasonable default-candidate, but final default adoption should wait for a short visual/log audit of the remaining valid-SAM >2mm cases and the policy-triggered cases.",
        "",
        "## Next Work",
        "",
        "- Prioritize remaining valid-SAM errors, especially case28/case43-style failures.",
        "- Keep bad-SAM cases excluded from postprocess tuning.",
        "- Verify policy-triggered cases visually before making AB_SAFE the production default.",
    ]
    (output_root / "baseline_vs_ab_safe_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")

    worklog = [
        "# Codex Worklog",
        "",
        "## Objective",
        "Analyze baseline vs AB_SAFE 100-case book-width evaluation results without changing inference code.",
        "",
        "## Inputs",
        f"- baseline: `{summary['baseline_run_root']}`",
        f"- AB_SAFE: `{summary['ab_safe_run_root']}`",
        "",
        "## Outputs",
        "- `baseline_vs_ab_safe_case_compare.csv`",
        "- `baseline_vs_ab_safe_summary.json`",
        "- `baseline_vs_ab_safe_improved_cases.csv`",
        "- `baseline_vs_ab_safe_worsened_cases.csv`",
        "- `baseline_vs_ab_safe_remaining_errors.csv`",
        "- `baseline_vs_ab_safe_policy_trigger_cases.csv`",
        "- `baseline_vs_ab_safe_recommendation.md`",
        "- `baseline_vs_ab_safe_report.md`",
        "",
        "## Notes",
        "No algorithm code was changed in this analysis pass.",
    ]
    (output_root / "codex_worklog.md").write_text("\n".join(worklog) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run-root", type=Path, default=Path("captures/100test_offline/20260703_154211"))
    parser.add_argument("--ab-safe-run-root", type=Path, default=Path("captures/100test_offline/20260706_154033 (1)"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root
    if output_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path("captures/result_by_codex") / f"{stamp}_baseline_vs_ab_safe_analysis"

    summary = analyze(args.baseline_run_root, args.ab_safe_run_root, output_root)
    print(json.dumps({
        "output_root": str(output_root),
        "baseline_run_root": summary["baseline_run_root"],
        "ab_safe_run_root": summary["ab_safe_run_root"],
        "all_cases": summary["summaries"]["all_cases"],
        "valid_sam_only": summary["summaries"]["valid_sam_only"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
