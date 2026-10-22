#!/usr/bin/env python3
"""選択済みマスクに対する「削除だけ」の後処理(飛び地除去)。

2026-10-22: 先輩の書籍出納システム(bookp0650-cpu/book)の
`book_mask_postprocess.py`から、本番で有効化されている`remove_islands`
メソッドだけを移植した簡略版。元のGT比較付きの`morphology`/
`ocr_axis_ransac`/`ocr_axis_smooth`は使っていないため含めていない
(必要になったら元リポジトリから改めて移植する)。

GTマスクは一切知らない。既存のOCR/SAMマッチャーが選択したマスクを受け取り、
ピクセルを削除するだけ。安全チェックに引っかかった場合は必ず元のマスクを返す。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from catheter.scripts.multikey_matcher import FORCED_ANGLE, _unrotate_poly

METHODS = ("baseline", "remove_islands")

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "method": "baseline",
    "min_area_retention": 0.80,
}


@dataclass(frozen=True)
class PostprocessResult:
    mask: np.ndarray
    info: dict[str, Any]


def normalize_postprocess_config(value: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if value:
        cfg.update(value)
    cfg["enabled"] = bool(cfg["enabled"])
    cfg["method"] = str(cfg["method"])
    if cfg["method"] not in METHODS:
        raise ValueError(f"unknown postprocess method: {cfg['method']!r}")
    cfg["min_area_retention"] = float(cfg["min_area_retention"])
    if not 0.0 < cfg["min_area_retention"] <= 1.0:
        raise ValueError("postprocess.min_area_retention must be in (0, 1]")
    return cfg


def load_linked_ocr_polygons(
    shot_dir: str | Path,
    selected_mask_index: int,
    image_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    """multikey_matcher.pyが書き出すmultikey_match_debug.jsonから、既存の
    マッチャーが選択マスクに割り当てたOCRポリゴンだけを読み込む。

    書籍側(load_linked_ocr_polygons)との違い: 書籍側は
    `text_mask_iou_debug.json`の`assignments`/`per_angle_assignments`
    (複数角度OCR対応)を読むが、カテーテル側は`multikey_match_debug.json`の
    `ocr_assignments`(`{"ocr_index","text","matched","overlap_ratio"}`、
    単一角度のみ)という別スキーマ。カテーテル側は現状3角度OCRに対応していない
    ため、角度分岐は持たない(3角度OCR対応時に改めて拡張する)。
    """
    shot_dir = Path(shot_dir)
    debug_path = shot_dir / "multikey_match_debug.json"
    ocr_path = shot_dir / "ocr_result.json"
    if not debug_path.is_file() or not ocr_path.is_file():
        return []

    debug = json.loads(debug_path.read_text(encoding="utf-8"))
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    target_name = f"mask_{int(selected_mask_index)}"
    height, width = int(image_shape[0]), int(image_shape[1])
    polygons = ocr.get("dt_polys") or []
    texts = ocr.get("rec_texts") or []

    result: list[dict[str, Any]] = []
    for assignment in debug.get("ocr_assignments") or []:
        if assignment.get("matched") != target_name:
            continue
        try:
            index = int(assignment["ocr_index"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if index < 0 or index >= len(polygons):
            continue
        try:
            polygon = _unrotate_poly(polygons[index], FORCED_ANGLE, w=width, h=height)
        except Exception:
            continue
        if polygon.shape[0] < 3 or not np.all(np.isfinite(polygon)):
            continue
        result.append(
            {
                "polygon": polygon,
                "center": polygon.mean(axis=0),
                "text": str(texts[index]) if index < len(texts) else str(
                    assignment.get("text", "")
                ),
                "ocr_index": index + 1,
            }
        )
    return result


def _center_survives(mask: np.ndarray, center: np.ndarray) -> bool:
    x = int(round(float(center[0])))
    y = int(round(float(center[1])))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x])


def _remove_islands(
    original: np.ndarray, linked_ocr: Iterable[dict[str, Any]]
) -> tuple[np.ndarray, dict[str, Any]]:
    """本体から切り離された連結成分だけを削除する(GTの形状補正ではない)。

    マスクを8連結で分割し、保存済みOCRの紐づけ情報から「本体」の連結成分を
    1つ選び、それ以外を全部削除する。本体成分の画素はそのままコピーする
    ので、本体に繋がった突起・くぼみ・輪郭のギザギザは変更しない。
    """
    binary = np.asarray(original, dtype=bool).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    component_ids = list(range(1, int(component_count)))
    if not component_ids:
        return original.copy(), {
            "component_count": 0,
            "main_component_label": None,
            "main_component_source": "none",
            "removed_component_count": 0,
            "removed_pixel_count": 0,
        }

    linked = list(linked_ocr)
    overlap_by_label = {label: 0 for label in component_ids}
    center_count_by_label = {label: 0 for label in component_ids}
    height, width = original.shape

    for item in linked:
        try:
            polygon = np.asarray(item["polygon"], dtype=np.float64).reshape(-1, 2)
        except Exception:
            continue
        if polygon.shape[0] < 3 or not np.all(np.isfinite(polygon)):
            continue
        points = np.round(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        polygon_mask = np.zeros_like(binary)
        cv2.fillPoly(polygon_mask, [points.reshape(-1, 1, 2)], 1)
        intersecting_labels, counts = np.unique(
            labels[polygon_mask > 0], return_counts=True
        )
        for label, count in zip(intersecting_labels, counts):
            label = int(label)
            if label in overlap_by_label:
                overlap_by_label[label] += int(count)

        center = item.get("center")
        if center is not None:
            try:
                x = int(round(float(center[0])))
                y = int(round(float(center[1])))
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= x < width and 0 <= y < height:
                label = int(labels[y, x])
                if label in center_count_by_label:
                    center_count_by_label[label] += 1

    areas = {label: int(stats[label, cv2.CC_STAT_AREA]) for label in component_ids}
    maximum_overlap = max(overlap_by_label.values(), default=0)
    if maximum_overlap > 0:
        main_label = max(
            component_ids,
            key=lambda label: (overlap_by_label[label], areas[label], -label),
        )
        source = "maximum_ocr_overlap"
    else:
        maximum_center_count = max(center_count_by_label.values(), default=0)
        if maximum_center_count > 0:
            main_label = max(
                component_ids,
                key=lambda label: (
                    center_count_by_label[label], areas[label], -label,
                ),
            )
            source = "ocr_center_component"
        else:
            main_label = max(component_ids, key=lambda label: (areas[label], -label))
            source = "largest_area_component"

    cleaned = labels == int(main_label)
    removed_pixel_count = int(original.sum()) - int(cleaned.sum())
    return cleaned, {
        "component_count": len(component_ids),
        "main_component_label": int(main_label),
        "main_component_source": source,
        "removed_component_count": int(len(component_ids) - 1),
        "removed_pixel_count": int(removed_pixel_count),
    }


def _fallback(
    original: np.ndarray, info: dict[str, Any], reason: str, started: float
) -> PostprocessResult:
    info.update(
        {
            "fallback": True,
            "fallback_reason": str(reason),
            "mask_area_after": int(original.sum()),
            "area_ratio": 1.0,
            "processing_time_ms": (time.perf_counter() - started) * 1000.0,
        }
    )
    return PostprocessResult(original.copy(), info)


def postprocess_selected_mask(
    mask: np.ndarray,
    linked_ocr: Iterable[dict[str, Any]],
    config: dict[str, Any] | None,
) -> PostprocessResult:
    """飛び地除去だけを行う(削除専用)。既定は無効(enabled=False)。"""
    started = time.perf_counter()
    original = np.asarray(mask, dtype=bool)
    cfg = normalize_postprocess_config(config)
    linked = list(linked_ocr)
    area_before = int(original.sum())
    info: dict[str, Any] = {
        "method": cfg["method"],
        "params": cfg,
        "mask_area_before": area_before,
        "mask_area_after": area_before,
        "area_ratio": 1.0,
        "num_linked_ocr": len(linked),
        "component_count": None,
        "removed_component_count": 0,
        "removed_pixel_count": 0,
        "fallback": False,
        "fallback_reason": "",
    }

    if area_before <= 0:
        return _fallback(original, info, "empty_original_mask", started)
    if not cfg["enabled"] or cfg["method"] == "baseline":
        info["processing_time_ms"] = (time.perf_counter() - started) * 1000.0
        return PostprocessResult(original.copy(), info)

    try:
        candidate, component_info = _remove_islands(original, linked)
        candidate = np.asarray(candidate, dtype=bool) & original
        # remove_islandsは成分の掃除であってGT形状補正ではない。main_labelの
        # 画素はそのままコピーするので、面積・幅の保持率チェック等の汎用安全
        # ガードは意図的に通さない(書籍側の設計をそのまま踏襲)。
        info.update(component_info)
        area_after = int(candidate.sum())
        info.update(
            {
                "mask_area_after": area_after,
                "area_ratio": float(area_after / max(area_before, 1)),
                "processing_time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return PostprocessResult(candidate, info)
    except Exception as exc:
        return _fallback(
            original, info, f"processing_error:{type(exc).__name__}:{exc}", started
        )
