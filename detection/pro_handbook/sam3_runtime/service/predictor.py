from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from .settings import (
    MIN_AREA,
    NMS_IOU_THRESHOLD,
    PROCESSOR_CONFIDENCE_THRESHOLD,
    PROMPT,
    SCORE_THRESHOLD,
)


def infer_array(loaded, image: np.ndarray, prompt: str = PROMPT):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected RGB uint8 HxWx3 array, got {image.shape} {image.dtype}")
    baseline_mb = 0.0
    if loaded.device == "cuda":
        torch.cuda.synchronize()
        # 画像ごとの推論ピークを測るため、推論の直前にカウンタを戻す。
        # max_memory_allocated() はプロセス開始からの累積最大なので、リセットしないと
        # モデル読み込み時のピークで張り付き、解像度ごとの比較ができない
        # (2026-10-15、解像度比較試験で 1280 の方が 1920 より大きい値が出たため追加)。
        # baseline_mb は常駐分(モデルの重み等)で、gpu_memory_mb との差が推論の増分になる。
        baseline_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # Sam3Adapter accepts ndarray as OpenCV BGR, while the service contract is
    # RGB. Passing RGB as PIL-equivalent is achieved by reversing to BGR here;
    # the adapter then performs its canonical BGR->RGB conversion exactly once.
    raw_instances = loaded.adapter.predict(
        np.ascontiguousarray(image[:, :, ::-1]),
        prompt=prompt,
        score_threshold=SCORE_THRESHOLD,
        min_area=MIN_AREA,
    )
    from core.mask_nms import apply_mask_nms
    instances = apply_mask_nms(
        raw_instances, iou_thresh=NMS_IOU_THRESHOLD, metric="iou", mode="suppress"
    ).instances
    if loaded.device == "cuda":
        torch.cuda.synchronize()
    masks = instances.masks.astype(bool, copy=False)
    scores = instances.scores.astype(np.float32, copy=False)
    xywh = instances.bboxes.astype(np.float32, copy=False)
    boxes = xywh.copy()
    if len(boxes):
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - 1
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - 1
    gpu_mb = 0.0
    if loaded.device == "cuda":
        gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    meta = {
        "count": int(len(masks)),
        "raw_count": int(raw_instances.count),
        "nms_count": int(instances.count),
        "inference_seconds": time.perf_counter() - started,
        "gpu_memory_mb": gpu_mb,
        "gpu_baseline_mb": baseline_mb,
        "score_threshold": SCORE_THRESHOLD,
        "processor_confidence_threshold": PROCESSOR_CONFIDENCE_THRESHOLD,
        "min_area": MIN_AREA,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
    }
    return masks, boxes, scores, meta


def infer_to_npz(loaded, input_path: Path, output_path: Path, prompt: str = PROMPT) -> dict:
    image = np.load(input_path, allow_pickle=False)
    masks, boxes, scores, meta = infer_array(loaded, image, prompt)
    np.savez_compressed(output_path, masks=masks, boxes=boxes, scores=scores)
    return meta
