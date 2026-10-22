#!/usr/bin/env python3
"""一時調査用(2026-10-14)。デコーダのquery間self-attention(tgt_mask=None、
全クエリが互い に影響し合う設計)が、特定クエリの確信度抑制の直接的な
原因かどうかを検証する。

手法: TransformerDecoder.forwardをmonkeypatchし、対象クエリ1個だけ
self-attentionで「自分自身以外は見えない(他クエリからの影響を一切
受けない)」状態に強制する。画像・エンコーダ特徴は一切変えない
(cropはしない)。これでスコアが回復するなら、原因はデコーダの
self-attentionそのものにある(encoder側の特徴混合ではない)と言える。

対象: 1280x720 shot3のSHOURYU2(SS2-40-070H)隙間候補(スコア0.0435、
tmp_investigate_below_floor.pyで発見)。本番コードは変更しない。
"""
from __future__ import annotations

import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_DIR.parents[2]))
sys.path.insert(0, str(RUNTIME_DIR / "sam3_source"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from detection.pro_handbook.sam3_runtime.service.model_loader import load_model  # noqa: E402

RGB_PATH = (
    RUNTIME_DIR.parents[2]
    / "captures/resolution_test/4/ocr_test_reco_result_1280x720_20261007_173131/work/3/after_init_rgb.png"
)

# 隙間(x=749-754)に最も近い候補。tmp_investigate_below_floor.pyで確認済み。
GAP_CX_MIN, GAP_CX_MAX = 650, 850


def find_gap_candidate_idx(adapter) -> tuple[int, float]:
    instances = adapter.predict(str(RGB_PATH), prompt="book spine", score_threshold=0.0, min_area=0)
    boxes = instances.bboxes.astype(np.float32)
    scores = instances.scores
    candidates = []
    for i in range(len(scores)):
        x0, y0, w, h = boxes[i]
        cx = x0 + w / 2
        if GAP_CX_MIN <= cx <= GAP_CX_MAX and scores[i] < 0.3:
            candidates.append((i, float(scores[i])))
    if not candidates:
        raise RuntimeError("隙間候補が見つかりませんでした(再現せず)")
    candidates.sort(key=lambda c: -c[1])
    # もっともそれらしい(隙間に近く、スコアが最大の)候補を選ぶ
    idx, score = candidates[0]
    print(f"対象クエリ: idx={idx} baseline_score={score:.4f} (隙間候補 {len(candidates)}件中)")
    return idx, score


def make_isolating_mask(seq_len: int, target_query_idx: int, device, dtype) -> torch.Tensor:
    """target_query_idx(0-origin、presence_token抜きの実クエリ番号)が
    自分自身とpresence_token以外からの影響を一切受けないマスクを作る。
    他の行(他クエリ)は一切変更しない(=tgt_mask=Noneと同じ全結合)。"""
    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    row = target_query_idx + 1  # 先頭にpresence_tokenが1個挿入されているため+1
    mask[row, :] = float("-inf")
    mask[row, 0] = 0.0    # presence_tokenは見えるままにする
    mask[row, row] = 0.0  # 自分自身は見える
    return mask


def run_with_patch(adapter, target_query_idx: int | None) -> float:
    decoder = adapter.processor.model.transformer.decoder
    original_forward = decoder.forward.__func__  # unbound

    def patched_forward(self, *args, **kwargs):
        if target_query_idx is not None:
            tgt = kwargs.get("tgt", args[0] if args else None)
            num_queries = tgt.shape[0]
            seq_len = num_queries + 1  # +presence_token
            custom_mask = make_isolating_mask(
                seq_len, target_query_idx, device=tgt.device, dtype=tgt.dtype
            )
            kwargs["tgt_mask"] = custom_mask
        return original_forward(self, *args, **kwargs)

    decoder.forward = patched_forward.__get__(decoder, type(decoder))
    try:
        instances = adapter.predict(str(RGB_PATH), prompt="book spine", score_threshold=0.0, min_area=0)
    finally:
        decoder.forward = original_forward.__get__(decoder, type(decoder))

    scores = instances.scores
    if target_query_idx is not None and target_query_idx >= len(scores):
        raise RuntimeError(f"idx={target_query_idx} が候補数{len(scores)}を超えています")
    return float(scores[target_query_idx]) if target_query_idx is not None else -1.0


def main() -> None:
    print("モデル読み込み中...")
    loaded = load_model()
    adapter = loaded.adapter
    print(f"読み込み完了 ({loaded.model_load_seconds:.1f}秒)")
    adapter.processor.confidence_threshold = 0.0

    print("\n=== 1. ベースライン(patch無し、tgt_mask=None) ===")
    target_idx, baseline_score = find_gap_candidate_idx(adapter)

    print("\n=== 2. 対象クエリだけself-attentionから孤立させた場合 ===")
    isolated_score = run_with_patch(adapter, target_idx)
    print(f"  isolated_score = {isolated_score:.4f}  (baseline={baseline_score:.4f})")

    print("\n=== 3. 対照実験: 何もpatchしない場合に同じコードパスで再取得(再現性確認) ===")
    rerun_score = run_with_patch(adapter, None)
    _ = rerun_score
    instances2 = adapter.predict(str(RGB_PATH), prompt="book spine", score_threshold=0.0, min_area=0)
    control_score = float(instances2.scores[target_idx])
    print(f"  control_score(無patch再実行) = {control_score:.4f}")

    print("\n=== 結果まとめ ===")
    print(f"  baseline(無patch)      : {baseline_score:.4f}")
    print(f"  control(無patch再実行) : {control_score:.4f}")
    print(f"  isolated(self-attn孤立): {isolated_score:.4f}")
    print(f"  (参考)完全crop切り出し実験での値: 0.1309 -> 0.99")


if __name__ == "__main__":
    main()
