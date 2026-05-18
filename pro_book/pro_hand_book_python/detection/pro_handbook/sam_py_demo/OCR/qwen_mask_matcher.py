from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import time

import cv2
import numpy as np
from PIL import Image
import torch
from rapidfuzz import fuzz
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


class QwenMaskMatcher:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=False,   # 読みが安定するか確認用
        )

    def _crop_from_mask(
        self,
        rgb_bgr: np.ndarray,
        mask01: np.ndarray,
        pad: int = 12,
        bg_color: int = 255,
        scale: int = 2,
    ) -> Image.Image | None:
        """
        mask領域を外接矩形で切り出し、
        マスク外を白背景にした上で拡大して PIL.Image を返す。
        """
        ys, xs = np.where(mask01 > 0)
        if xs.size == 0:
            return None

        x1 = max(0, int(xs.min()) - pad)
        y1 = max(0, int(ys.min()) - pad)
        x2 = min(rgb_bgr.shape[1], int(xs.max()) + 1 + pad)
        y2 = min(rgb_bgr.shape[0], int(ys.max()) + 1 + pad)

        crop = rgb_bgr[y1:y2, x1:x2].copy()
        crop_mask = mask01[y1:y2, x1:x2]

        # 白背景にする
        white_bg = np.full_like(crop, bg_color)
        white_bg[crop_mask > 0] = crop[crop_mask > 0]
        crop = white_bg

        # 拡大（文字を読みやすくする）
        if scale > 1:
            crop = cv2.resize(
                crop,
                (crop.shape[1] * scale, crop.shape[0] * scale),
                interpolation=cv2.INTER_CUBIC,
            )

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)

    def _save_input_variants(
        self,
        crop_pil: Image.Image,
        out_dir: Path,
        mask_name: str,
    ) -> dict[str, Image.Image]:
        """
        Qwenに入力する画像を保存しつつ、向き違いを返す。
        今回は orig が上下逆なので、180度回転した画像だけ使う。
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        variants = {
            "rot180": crop_pil.rotate(180, expand=True),
        }

        for tag, img in variants.items():
            img.save(out_dir / f"{mask_name}_{tag}.png")

        return variants

    def _read_text_from_image(
        self,
        image: Image.Image,
        max_new_tokens: int = 96,
    ) -> str:
        start = time.time()

        prompt = (
            "You are an OCR system for Japanese book spines.\n"
            "Read all visible printed Japanese text exactly as written.\n"
            "This may be vertical Japanese text on a book spine.\n"
            "Return plain text only.\n"
            "Do not explain.\n"
            "Do not add extra words.\n"
            "If unreadable, return an empty string."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.model.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        output_text = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        elapsed = time.time() - start
        print(f"[QWEN] inference time: {elapsed:.2f} sec")

        return output_text.strip()

    def match_query_to_masks(
        self,
        query: str,
        rgb_bgr: np.ndarray,
        masks: list[np.ndarray],
        shot_dir: str | Path,
        threshold: int = 40,
    ) -> list[dict[str, Any]]:
        """
        各 mask をクロップし、Qwen で文字認識。
        向き違いも試し、query と最も近い結果を各 mask に対して採用する。

        追加:
        - OCR全体の合計時間を表示
        - 有効な認識結果（textが空でない）のみで平均推論時間を表示
        """
        shot_dir = Path(shot_dir)
        qwen_dir = shot_dir / "qwen"
        qwen_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        debug_rows: list[dict[str, Any]] = []

        # ===== 時間計測用 =====
        total_ocr_time_valid = 0.0   # 空文字でないOCR結果だけ加算
        valid_ocr_count = 0          # 空文字でないOCR結果の件数

        for i, m in enumerate(masks, start=1):
            mask_name = f"mask_{i}"
            mask01 = (np.asarray(m) > 0).astype(np.uint8)

            crop_pil = self._crop_from_mask(rgb_bgr, mask01)
            if crop_pil is None:
                results.append({
                    "name": mask_name,
                    "score": 0,
                    "text": "",
                    "orientation": "none",
                })
                continue

            variants = self._save_input_variants(crop_pil, qwen_dir, mask_name)

            best_text = ""
            best_score = -1
            best_orientation = "orig"

            candidate_logs = []

            # 1 mask あたりの時間（有効OCRのみ）
            mask_valid_time = 0.0
            mask_valid_count = 0

            for orientation, img in variants.items():
                t0 = time.time()

                try:
                    recognized_text = self._read_text_from_image(img)
                except Exception as e:
                    recognized_text = ""
                    elapsed = time.time() - t0

                    candidate_logs.append({
                        "orientation": orientation,
                        "text": "",
                        "score": 0,
                        "time_sec": elapsed,
                        "error": str(e),
                    })
                    continue

                elapsed = time.time() - t0
                score = int(fuzz.partial_ratio(query, recognized_text))

                candidate_logs.append({
                    "orientation": orientation,
                    "text": recognized_text,
                    "score": score,
                    "time_sec": elapsed,
                })

                # 空文字・改行だけの結果は除外
                if recognized_text.strip():
                    total_ocr_time_valid += elapsed
                    valid_ocr_count += 1
                    mask_valid_time += elapsed
                    mask_valid_count += 1

                if score > best_score:
                    best_score = score
                    best_text = recognized_text
                    best_orientation = orientation

            result = {
                "name": mask_name,
                "score": best_score,
                "text": best_text,
                "orientation": best_orientation,
            }
            results.append(result)

            debug_rows.append({
                "mask_name": mask_name,
                "best": result,
                "candidates": candidate_logs,
                "valid_ocr_time_sec": mask_valid_time,
                "valid_ocr_count": mask_valid_count,
            })

            print(
                f"[QWEN] {mask_name}: "
                f"best_score={best_score}, "
                f"orientation={best_orientation}, "
                f"text={best_text!r}, "
                f"valid_time={mask_valid_time:.2f}s, "
                f"valid_count={mask_valid_count}"
            )

        results.sort(key=lambda x: x["score"], reverse=True)

        out_path = qwen_dir / "qwen_ocr_results.json"
        out_path.write_text(
            json.dumps(
                {
                    "query": query,
                    "threshold": threshold,
                    "results": debug_rows,
                    "time_summary": {
                        "total_ocr_time_valid_sec": total_ocr_time_valid,
                        "valid_ocr_count": valid_ocr_count,
                        "average_time_per_valid_ocr_sec": (
                            total_ocr_time_valid / valid_ocr_count
                            if valid_ocr_count > 0 else 0.0
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        avg_time = total_ocr_time_valid / valid_ocr_count if valid_ocr_count > 0 else 0.0

        print("\n===== QWEN OCR TIME SUMMARY =====")
        print(f"Total OCR time (valid only): {total_ocr_time_valid:.2f} sec")
        print(f"Valid OCR count           : {valid_ocr_count}")
        print(f"Average time per valid OCR: {avg_time:.2f} sec")
        print("=================================\n")

        return results