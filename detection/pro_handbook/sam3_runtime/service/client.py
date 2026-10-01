from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class Sam3ServiceError(RuntimeError): pass


class Sam3BatchInfer:
    """Compatibility facade for the legacy `infer_masks(PIL.Image, ...)` call."""
    def __init__(self, endpoint=None, timeout=None, prompt="book spine"):
        self.endpoint = endpoint or os.getenv("SAM3_ENDPOINT", "http://127.0.0.1:8765")
        self.timeout = float(timeout or os.getenv("SAM3_TIMEOUT", "120")); self.prompt = prompt
        self.last_metadata = None

    def infer_masks(self, image, **kwargs):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        request_id = str(uuid.uuid4())
        base = "/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"
        with tempfile.TemporaryDirectory(prefix="sam3-", dir=base) as tmp:
            inp, out = os.path.join(tmp, "input.npy"), os.path.join(tmp, "output.npz")
            np.save(inp, rgb, allow_pickle=False)
            body = json.dumps({"input_npy": inp, "output_npz": out, "prompt": self.prompt}).encode()
            req = urllib.request.Request(self.endpoint + "/infer", data=body, headers={"Content-Type": "application/json", "X-Request-ID": request_id})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response: meta = json.load(response)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise Sam3ServiceError(f"SAM3 service unavailable at {self.endpoint} (request {request_id}): {exc}") from exc
            data = np.load(out, allow_pickle=False)
            masks, boxes, scores = data["masks"].astype(bool), data["boxes"], data["scores"]
        sam_data = [{"name": f"book{i+1}", "score": float(scores[i]), "box": {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3])}} for i, b in enumerate(boxes)]
        self.last_metadata = meta
        stage_save = kwargs.get("stage_save")
        output_dir = getattr(stage_save, "out_dir", None)
        if output_dir is not None:
            output_dir = os.fspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(output_dir, "sam3_service_masks.npz"),
                masks=masks,
                boxes=boxes.astype(np.float32, copy=False),
                scores=scores.astype(np.float32, copy=False),
            )
            overlay = rgb.copy()
            colors = np.asarray(
                [[255, 64, 64], [64, 255, 64], [64, 128, 255], [255, 192, 64]],
                dtype=np.float32,
            )
            for index, mask in enumerate(masks):
                color = colors[index % len(colors)]
                overlay[mask] = (0.58 * overlay[mask] + 0.42 * color).astype(np.uint8)
            overlay_img = Image.fromarray(overlay, mode="RGB")
            # 2026-09-30追加(ユーザー要望): 足切りされた認識がどのマスクに対応するはずだったか
            # multikey_match_debug.jsonのmask_{index+1}表記と照合できるよう、各マスクの重心に
            # 番号を書き込む(masks[index]がmask_{index+1}に対応。命名規則はmultikey_matcher.py側)。
            draw = ImageDraw.Draw(overlay_img)
            try:
                font = ImageFont.truetype(_LABEL_FONT_PATH, 22)
            except OSError:
                font = ImageFont.load_default()
            for index, mask in enumerate(masks):
                ys, xs = np.nonzero(mask)
                if len(xs) == 0:
                    continue
                cx, cy = int(xs.mean()), int(ys.mean())
                label = str(index + 1)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        draw.text((cx + dx, cy + dy), label, fill=(0, 0, 0), font=font, anchor="mm")
                draw.text((cx, cy), label, fill=(255, 255, 0), font=font, anchor="mm")
            overlay_img.save(os.path.join(output_dir, "sam3_all_masks_overlay.png"))
            with open(
                os.path.join(output_dir, "sam3_service_inference.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "prompt": self.prompt,
                        "raw_mask_count": meta.get("raw_count"),
                        "nms_mask_count": meta.get("nms_count", len(masks)),
                        "service_metadata": meta,
                        "instances": sam_data,
                        "mask_areas_px": [int(mask.sum()) for mask in masks],
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        return list(masks), sam_data
