#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path

OCR_DIR = Path(__file__).resolve().parent
if str(OCR_DIR) not in sys.path:
    sys.path.insert(0, str(OCR_DIR))

try:
    import paddle_ocr_test as _ocr_mod
    if hasattr(_ocr_mod, "OCR_main_cached"):
        _OCR_FUNC = _ocr_mod.OCR_main_cached
    else:
        _OCR_FUNC = _ocr_mod.OCR_main
except Exception as e:
    print("__OCR_WORKER_IMPORT_FAILED__" + repr(e), flush=True)
    raise

print("__OCR_WORKER_READY__", flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    if line == "__quit__":
        break

    req = {}
    try:
        req = json.loads(line)
        req_id = int(req.get("id", -1))
        shot_dir = Path(req.get("shot_dir")).expanduser().resolve()
        t0 = time.perf_counter()
        _OCR_FUNC(str(shot_dir))
        elapsed_sec = time.perf_counter() - t0
        payload = {
            "id": req_id,
            "ok": True,
            "shot_dir": str(shot_dir),
            "elapsed_sec": float(elapsed_sec),
        }
    except Exception:
        payload = {
            "id": int(req.get("id", -1)) if isinstance(req, dict) else -1,
            "ok": False,
            "error": traceback.format_exc(),
        }

    print("__OCR_WORKER_DONE__" + json.dumps(payload, ensure_ascii=False), flush=True)
