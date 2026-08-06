#!/usr/bin/env python3
"""Persistent OCR worker for batch offline evaluation.

OCR/paddle_ocr_test.py already caches its PaddleOCR model at module level
(OCR_main_cached), but get_book_points.start_ocr_subprocess() launches a
brand new subprocess for every single call, so that cache never survives
past one shot_dir (~2 sec model reconstruction every time, on top of the
python-process/import startup cost).

This session keeps one `paddle_ocr_test.py --worker` process alive and
feeds it shot_dir paths one at a time over stdin/stdout, so the model is
built once and reused for every case in a batch run.

Not wired into the live/production recognition path
(get_book_points.start_ocr_subprocess / wait_ocr_subprocess), which is
left untouched: this is opt-in, for offline batch evaluators only.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path

from . import get_book_points as current


class OcrWorkerSession:
    """Launches and talks to one resident OCR worker subprocess.

    stdout is drained by a dedicated background thread into a Queue.
    select() on a buffered text-mode pipe was tried first and is unsafe:
    Python's io layer can pull a whole burst of already-printed lines into
    its internal buffer in one OS-level read, after which select() reports
    "no data ready" even though readline() would return instantly - the
    wait() call then times out despite the answer having arrived. A
    reader thread + Queue sidesteps that entirely.
    """

    def __init__(self) -> None:
        ocr_py, ocr_script = current._resolve_ocr_subprocess_paths()
        env = current._build_offline_ocr_env()
        self._proc = subprocess.Popen(
            [ocr_py, ocr_script, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._out_queue: "queue.Queue[str | None]" = queue.Queue()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._out_queue.put(line.rstrip("\n"))
        # stdout closed (worker exited): unblock any pending wait() with EOF.
        self._out_queue.put(None)

    @property
    def pid(self) -> int:
        return self._proc.pid

    def start(self, shot_dir) -> str:
        """Send a shot_dir to the resident worker without blocking for the result.

        Mirrors get_book_points.start_ocr_subprocess()'s async shape so OCR
        keeps running concurrently with SAM3 mask generation; call wait()
        with the returned shot_dir_str to block for completion.
        """
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"OCR worker process already exited (returncode={self._proc.returncode})"
            )
        assert self._proc.stdin is not None
        shot_dir_str = str(Path(shot_dir).expanduser().resolve())
        self._proc.stdin.write(shot_dir_str + "\n")
        self._proc.stdin.flush()
        return shot_dir_str

    def wait(self, shot_dir_str: str, *, timeout: float = 120.0) -> None:
        """Block until the worker reports completion for shot_dir_str."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"OCR worker timeout while processing {shot_dir_str}")
            try:
                line = self._out_queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"OCR worker timeout while processing {shot_dir_str}")
            if line is None:
                raise RuntimeError(
                    f"OCR worker process exited unexpectedly while processing {shot_dir_str}"
                )
            status = _try_parse_status_line(line)
            if status is None:
                # worker startup/library log line (paddle/paddleocr prints,
                # [OCR CACHE] progress messages, ...) - forward and keep reading
                print(line, flush=True)
                continue
            if status.get("shot_dir") != shot_dir_str:
                print(line, flush=True)
                continue
            if not status.get("ok"):
                raise RuntimeError(f"OCR worker failed for {shot_dir_str}: {status.get('error')}")
            return

    def run(self, shot_dir, *, timeout: float = 120.0) -> None:
        """Convenience: start() + wait() for callers that don't need overlap."""
        shot_dir_str = self.start(shot_dir)
        self.wait(shot_dir_str, timeout=timeout)

    def stop(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=10.0)
        except Exception:
            self._proc.kill()
            self._proc.wait()
        self._reader_thread.join(timeout=5.0)

    def __enter__(self) -> "OcrWorkerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _try_parse_status_line(line: str):
    try:
        obj = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict) and "ok" in obj and "shot_dir" in obj:
        return obj
    return None
