import threading
import time
import os


class XArmMonitor:

    def __init__(self, arm, check_period=0.5, auto_stop=True, on_abnormal=None):
        self.arm = arm
        self.check_period = check_period
        self.auto_stop = auto_stop
        self.exception_count = 0
        self.EXCEPTION_THRESHOLD = 5   # 連続5回で異常扱い
        self.abnormal_count = 0
        self.ABNORMAL_THRESHOLD = 3    # state異常3回連続で停止
        self.state = None
        self.err = None
        self.warn = None
        self.on_abnormal = on_abnormal
        self.abnormal = False
        self._lock = threading.Lock()

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ==============================
    # メイン監視ループ
    # ==============================
    def _loop(self):
        while self._running:
            try:
                state = self.arm.get_state()
                err, warn = self.arm.get_err_warn()

                with self._lock:
                    self.state = state
                    self.err = err
                    self.warn = warn

                # ===== 通信成功したら例外カウンタリセット =====
                self.exception_count = 0

                # ===== 状態異常カウント =====
                if state in (4, 5) or err != 0:
                    self.abnormal_count += 1
                    print(f"[MONITOR] abnormal_count={self.abnormal_count}")
                else:
                    self.abnormal_count = 0

                # 連続異常のみ停止
                if self.abnormal_count >= self.ABNORMAL_THRESHOLD:
                    self._handle_abnormal(
                        f"state={state}, err={err}, warn={warn}"
                    )

            except Exception as e:
                self.exception_count += 1
                print(f"[MONITOR] transient exception ({self.exception_count})")

                if self.exception_count >= self.EXCEPTION_THRESHOLD:
                    self._handle_abnormal(f"Monitor exception: {e}")

            time.sleep(self.check_period)

    # ==============================
    # 異常処理
    # ==============================
    def _handle_abnormal(self, msg):
        with self._lock:
            if self.abnormal:
                return
            self.abnormal = True

        print(f"\n[MONITOR] ABNORMAL DETECTED | {msg}")

            # 🔥 ここ追加（最優先）
        if hasattr(self, "on_abnormal") and self.on_abnormal:
            try:
                self.on_abnormal(msg)
            except Exception as e:
                print(f"[MONITOR] log failed: {e}")

        # 即アーム停止
        try:
            self.arm.emergency_stop()
        except Exception:
            pass

        # 少し待って安定
        time.sleep(0.05)

        if self.auto_stop:
            print("[MONITOR] SYSTEM TERMINATED")
            os._exit(1)

    # ==============================
    # 外部用
    # ==============================
    def is_abnormal(self):
        with self._lock:
            return self.abnormal

    def get_status(self):
        with self._lock:
            return self.state, self.err, self.warn

    def stop(self):
        self._running = False
        self._thread.join()

def safe_motion(func, monitor, where=""):

    if monitor.is_abnormal():
        return

    state, err, warn = monitor.get_status()

    # 起動直後対策
    if state is None:
        return

    if state in (4,5) or err != 0:
        print(f"[SAFE_MOTION] Pre-motion abnormal at {where}")
        monitor._handle_abnormal(
            f"state={state}, err={err}, warn={warn}"
        )
        return

    try:
        ret = func()

        #print(f"[SAFE_MOTION DEBUG] ret={ret}, type={type(ret)}")

        # 🔥 Falseは「正常だけど失敗扱い」にする（停止しない）
        if ret is False:
            print(f"[SAFE_MOTION] Non-critical failure at {where}")
            return

        # 🔥 intだけガチエラー
        if isinstance(ret, int) and ret != 0:
            print(f"[SAFE_MOTION] API error at {where}, code={ret}")
            monitor._handle_abnormal(f"API error code={ret}")
            return

        state, err, warn = monitor.get_status()

        if state in (4,5) or err != 0:
            print(f"[SAFE_MOTION] Post-motion abnormal at {where}")
            monitor._handle_abnormal(
                f"state={state}, err={err}, warn={warn}"
            )

    except Exception as e:
        print(f"[SAFE_MOTION] Motion exception at {where}: {e}")
        monitor._handle_abnormal(f"Motion exception: {e}")

