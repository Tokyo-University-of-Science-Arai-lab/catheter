# Ubuntu/Linux 専用 KBHit
# - 端末を非カノニカル・非エコーに設定
# - select でノンブロッキング入力

import sys
import termios
import atexit
from select import select
import time

class KBHit:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_term = termios.tcgetattr(self.fd)
        self.new_term = termios.tcgetattr(self.fd)
        # 非カノニカル・非エコー
        self.new_term[3] = self.new_term[3] & ~termios.ICANON & ~termios.ECHO
        # ★ 完全ノンブロッキングに
        self.new_term[6][termios.VMIN]  = 0
        self.new_term[6][termios.VTIME] = 0

        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)
        atexit.register(self.set_normal_term)

    def set_normal_term(self):
        try:
            termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)
        except Exception:
            pass

    def kbhit(self) -> bool:
        dr, _, _ = select([sys.stdin], [], [], 0)
        return bool(dr)

    def _read_char(self, timeout: float = 0.02) -> str:
        """最大1文字を timeout 付きで読み取る（ブロックしない）"""
        dr, _, _ = select([sys.stdin], [], [], timeout)
        if not dr:
            return ""
        ch = sys.stdin.read(1)  # 1文字(str)
        return ch if ch else ""

    def getch(self) -> str:
        c1 = self._read_char(0.0)
        if not c1:
            return ""

        if c1 != "\x1b":          # 通常キー
            return c1

        # ESC 以降を丁寧に集める
        deadline = time.time() + 0.1
        seq = "\x1b"
        # まず '[' を待つ（無ければ単独 ESC）
        while time.time() < deadline and len(seq) < 2:
            c = self._read_char(0.01)
            if c:
                seq += c
                break
        if len(seq) < 2 or seq[1] != '[':
            return "\x1b"

        # '[' の後は、最終の A/B/C/D が来るまで読む（修飾付き対応）
        final = ""
        while time.time() < deadline:
            c = self._read_char(0.01)
            if not c:
                continue
            seq += c
            if c in "ABCD":
                final = c
                break

        if final == "A": return "H"  # Up
        if final == "B": return "P"  # Down
        if final == "C": return "M"  # Right
        if final == "D": return "K"  # Left
        return "\x1b"

    def getarrow(self) -> int:
        """
        矢印キーを 0..3 に正規化して返す（非矢印は -1）
          0: up, 1: right, 2: down, 3: left
        """
        ch = self.getch()
        if   ch == "H": return 0  # Up
        elif ch == "M": return 1  # Right
        elif ch == "P": return 2  # Down
        elif ch == "K": return 3  # Left
        return -1

if __name__ == "__main__":
    kb = KBHit()
    print("Press keys (ESC to exit). Arrows normalized to H/M/P/K.")
    try:
        while True:
            if kb.kbhit():
                c = kb.getch()
                if not c:
                    continue
                o = ord(c)
                print(f"key='{c}' ord={o}")
                if o == 27:  # ESC
                    print("ESC -> exit")
                    break
    finally:
        kb.set_normal_term()