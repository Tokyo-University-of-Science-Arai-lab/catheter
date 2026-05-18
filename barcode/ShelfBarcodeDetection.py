#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import cv2
import numpy as np
from pyzbar.pyzbar import decode

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String, Bool


def preprocess_for_barcode(gray: "cv2.Mat") -> "cv2.Mat":
    """バーコード検出前の前処理."""
    contrast_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    blur = cv2.GaussianBlur(contrast_img, (3, 3), 0)
    sharpened = cv2.addWeighted(
        blur, 1.5,
        cv2.GaussianBlur(gray, (0, 0), 3), -0.5,
        0
    )
    _, binary = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
    return binary


def find_barcode(
    gray: "cv2.Mat",
    target_code: Optional[str] = None
) -> Optional[Tuple[str, int, int, int, int]]:
    """
    グレースケール画像からバーコードを1つ検出して返す。

    Returns:
        (code, left, top, width, height) or None
    """
    binary = preprocess_for_barcode(gray)
    decoded = decode(binary)
    if not decoded:
        return None

    # 特定のコードを優先して探したい場合
    if target_code:
        for b in decoded:
            s = b.data.decode("utf-8", errors="ignore")
            if s == str(target_code):
                r = b.rect
                return (s, r.left, r.top, r.width, r.height)

    # とりあえず最初の1つ
    b0 = decoded[0]
    s0 = b0.data.decode("utf-8", errors="ignore")
    r0 = b0.rect
    return (s0, r0.left, r0.top, r0.width, r0.height)


class BarcodeOffsetTrigger(Node):
    """
    /navigation_goal (Bool) の立ち上がりエッジで 1 フレームだけ撮影し、
    バーコードと横方向オフセットを publish するノード。

    出力:
        /barcode_offset_u_px (Float32): 画像中心からの横オフセット [pixel]
        /barcode_offset_u_m  (Float32): 同オフセット [m]（assumed_z_m から計算）
        /barcode_code        (String):  読み取ったバーコード文字列
    """

    def __init__(
        self,
        *,
        node_name: str = "barcode_offset_trigger",
        camera_index: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        hfov_deg: Optional[float] = None,
        assumed_z_m: Optional[float] = None,
    ) -> None:
        """
        統合コードからも使いやすいように、引数で上書きもできる形にしておく。

        引数を省略した場合は ROS パラメータの値を使う。
        """
        super().__init__(node_name)

        # ===== パラメータ宣言 =====
        # ここで宣言しておくと、ros2 param や yaml からも上書き可能
        self.declare_parameter("camera_index", 1)
        self.declare_parameter("width",       3840)
        self.declare_parameter("height",      2160)
        self.declare_parameter("hfov_deg",    80.0)
        self.declare_parameter("assumed_z_m", 2.0)

        # 引数優先 / なければパラメータ値を使用
        if camera_index is None:
            camera_index = int(self.get_parameter("camera_index").value)
        if width is None:
            width = int(self.get_parameter("width").value)
        if height is None:
            height = int(self.get_parameter("height").value)
        if hfov_deg is None:
            hfov_deg = float(self.get_parameter("hfov_deg").value)
        if assumed_z_m is None:
            assumed_z_m = float(self.get_parameter("assumed_z_m").value)

        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)
        self.assumed_z_m = float(assumed_z_m)

        # カメラ内部パラメータ
        fov_rad = math.radians(hfov_deg)
        self.fx = (self.width / 2.0) / math.tan(fov_rad / 2.0)
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        # ===== カメラ初期化 =====
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")

        # ===== Publisher =====
        self.pub_offset_px = self.create_publisher(Float32, "/barcode_offset_u_px", 10)
        self.pub_offset_m  = self.create_publisher(Float32, "/barcode_offset_u_m",  10)
        self.pub_code      = self.create_publisher(String,  "/barcode_code",        10)

        # ===== Subscriber =====
        self.prev_goal = False
        self.sub_goal = self.create_subscription(
            Bool,
            "/navigation_goal",
            self._on_navigation_goal,
            10,
        )

        self.get_logger().info(
            f"{node_name} started. "
            f"(cam_index={self.camera_index}, size={self.width}x{self.height}, "
            f"hfov={hfov_deg}deg, Z={self.assumed_z_m}m)"
        )

    def destroy_node(self) -> None:
        # カメラリソースの解放を忘れない
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        super().destroy_node()

    # ============================================================
    # /navigation_goal が True になった瞬間だけ実行
    # ============================================================
    def _on_navigation_goal(self, msg: Bool) -> None:
        curr = bool(msg.data)

        # 立ち上がりエッジ検出（False→True）
        if curr and not self.prev_goal:
            self.get_logger().info("navigation_goal rising edge detected.")
            self._run_once()

        self.prev_goal = curr

    # ============================================================
    # 1回だけ処理を実行
    # ============================================================
    def _run_once(self):

        # ---- 1. フレーム取得 ----
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().error("Failed to capture frame.")
            return

        # デバッグ用にコピーしておく
        debug_frame = frame.copy()

        # 元画像サイズ（4K）
        full_h, full_w = frame.shape[:2]

        # ---- 2. リサイズしてバーコード検出（4Kは重いので縮小）----
        # 例として横幅を 1920 にする
        target_width = 1920
        scale = target_width / full_w
        target_height = int(full_h * scale)

        small = cv2.resize(frame, (target_width, target_height))
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        from pyzbar.pyzbar import decode
        decoded = decode(gray_small)

        if not decoded:
            self.get_logger().warn("Barcode not found.")

            # デバッグ表示（バーコードが見えているか確認用）
            cv2.imshow("barcode_debug", debug_frame)
            cv2.waitKey(1)
            return

        # とりあえず最初のバーコードだけ使う
        b0 = decoded[0]
        code = b0.data.decode("utf-8", errors="ignore")
        r = b0.rect  # small 画像上での矩形

        # ---- 3. small→full座標へスケールバック ----
        # pyzbar の rect は namedtuple(rect.left, rect.top, rect.width, rect.height)
        left_full = int(r.left / scale)
        top_full  = int(r.top  / scale)
        w_full    = int(r.width  / scale)
        h_full    = int(r.height / scale)

        # ---- 4. BB を full フレーム上に描画 ----
        cv2.rectangle(
            debug_frame,
            (left_full, top_full),
            (left_full + w_full, top_full + h_full),
            (0, 255, 0),
            3,
        )
        cv2.putText(
            debug_frame,
            code,
            (left_full, max(0, top_full - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )

        cv2.imshow("barcode_debug", debug_frame)
        cv2.waitKey(1)

        # ---- 5. オフセット計算（full 解像度基準）----
        u_center_full = left_full + (w_full / 2.0)
        du_px = u_center_full - self.cx   # self.cx は full 解像度幅/2 を想定

        Z = self.assumed_z_m
        offset_u_m = (du_px * Z) / self.fx

        # ---- 6. publish ----
        msg_px = Float32()
        msg_px.data = float(du_px)
        self.pub_offset_px.publish(msg_px)

        msg_m = Float32()
        msg_m.data = float(offset_u_m)
        self.pub_offset_m.publish(msg_m)

        msg_code = String()
        msg_code.data = code
        self.pub_code.publish(msg_code)

        self.get_logger().info(
            f"code={code}, du_px={du_px:.1f}, offset_u_m={offset_u_m:.3f}m"
        )


# ------------------------------------------------------------
# 単体テスト用 (python barcode_offset_trigger_node.py で動かしたい場合用)
# 統合コードから使うときはこの main は呼ばない。
# ------------------------------------------------------------
def main():
    rclpy.init()
    node = BarcodeOffsetTrigger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()