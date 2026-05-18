#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple

import cv2
from pyzbar.pyzbar import decode

import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


def preprocess_for_barcode(gray: "cv2.Mat") -> "cv2.Mat":
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
    """戻り値: (code_string, left, top, width, height)"""
    binary = preprocess_for_barcode(gray)
    decoded = decode(binary)
    if not decoded:
        return None

    if target_code:
        for b in decoded:
            s = b.data.decode("utf-8", errors="ignore")
            if s == str(target_code):
                r = b.rect
                return (s, r.left, r.top, r.width, r.height)

    b0 = decoded[0]
    s0 = b0.data.decode("utf-8", errors="ignore")
    r0 = b0.rect
    return (s0, r0.left, r0.top, r0.width, r0.height)


class BarcodeFromRealSenseOnGoal(Node):
    """
    - RealSenseからその場で1フレーム取得（pyrealsense2）
    - /wall_distance (Float32): バーコード無し壁までの距離
    - /navigation_goal (Bool): False→Trueの瞬間に処理実行
    - /error_x (String): 結果を文字列でpublish
    """

    def __init__(self):
        super().__init__("barcode_from_realsense_on_goal")

        # ===== パラメータ =====
        self.declare_parameter("wall_width", 0.800)  # [m] 壁面間距離（実測）
        self.wall_width = float(self.get_parameter("wall_width").value)

        self.declare_parameter("target_barcode", "")  # 空なら最初に見つかったバーコード
        tb = str(self.get_parameter("target_barcode").value).strip()
        self.target_barcode = tb if tb != "" else None

        # RealSense 스트림設定
        self.declare_parameter("rs_width", 1280)
        self.declare_parameter("rs_height", 720)
        self.declare_parameter("rs_fps", 30)
        self.declare_parameter("rs_serial", "")  # 複数台あるなら指定（空なら自動）

        self.rs_width = int(self.get_parameter("rs_width").value)
        self.rs_height = int(self.get_parameter("rs_height").value)
        self.rs_fps = int(self.get_parameter("rs_fps").value)
        self.rs_serial = str(self.get_parameter("rs_serial").value).strip()

        # ROS2トピック
        self.declare_parameter("wall_distance_topic", "/wall_distance")
        self.declare_parameter("navigation_goal_topic", "/navigation_goal")
        self.declare_parameter("error_x_topic", "/error_x")

        wall_topic = str(self.get_parameter("wall_distance_topic").value)
        goal_topic = str(self.get_parameter("navigation_goal_topic").value)
        out_topic = str(self.get_parameter("error_x_topic").value)

        # ===== 状態 =====
        self.latest_wall_distance_m: Optional[float] = None
        self.prev_goal: bool = False
        self.pending_goal: bool = False   # ★追加：wall_distance待ちのときにTrue


        # ===== RealSense初期化 =====
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.rs_serial:
            self.config.enable_device(self.rs_serial)

        self.config.enable_stream(rs.stream.color, self.rs_width, self.rs_height, rs.format.bgr8, self.rs_fps)

        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            self.get_logger().error(f"RealSense pipeline start failed: {e}")
            raise

        # intrinsics（fx,cx）はRealSenseから取得して使う
        color_stream_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream_profile.get_intrinsics()
        self.fx = float(intr.fx)
        self.cx = float(intr.ppx)

        self.get_logger().info(
            f"RealSense started: {self.rs_width}x{self.rs_height}@{self.rs_fps}, "
            f"fx={self.fx:.3f}, cx={self.cx:.3f}, serial={'(auto)' if not self.rs_serial else self.rs_serial}"
        )

        # ===== Pub/Sub =====
        self.sub_wall = self.create_subscription(Float32, wall_topic, self._on_wall_distance, 10)
        self.sub_goal = self.create_subscription(Bool, goal_topic, self._on_navigation_goal, 10)
        self.pub_error_x = self.create_publisher(String, out_topic, 10)

        self.get_logger().info(
            f"Started. wall_width={self.wall_width} m, target_barcode={self.target_barcode}, "
            f"goal_topic={goal_topic}, out={out_topic}"
        )

    def destroy_node(self):
        try:
            if hasattr(self, "pipeline") and self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()

    def _publish_error_x(self, text: str) -> None:
        self.pub_error_x.publish(String(data=text))

    def _on_wall_distance(self, msg: Float32) -> None:
        # もう latched 済みなら無視
        if self.latest_wall_distance_m is not None:
            return

        d = float(msg.data)
        if d > 0.0 and math.isfinite(d):
            self.latest_wall_distance_m = d
            self.get_logger().info(f"wall_distance latched: {d:.3f} (unsubscribe)")

            # 購読解除（以降は受け取らない）
            if getattr(self, "sub_wall", None) is not None:
                self.destroy_subscription(self.sub_wall)
                self.sub_wall = None

            # ★ goal が先に来ていたら、このタイミングで1回実行
            if self.pending_goal:
                self.get_logger().info("pending_goal=True -> run once now")
                self.pending_goal = False
                self._run_once()

    def _grab_color_frame_bgr(self, timeout_ms: int = 1500) -> Optional["cv2.Mat"]:
        """
        RealSenseからカラー1枚を取得してBGR(OpenCV)画像として返す
        """
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            color = frames.get_color_frame()
            if not color:
                return None
            img = color.get_data()  # memoryview
            bgr = cv2.cvtColor(
                cv2.cvtColor(
                    cv2.UMat(img).get(),  # ensure numpy
                    cv2.COLOR_RGB2BGR
                ),
                cv2.COLOR_BGR2BGR
            )
            # ↑ RealSenseのbgr8なら本来変換不要なこともあるが、安全側でnumpy化
            return bgr
        except Exception:
            return None
        
    def _on_navigation_goal(self, msg: Bool) -> None:
        self.get_logger().info(f"navigation_goal received: {msg.data}")

        curr = bool(msg.data)

        # False→True の瞬間だけ動かす
        if not (curr and not self.prev_goal):
            self.prev_goal = curr
            self.get_logger().info("skip (not rising edge)")
            return
        self.prev_goal = curr

        # wall_distance がまだなら「待つ」
        if self.latest_wall_distance_m is None:
            self.get_logger().warn("wall_distance not ready -> set pending_goal=True and wait")
            self.pending_goal = True
            return

        # もう wall_distance があるなら即実行
        self._run_once()

    def _run_once(self) -> None:
        """RealSenseで1枚撮影→バーコード→距離計算→/error_x publish を1回だけ実行"""
        self.get_logger().info("capturing RealSense frame...")
        bgr = self._grab_color_frame_bgr()
        if bgr is None:
            self.get_logger().error("RealSense frame capture failed -> publish error_x")
            self._publish_error_x("ERROR: RealSense frame capture failed")
            return

        self.get_logger().info("decoding barcode...")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found = find_barcode(gray, target_code=self.target_barcode)
        if found is None:
            self.get_logger().error("barcode not found -> publish error_x")
            self._publish_error_x("ERROR: barcode not found")
            return

        code, left, top, w, h = found
        self.get_logger().info(f"barcode found: {code}")

        u_center = left + (w / 2.0)
        theta = math.atan((u_center - self.cx) / self.fx)
        c = math.cos(theta)
        if abs(c) < 1e-6:
            self.get_logger().error("cos(theta) too small -> publish error_x")
            self._publish_error_x("ERROR: cos(theta) too small")
            return

        wall_no_barcode = float(self.latest_wall_distance_m)
        wall_to_barcode = self.wall_width - wall_no_barcode
        if wall_to_barcode <= 0.0:
            self.get_logger().error("wall_to_barcode<=0 -> publish error_x")
            self._publish_error_x(
                f"ERROR: wall_to_barcode<=0 wall_width_m={self.wall_width:.3f} wall_no_barcode_m={wall_no_barcode:.3f}"
            )
            return

        range_m = wall_to_barcode / c
        theta_deg = math.degrees(theta)

        error_x = (
            f"code={code} "
            f"range_m={range_m:.3f} "
            f"wall_to_barcode_m={wall_to_barcode:.3f} "
            f"wall_no_barcode_m={wall_no_barcode:.3f} "
            f"wall_width_m={self.wall_width:.3f} "
            f"u={u_center:.1f} "
            f"theta_deg={theta_deg:.2f}"
        )
        self.get_logger().info(f"publishing error_x: {error_x}")
        self._publish_error_x(error_x)



def main():
    rclpy.init()
    node = BarcodeFromRealSenseOnGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
