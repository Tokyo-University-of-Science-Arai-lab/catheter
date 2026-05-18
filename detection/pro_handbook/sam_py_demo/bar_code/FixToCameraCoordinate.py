from pathlib import Path
from typing import Optional, Tuple, Union, Any
from detection.pro_handbook.sam_py_demo.bar_code.code_1_pic import barcode_inference
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

FX = 3340.4421,#WEBカメラ内部パラメータ
FY = 3344.0310,
CX = 1860.6919,
CY = 1435.0312,


def receive_wall_distance_once(
    topic: str = "/wall_distance",
    timeout_sec: float = 1.0,
) -> Optional[float]:
    """
    /wall_distance を1回だけ受信して float を返す。
    timeout_sec 秒以内に来なければ None を返す。
    """
    rclpy.init(args=None)
    node = Node("wall_distance_once_receiver")

    received: dict[str, Optional[float]] = {"depth_m": None}

    def cb(msg: Float32):
        received["depth_m"] = float(msg.data)

    sub = node.create_subscription(Float32, topic, cb, 10)

    # 1回受け取るまで（またはタイムアウトまで）spin_once
    start = node.get_clock().now()
    while rclpy.ok() and received["depth_m"] is None:
        rclpy.spin_once(node, timeout_sec=0.05)
        elapsed = (node.get_clock().now() - start).nanoseconds / 1e9
        if elapsed >= timeout_sec:
            break

    node.destroy_subscription(sub)
    node.destroy_node()
    rclpy.shutdown()

    return received["depth_m"]


def rect_center_uv(rect: Any) -> Tuple[float, float]:
    """rectから画像中心(u,v)を計算（pyzbar Rect互換: left, top, width, height）"""
    u = float(rect.left) + float(rect.width) / 2.0
    v = float(rect.top)  + float(rect.height) / 2.0
    return u, v


def pixel_to_camera(u: float, v: float, Z: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    """ピクセル(u,v)と深度Z[m]からカメラ座標(X,Y,Z)[m]へ（optical frame）"""
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z
def camera_to_amr(X: float):
    """カメラ座標(X,Y,Z)[m]からAMR座標へ変換"""
    #TODO AMR座標系のオフセット追加
    amr_X = -X - 0.395  #例: カメラがAMRの前方0.41mにある場合
    return amr_X

def pixel_to_camera_sequence(barcode_output_rect: Any):
    
    depth_m = receive_wall_distance_once()
    u, v = rect_center_uv(barcode_output_rect)
    X, Y, Z = pixel_to_camera(u, v, depth_m, FX, FY, CX, CY)
    amr_X = camera_to_amr(X)
    #TODO AMR座標系のオフセット追加
    return amr_X

# def barcode_to_camera_xyz(
#     barcode_number: str,
#     image_path: Union[str, Path],
#     depth_m: float = 0.3,
#     fx: float = 3340.4421,#WEBカメラ内部パラメータ
#     fy: float = 3344.0310,
#     cx: float = 1860.6919,
#     cy: float = 1435.0312,
# ) -> Tuple[bool, Optional[Tuple[float, float, float]], Optional[Tuple[float, float]], Optional[Any]]:
#     """
#     バーコード認識結果（rect）から、カメラ座標(X,Y,Z)を返す。

#     Returns:
#       matched: bool
#       xyz: (X,Y,Z) [m] or None
#       uv:  (u,v) pixel or None
#       rect: pyzbar rect or None
#     """
#     image_path = Path(image_path)

#     matched, rect = barcode_inference(barcode_number, image_path)
#     if not matched or rect is None:
#         return False, None, None, None

#     u, v = rect_center_uv(rect)
#     X, Y, Z = pixel_to_camera(u, v, depth_m, fx, fy, cx, cy)
#     return True, (X, Y, Z), (u, v), rect

if __name__ == "__main__":
    ok, xyz, uv, rect = barcode_to_camera_xyz("15 14", "/home/book/Desktop/before_init_rgb.png")
    print("matched:", ok)
    print("rect:", rect)
    print("uv:", uv)
    print("camera xyz [m]:", xyz)

    error_x = xyz + 0.41
    print("error_x", error_x)
