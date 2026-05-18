from dataclasses import dataclass

@dataclass
class Rect:
    left: int
    top: int
    width: int
    height: int

def rect_center_uv(rect: Rect):
    """rectから画像中心(u,v)を計算"""
    u = rect.left + rect.width  / 2.0
    v = rect.top  + rect.height / 2.0
    return u, v

def pixel_to_camera(u, v, Z, fx, fy, cx, cy):
    """
    ピクセル(u,v)と深度Z[m]からカメラ座標(X,Y,Z)[m]へ
    OpenCV/ROS optical frame: x=right, y=down, z=forward
    """
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

# ===== 入力例（あなたのログを模したもの）=====
rect = Rect(left=1501, top=746, width=208, height=38)

# 画像中心点（u,v）を作る
u, v = rect_center_uv(rect)
print("rect center (u,v) =", (u, v))  # (1605.0, 765.0) になるはず

# 深度[m]
Z = 0.3

# ===== カメラ内部パラメータ（ここを実機の値に置き換えてください）=====
# 例: 1920x1080なら cx=960, cy=540 付近、fx/fyはCameraInfoのK[0],K[4]など
fx = 1362.24267578125
fy = 1359.7244873046875
cx = 956.697509765625
cy = 556.5321044921875


X, Y, Z = pixel_to_camera(u, v, Z, fx, fy, cx, cy)
print("camera coords (X,Y,Z) [m] =", (X, Y, Z))