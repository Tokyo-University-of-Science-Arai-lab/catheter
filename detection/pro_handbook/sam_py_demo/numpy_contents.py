import numpy as np, cv2
from pathlib import Path

img_path = Path("/home/tai/Downloads/masks/after_init_rgb (0).png")  # 同じペアの画像
npy_path = Path("/home/tai/Downloads/masks/mask1.npy")

img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
H, W = img.shape[:2]

arr = np.load(str(npy_path), allow_pickle=True)
pts = arr.item() if (isinstance(arr, np.ndarray) and arr.dtype==object and arr.size==1) else arr
# pts は object の可能性があるので強制的に数値化
pts = np.asarray(pts, dtype=np.float32)  # ここが重要
pts = np.rint(pts).astype(np.int32)      # roundでもOKだが rint のほうが明確


vis = img[:, :, :3].copy() if img.ndim==3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.polylines(vis, [pts], isClosed=True, color=(0,255,0), thickness=2)  # 輪郭を描く
cv2.imwrite("debug_polyline.png", vis)

mask = np.zeros((H,W), np.uint8)
cv2.fillPoly(mask, [pts], 1)
cv2.imwrite("debug_fill.png", (mask*255).astype(np.uint8))
print("pts:", pts.shape, "closed?", np.all(pts[0]==pts[-1]))
