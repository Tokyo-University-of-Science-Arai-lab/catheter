import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
 
start_time = time.time()

# 创建以当前时间命名的输出文件夹
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("outputs", f"sam3_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# Load the model
model = build_sam3_image_model(checkpoint_path="sam3.pt")
 
processor = Sam3Processor(model)
# Load an image
image = Image.open(r"C:\Users\24431\Box\ocr\other_pic\IMG20251217193039.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="book spine")
 
# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
 
# 转移到CPU
masks = masks.cpu().numpy()
boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()

#构建要保存的数据
output_data = []
for i, (box, score) in enumerate(zip(boxes, scores)):
    x1, y1, x2, y2 = box.tolist()
    output_data.append({
        "name": f"book{i + 1}",
        "score": float(score),
        "box": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        }
    })

# 保存为 JSON 文件
json_path = os.path.join(output_dir, "sam3_output_books.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4)


 
print(f"找到 {len(masks)} 个目标")
print(f"置信度分数: {scores}")
print(f"边界框:\n{boxes}")
 
# 创建颜色映射，为每个实例分配不同颜色
colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
 
# 创建一张包含所有实例的大图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()
 
# 1. 显示原图
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')
 
# 2. 显示所有实例的合成mask
img_array = np.array(image)
all_masks_overlay = img_array.copy()
 
for i, (mask, score, color) in enumerate(zip(masks, scores, colors)):
    # 确保mask是2D的
    if len(mask.shape) == 3:
        mask = mask[0]
    
    # 调整mask大小以匹配图像
    if mask.shape != img_array.shape[:2]:
        from scipy.ndimage import zoom
        scale_y = img_array.shape[0] / mask.shape[0]
        scale_x = img_array.shape[1] / mask.shape[1]
        mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
    
    # 为每个mask创建彩色覆盖
    mask_bool = mask > 0.5
    # 使用不同颜色的半透明覆盖
    rgb_color = color[:3]  # 取RGB值，忽略alpha
    all_masks_overlay[mask_bool] = all_masks_overlay[mask_bool] * 0.4 + np.array(rgb_color) * 255 * 0.6
 
axes[1].imshow(all_masks_overlay.astype(np.uint8))
axes[1].set_title(f"All Masks Overlay\n({len(masks)} instances)")
axes[1].axis('off')
 
# 3. 显示带边界框的原图
axes[2].imshow(image)
for i, (box, score, color) in enumerate(zip(boxes, scores, colors)):
    x1, y1, x2, y2 = box
    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                         fill=False, color=color, linewidth=3)
    axes[2].add_patch(rect)
    # 添加标签
    axes[2].text(x1, y1-5, f"Obj {i+1}: {score:.3f}", 
                bbox=dict(boxstyle="round,pad=0.3", fc=color, alpha=0.7),
                fontsize=8, color='black')
axes[2].set_title("Bounding Boxes with Scores")
axes[2].axis('off')
 
# 4. 显示所有mask的合成图（黑白）
combined_mask = np.zeros(img_array.shape[:2], dtype=bool)
for i, mask in enumerate(masks):
    if len(mask.shape) == 3:
        mask = mask[0]
    
    # 调整mask大小以匹配图像
    if mask.shape != img_array.shape[:2]:
        from scipy.ndimage import zoom
        scale_y = img_array.shape[0] / mask.shape[0]
        scale_x = img_array.shape[1] / mask.shape[1]
        mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
    
    combined_mask = np.logical_or(combined_mask, mask > 0.5)
 
axes[3].imshow(combined_mask, cmap='gray')
axes[3].set_title(f"Combined Mask\n({len(masks)} instances)")
axes[3].axis('off')

end_time = time.time()
elapsed = end_time - start_time
print(f"时间：{elapsed:.2f} 秒") 
plt.tight_layout()
save_path = os.path.join(output_dir, "all_instances_result.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print("\n所有实例结果已保存到 all_instances_result.png")
plt.show()

# 保存所有掩码为 .npz 文件
mask_data = {}
for i, mask in enumerate(masks):
    if len(mask.shape) == 3:
        mask = mask[0]
    
    # 调整大小
    if mask.shape != img_array.shape[:2]:
        from scipy.ndimage import zoom
        scale_y = img_array.shape[0] / mask.shape[0]
        scale_x = img_array.shape[1] / mask.shape[1]
        mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5

    mask_data[f"mask_{i}"] = mask.astype(bool)

npz_path = os.path.join(output_dir, "sam3_masks.npz")
np.savez_compressed(npz_path, **mask_data)
print(f"掩码元数据已保存到 {npz_path}")


# 保存单独的mask
for i, mask in enumerate(masks):
    if len(mask.shape) == 3:
        mask = mask[0]
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(os.path.join(output_dir,f"mask_{i}.png"))
    print(f"Mask {i} 已保存到 mask_{i}.png")
 
# 额外：创建一个包含所有实例的详细对比图
if len(masks) > 0:
    # 计算需要多少行和列
    n_cols = min(4, len(masks))
    n_rows = (len(masks) + n_cols - 1) // n_cols
    
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    elif n_cols == 1:
        axes2 = axes2.reshape(-1, 1)
    
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        row = i // n_cols
        col = i % n_cols
        
        # 确保mask是2D的
        if len(mask.shape) == 3:
            mask = mask[0]
        
        # 调整mask大小以匹配图像
        if mask.shape != img_array.shape[:2]:
            from scipy.ndimage import zoom
            scale_y = img_array.shape[0] / mask.shape[0]
            scale_x = img_array.shape[1] / mask.shape[1]
            mask = zoom(mask, (scale_y, scale_x), order=0) > 0.5
        
        # 创建彩色mask overlay
        overlay = img_array.copy()
        mask_bool = mask > 0.5
        color = colors[i]
        rgb_color = color[:3]
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(rgb_color) * 255 * 0.5
        
        axes2[row, col].imshow(overlay.astype(np.uint8))
        
        # 绘制边界框
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, color=color, linewidth=2)
        axes2[row, col].add_patch(rect)
        
        axes2[row, col].set_title(f"Instance {i+1}\nScore: {score:.3f}")
        axes2[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(masks), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes2[row, col].axis('off')
    
    

    plt.tight_layout()
    save_path = os.path.join(output_dir, "detailed_instances_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print("详细实例对比图已保存到 detailed_instances_comparison.png")
    plt.show()

