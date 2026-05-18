import json
import time
from rapidfuzz import fuzz, process

start_time = time.time() 

# 读取书名-文字映射
with open(r"C:\Users\HP\Box\output\book_text.json", "r", encoding="utf-8") as f:
    book_data = json.load(f)

# 读取书本坐标信息
with open(r"C:\Users\HP\Box\sam3_output_books.json", "r", encoding="utf-8") as f:
    sam_data = json.load(f)

# 构建书名到坐标的映射
book_boxes = {entry["name"]: entry["box"] for entry in sam_data}

# 构建每本书的文字集合（合并为一个字符串用于匹配）
book_texts = {
    name: " ".join(words) for name, words in book_data.items()
}

# 模糊匹配函数
def match_book_name(query, threshold=40):
    results = []
    for name, text in book_texts.items():
        score = fuzz.WRatio(query, text)
        if score >= threshold:
            results.append((name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# 示例：尝试匹配一个模糊书名
query = "H-制御"
matches = match_book_name(query)

# 打印结果
print(f" 与「{query}」相似的书：")
for name, score in matches:
    print(f"   {name}（相似度: {score:.2f}）")

import cv2

# 读取原图
image_path = r"C:\Users\HP\Box\ocr\test_image.jpg"
image = cv2.imread(image_path)

# 画出匹配结果
for name, score in matches:
    if name in book_boxes:
        box = book_boxes[name]
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])

        # 画矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 标注相似度
        label = f"{score:.2f}%"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
        
end_time = time.time()  

elapsed = end_time - start_time
print(f"时间：{elapsed:.2f} 秒")
# 显示结果图像
cv2.imshow("Matched Books", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 保存或显示结果
# output_path = r"C:\Users\HP\Box\output\matched_books_2.png"
# cv2.imwrite(output_path, image)
# print(f"✅ 匹配结果已保存到：{output_path}")

