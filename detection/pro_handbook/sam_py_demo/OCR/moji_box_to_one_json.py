import json

def connect_to_database(masks, out):
    # 读取 OCR 结果
    with open(r"C:\Users\HP\Box\output_20251218_105526\IMG20251217193039_res.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    all_boxes = data.get("dt_polys", [])
    all_mojis = data.get("rec_texts", [])
    moji_box = dict(zip(all_mojis, all_boxes))

    # 读取 SAM3 结果
    with open(r"C:\Users\HP\Box\sam3_20251218_114416\sam3_output_books.json", "r", encoding="utf-8") as f:
        sam_data = json.load(f)

    # 准备结果字典：每本书名对应一个文字列表
    book_texts = {entry["name"]: [] for entry in sam_data}

    # 辅助函数：计算两个矩形的交集面积
    def intersection_area(a, b):
        x_overlap = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
        y_overlap = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
        return x_overlap * y_overlap

    # 遍历每段文字及其坐标
    for moji, poly in moji_box.items():
        # 计算文字框的最小外接矩形
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        moji_rect = {"x1": min_x, "y1": min_y, "x2": max_x, "y2": max_y}
        moji_area = (max_x - min_x) * (max_y - min_y)

        if moji_area == 0:
            continue  # 跳过无效区域

        # 找出重叠占比最大的书框
        best_match = None
        best_ratio = 0

        for entry in sam_data:
            book_box = entry["box"]
            inter_area = intersection_area(moji_rect, book_box)
            ratio = inter_area / moji_area
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = entry["name"]

        # 如果有匹配，就归属到占比最大的书
        if best_match:
            book_texts[best_match].append(moji)

    # 打印结果
    for book_name, texts in book_texts.items():
        print(f"{book_name}: {texts}")

    # #可选：保存为 JSON 文件
    # with open(r"C:\Users\HP\Box\output_20251218_105526\book_text.json", "w", encoding="utf-8") as f:
    #     json.dump(book_texts, f, ensure_ascii=False, indent=2)

    print("文字归属信息已保存到 book_text_mapping.json")


    # import json

    # # 读取 OCR 结果
    # with open(r"C:\Users\HP\Box\output_20251218_105526\IMG20251217193039_res.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)

    # all_boxes = data.get("dt_polys", [])
    # all_mojis = data.get("rec_texts", [])
    # moji_box = dict(zip(all_mojis, all_boxes))

    # # 读取 SAM3 结果
    # with open(r"C:\Users\HP\Box\sam3_20251218_114416\sam3_output_books.json", "r", encoding="utf-8") as f:
    #     sam_data = json.load(f)

    # # 准备结果字典：每本书名对应一个文字列表
    # book_texts = {entry["name"]: [] for entry in sam_data}

    # # 遍历每段文字及其坐标
    # for moji, poly in moji_box.items():
    #     for entry in sam_data:
    #         box = entry["box"]
    #         # 检查是否有任意一个点在书框内
    #         if any(box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"] for x, y in poly):
    #             book_texts[entry["name"]].append(moji)


    # # 打印结果
    # for book_name, texts in book_texts.items():
    #     print(f"{book_name}: {texts}")

    # #可选：保存为 JSON 文件
    # with open(r"C:\Users\HP\Box\output\book_text.json", "w", encoding="utf-8") as f:
    #      json.dump(book_texts, f, ensure_ascii=False, indent=2)

    # print("文字归属信息已保存到 book_text.json")








    # import json

    # # 读取 OCR 结果
    # with open(r"C:\Users\HP\Box\output_20251218_105526\IMG20251217193039_res.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)

    # all_boxes = data.get("dt_polys", [])
    # all_mojis = data.get("rec_texts", [])
    # moji_box = dict(zip(all_mojis, all_boxes))

    # # 读取 SAM3 结果
    # with open(r"C:\Users\HP\Box\sam3_20251218_114416\sam3_output_books.json", "r", encoding="utf-8") as f:
    #     sam_data = json.load(f)

    # # 准备结果字典：每本书名对应一个文字列表
    # book_texts = {entry["name"]: [] for entry in sam_data}

    # # 计算两个矩形的交集面积
    # def intersection_area(a, b):
    #     x_overlap = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    #     y_overlap = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    #     return x_overlap * y_overlap

    # # 遍历每段文字及其坐标
    # for moji, poly in moji_box.items():
    #     xs = [pt[0] for pt in poly]
    #     ys = [pt[1] for pt in poly]
    #     min_x, max_x = min(xs), max(xs)
    #     min_y, max_y = min(ys), max(ys)
    #     moji_rect = {"x1": min_x, "y1": min_y, "x2": max_x, "y2": max_y}
    #     moji_area = (max_x - min_x) * (max_y - min_y)

    #     if moji_area == 0:
    #         continue  # 跳过无效区域

    #     best_match = None
    #     best_ratio = 0

    #     for entry in sam_data:
    #         book_box = entry["box"]
    #         inter_area = intersection_area(moji_rect, book_box)
    #         ratio = inter_area / moji_area
    #         if ratio > best_ratio:
    #             best_ratio = ratio
    #             best_match = entry["name"]

    #     # 设置一个较低的阈值，比如 0.2（20% 以上重叠就算匹配）
    #     if best_ratio > 0.01:
    #         book_texts[best_match].append(moji)

    # # 打印结果
    # for book_name, texts in book_texts.items():
    #     print(f"{book_name}: {texts}")

    # 保存为 JSON 文件
    output_path = r"C:\Users\HP\Box\output_20251218_105526\book_text.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(book_texts, f, ensure_ascii=False, indent=2)

    print(f"✅ 文字归属信息已保存到 {output_path}")





    matched_mojis = sum(len(texts) for texts in book_texts.values())
    total_mojis = len(all_mojis)
    unmatched_mojis = total_mojis - matched_mojis

    print(f"\n📊 总文字数: {total_mojis}")
    print(f"✅ 匹配到书的文字数: {matched_mojis}")
    print(f"❌ 未匹配的文字数: {unmatched_mojis}")
    return out_appemd_id
