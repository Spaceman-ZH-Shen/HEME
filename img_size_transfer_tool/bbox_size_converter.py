import json
import os
from tqdm import tqdm
from config import get_args_parser
path_args = get_args_parser().parse_args()

def transfer_1120_to_ori(bbox_input_path, output_path):
    size_json_path = path_args.project_path + "/img_size_transfer_tool/image_size.json"
    # 读取图片尺寸信息
    with open(size_json_path, "r", encoding="utf-8") as f:
        size_data = json.load(f)

    # 构建一个 {image_name: [w, h]} 的映射表
    size_map = {item["image_name"]: item["original_size"] for item in size_data}

    # 读取 bbox 文件
    with open(bbox_input_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)

    # 执行 bbox 逆缩放
    restored_data = []
    for item in bbox_data:
        image_name = os.path.basename(item["image_path"]).replace("\\", "/")
        if image_name not in size_map:
            print(f"[Warning] 找不到图片尺寸信息: {image_name}")
            continue

        orig_w, orig_h = size_map[image_name]
        scale_x = orig_w / 1120
        scale_y = orig_h / 1120

        restored_bbox = [
            [int(item["result"][0][0] * scale_x), int(item["result"][0][1] * scale_y)],
            [int(item["result"][1][0] * scale_x), int(item["result"][1][1] * scale_y)],
        ]

        restored_data.append({
            "image_path": item["image_path"],
            "question": item["question"],
            "result": restored_bbox
        })

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(restored_data, f, indent=2, ensure_ascii=False)

    print(f"还原完成，结果保存至：{output_path}")


def transfer_1400_to_ori(bbox_json_path, output_path):
    size_json_path = path_args.project_path + "/img_size_transfer_tool/image_size.json"
    target_size = 1400

    # 加载图像原始尺寸信息
    with open(size_json_path, 'r') as f:
        size_data = json.load(f)
    image_size_dict = {
        item["image_name"]: item["original_size"] for item in size_data
    }

    # 加载bbox数据
    with open(bbox_json_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)

    restored_bboxes = []
    for item in tqdm(bbox_data):
        image_name = os.path.basename(item["image_path"]).replace("\\", "/")
        orig_size = image_size_dict.get(image_name)
        if not orig_size:
            print(f"[Warning] No size info for {image_name}")
            continue

        orig_w, orig_h = orig_size
        scale = max(orig_w, orig_h) / target_size

        clipped_bbox = []
        for coord in item["result"]:
            x = int(coord[0] * scale)
            y = int(coord[1] * scale)
            # 裁剪到图像边界内
            x = min(max(x, 0), orig_w - 1)
            y = min(max(y, 0), orig_h - 1)
            clipped_bbox.append([x, y])

        restored_bboxes.append({
            "image_path": item["image_path"],
            "question": item["question"],
            "result": clipped_bbox
        })

    # 写入结果
    with open(output_path, 'w') as f:
        json.dump(restored_bboxes, f, indent=2)

def transfer_ori_to_1120(bbox_input_path, output_path):
    size_json_path = path_args.project_path + "/img_size_transfer_tool/image_size.json"
    with open(size_json_path, "r", encoding="utf-8") as f:
        size_data = json.load(f)

    # 构建尺寸映射表
    size_map = {item["image_name"]: item["original_size"] for item in size_data}

    # 读取 bbox 数据
    with open(bbox_input_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)

    scaled_data = []
    for item in bbox_data:
        if "result" not in item or not item["result"] or not isinstance(item["result"], list):
            print(f"[Skip] Invalid or empty bbox for {item.get('image_path', 'Unknown')}")
            continue

        image_name = os.path.basename(item["image_path"]).replace("\\", "/")
        if image_name not in size_map:
            print(f"[Warning] 找不到图片尺寸信息: {image_name}")
            continue

        orig_w, orig_h = size_map[image_name]
        scale_x = 1120 / orig_w
        scale_y = 1120 / orig_h

        scaled_bbox = [
            [int(item["result"][0][0] * scale_x), int(item["result"][0][1] * scale_y)],
            [int(item["result"][1][0] * scale_x), int(item["result"][1][1] * scale_y)],
        ]

        scaled_data.append({
            "image_path": item["image_path"],
            "question": item["question"],
            "result": scaled_bbox
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scaled_data, f, indent=2, ensure_ascii=False)

    print(f"[Done] 缩放完成，保存至：{output_path}")

def transfer_ori_to_1400(bbox_json_path, output_path):
    size_json_path = path_args.project_path + "/img_size_transfer_tool/image_size.json"
    target_size = 1400

    # 加载图像原始尺寸信息
    with open(size_json_path, 'r', encoding='utf-8') as f:
        size_data = json.load(f)
    image_size_dict = {
        item["image_name"]: item["original_size"] for item in size_data
    }

    # 加载bbox数据
    with open(bbox_json_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)

    scaled_bboxes = []
    for item in tqdm(bbox_data):
        # 判断是否存在有效 bbox
        if "result" not in item or not item["result"] or not isinstance(item["result"], list):
            print(f"[Skip] Invalid or empty bbox for {item.get('image_path', 'Unknown')}")
            continue

        image_name = os.path.basename(item["image_path"]).replace("\\", "/")
        orig_size = image_size_dict.get(image_name)
        if not orig_size:
            print(f"[Warning] No size info for {image_name}")
            continue

        orig_w, orig_h = orig_size
        scale = target_size / max(orig_w, orig_h)

        try:
            new_bbox = [
                [int(coord[0] * scale), int(coord[1] * scale)]
                for coord in item["result"]
            ]
        except Exception as e:
            print(f"[Error] Bbox parse failed for {image_name}: {e}")
            continue

        scaled_bboxes.append({
            "image_path": item["image_path"].replace("\\", "/"),
            "question": item["question"],
            "result": new_bbox
        })

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scaled_bboxes, f, indent=2)