### 此文件如果运行不了则需要在llmdet官方目录下运行，参考地址：https://github.com/iSEE-Laboratory/LLMDet/tree/main?tab=readme-ov-file

import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection  # 本地模型定义
from config import get_args_parser

path_args = get_args_parser().parse_args()
model_id = os.path.join(path_args.model_dir, 'llmdet')
image_base_path = path_args.project_path + "/baseline/images/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型与处理器
processor = GroundingDinoProcessor.from_pretrained(model_id)
model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)

# 加载图像-问题对 JSON 数据
with open(path_args.project_path + "/baseline/VG-RS-question.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 替换图像路径为绝对 Linux 路径
for item in data:
    item["image_path"] = item["image_path"].replace("images\\", image_base_path)

# 初始化结果列表
results = []

# 遍历数据处理
for item in tqdm(data, desc="Processing images with LLMDet"):
    try:
        image_path = item["image_path"]
        question = item["question"].strip().lower()

        # 跳过空问题或无图像路径
        if not image_path or not question:
            print(f"[WARNING] Skipped empty entry: {item}")
            continue

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 准备文本（确保以句号结尾）
        text = question.rstrip(".") + "."

        # 编码输入，加 max_length 以避免 text_token_mask 错误
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        ).to(device)

        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)

        # 后处理，提取检测框
        results_post = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]  # 注意 PIL 是 (W, H)，这里需要 (H, W)
        )

        boxes = results_post[0]["boxes"].tolist()

        # 如果没有检测框，添加默认值
        if not boxes:
            formatted_boxes = [[[0, 0], [0, 0]]]
        else:
            # 选择面积最大的检测框
            boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
            largest_box = boxes[0]
            formatted_boxes = [[[int(largest_box[0]), int(largest_box[1])], [int(largest_box[2]), int(largest_box[3])]]]

        # 记录结果
        results.append({
            "image_path": image_path,
            "question": question,
            "result": formatted_boxes
        })

    except Exception as e:
        print(f"[ERROR] Failed to process item: {item}")
        print(f"Reason: {e}")
        continue

# 保存最终结果
output_path = path_args.project_path + "/baseline/baseline_results/baseline_llmdet.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ Done! All results saved to: {output_path}")
