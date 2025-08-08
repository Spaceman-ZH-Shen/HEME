import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from config import get_args_parser
# 设置设备和模型路径

path_args = get_args_parser().parse_args()
model_id = os.path.join(path_args.model_dir, 'grounding_dino')
image_base_path = path_args.project_path + "/baseline/images/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化模型和预处理器
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# 加载数据文件
with open(path_args.project_path + "/baseline/VG-RS-question.json", "r") as f:
    data = json.load(f)

# 替换图片路径（Windows 转 Linux 路径）
for item in data:
    item["image_path"] = item["image_path"].replace("images\\", image_base_path)

results = []

# 遍历数据，逐条处理
for item in tqdm(data, desc="Processing images"):
    try:
        image_path = item["image_path"]
        question = item["question"]

        # 加载图片
        image = Image.open(image_path).convert("RGB")

        # 构造输入文本
        text = question.lower().strip() + "."

        # 编码输入，关键改进：加 max_length 和 truncation
        inputs = processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256  # 保证和 Grounding DINO 模型一致
        ).to(device)

        # 模型前向
        with torch.no_grad():
            outputs = model(**inputs)

        # 后处理，获取目标检测结果
        detection_results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]  # PIL: (W, H) → target_sizes: (H, W)
        )

        # 提取坐标并格式化
        boxes = detection_results[0]["boxes"].tolist()
        scores = detection_results[0]["scores"].tolist()

        if boxes:
            # 选择得分最高的框
            max_score_idx = scores.index(max(scores))
            best_box = boxes[max_score_idx]
            formatted_box = [[int(best_box[0]), int(best_box[1])], [int(best_box[2]), int(best_box[3])]]
        else:
            # 如果没有检测到框，返回默认框
            formatted_box = [[0, 0], [image.size[0], image.size[1]]]  # 默认框为整个图片

        # 记录结果
        results.append({
            "image_path": image_path,
            "question": question,
            "result": formatted_box
        })

    except Exception as e:
        print(f"[ERROR] Failed to process: {image_path} | Question: {question}")
        print(f"Reason: {e}")
        continue

# 保存结果到 JSON 文件（注意路径写法）
output_path = path_args.project_path + "/baseline/baseline_results/baseline_gdino.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"\n✅ Done! Results saved to: {output_path}")