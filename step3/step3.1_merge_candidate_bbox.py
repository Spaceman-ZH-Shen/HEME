import os
import json
from collections import defaultdict
from config import get_args_parser
path_args = get_args_parser().parse_args()

# ===== 手动输入要合并的 JSON 文件路径 =====
input_files = [
    path_args.project_path + "/knowledge/bbox_florence_1120.json",
    path_args.project_path + "/knowledge/bbox_florence_ft_1120.json",
    path_args.project_path + "/knowledge/bbox_gdino_1120.json",
    path_args.project_path + "/knowledge/bbox_llmdet_1120.json",
    path_args.project_path + '/knowledge/baseline_72b_1120.json',
    path_args.project_path + '/knowledge/baseline_72b_1400_1120.json',
    path_args.project_path + '/knowledge/step2_output_v1_1120.json',
    path_args.project_path + '/knowledge/step2_output_v1_1400_1120.json',
    path_args.project_path + '/knowledge/step2_output_v2_1400_1120.json',
    path_args.project_path + '/knowledge/step2_output_v2_1120.json'
]

output_path = path_args.project_path + '/knowledge/step3_candidate_boxes.json'

# 文件名关键词 -> 输出字段名
key_map = {
    "baseline_72b_1400_1120": "qwen_72b_bbox",
    "step2_output_v2_1120": "step2_1120_bbox",
    "step2_output_v1_1400_1120": "step2_1400_bbox",
    "step2_output_v2_1400_1120": "step2_1400v2_bbox",
    "bbox_florence_1120": "florence_bbox",
    "bbox_llmdet_1120": "llmdet_bbox",
    "bbox_gdino_1120": "gdino_bbox",
    "bbox_florence_ft_1120": "florence_ft_bbox"
}

# 标准化 image 名字
def clean_image_name(path):
    return os.path.basename(path).replace("\\", "/").split("/")[-1]

# 主数据结构
merged_data = defaultdict(dict)

# 遍历每个手动输入的文件
for file_path in input_files:
    if not os.path.exists(file_path):
        print(f"⚠️ 文件不存在，跳过：{file_path}")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 根据文件名确定字段名
    field_name = None
    fname = os.path.basename(file_path).lower()
    for k in key_map:
        if k in fname:
            field_name = key_map[k]
            break
    if not field_name:
        print(f"⚠️ 跳过未知类型文件：{file_path}")
        continue

    for item in data:
        image_name = clean_image_name(item["image_path"])
        question = item["question"]
        bbox = item.get("result")

        if not bbox:
            continue

        key = (image_name, question)
        merged_data[key]["image_name"] = image_name
        merged_data[key]["question"] = question
        merged_data[key][field_name] = bbox

# 转换为列表并保存
output_list = list(merged_data.values())

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_list, f, indent=2, ensure_ascii=False)

print(f"✅ 合并完成，共保存 {len(output_list)} 条数据到 {output_path}")
