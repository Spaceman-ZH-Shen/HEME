import json
import os
from config import get_args_parser
from img_size_transfer_tool.bbox_size_converter import transfer_1120_to_ori
path_args = get_args_parser().parse_args()

# ======== 配置路径 ========
json1_path = path_args.project_path + "/knowledge/step3_output_1120_small_model.json"
json2_path = path_args.project_path + "/knowledge/step3_output_1120_large_model.json"
json3_path = path_args.project_path + '/knowledge/step2_output_v2_1120.json'
json4_path = path_args.project_path + '/knowledge/step2_output_v1_1400_1120.json'
json5_path = path_args.project_path + '/knowledge/baseline_72b_1400_1120.json'
json6_path = path_args.project_path + '/knowledge/step2_output_v2_1400_1120.json'
original_path = path_args.project_path + "/baseline/VG-RS-question.json"
output_path = path_args.project_path + "knowledge/step3_merged_result.json"

# ======== 读取 JSON ========
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

json1 = load_json(json1_path)
json2 = load_json(json2_path)
json3 = load_json(json3_path)
json4 = load_json(json4_path)
json5 = load_json(json5_path)
json6 = load_json(json6_path)
original = load_json(original_path)

# 方便后面查找的映射
def to_key(item):
    return f"{item['image_path']}||{item['question']}"

json1_keys = set(to_key(x) for x in json1)
original_keys = set(to_key(x) for x in original)

# ======== Step 1: 分出部分1、部分2 ========
part1_keys = json1_keys  # json1 中有的 image-question 对
part2_keys = original_keys - json1_keys  # 原问题中有，但 json1 中没有

# ======== Step 2: 从 json2 中提取部分2 ========
final_data = []
json2_map = {to_key(x): x for x in json2}
for key in part2_keys:
    if key in json2_map:
        final_data.append(json2_map[key])

# ======== Step 3: 从 json3 中提取部分1 ========
json3_map = {to_key(x): x for x in json3}
for key in part1_keys:
    if key in json3_map:
        final_data.append(json3_map[key])

# ======== Step 4: 检查缺失，按 json4 > json5 > json6 补全 ========
def fill_missing(final_list, ref_list):
    """按 ref_list 补全 final_list 中缺失的 image-question 对"""
    final_keys = set(to_key(x) for x in final_list)
    for item in ref_list:
        key = to_key(item)
        if key in original_keys and key not in final_keys:
            final_list.append(item)
            final_keys.add(key)

fill_missing(final_data, json4)
fill_missing(final_data, json5)
fill_missing(final_data, json6)

# ======== Step 5: 验证结果完整性 ========
final_keys = set(to_key(x) for x in final_data)
missing_keys = original_keys - final_keys
if missing_keys:
    print("⚠️ 警告: 仍有缺失的 image-question 对:")
    for key in missing_keys:
        print(key)
else:
    print("✅ 所有原问题都已补全到最终 JSON")

# ======== Step 6: 保存结果 ========
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"最终结果已保存到: {output_path}，共 {len(final_data)} 条记录")

transfer_1120_to_ori(path_args.project_path + "knowledge/step3_merged_result.json", path_args.project_path + "final_results.json")