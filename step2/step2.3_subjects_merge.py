import json
import os
from collections import defaultdict
from config import get_args_parser

path_args = get_args_parser().parse_args()
# 读取原始数据
with open(path_args.project_path + "/knowledge/question_subject.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 用于按 image_path 分组合并数据
merged = defaultdict(lambda: {"subjects": set(), "objects": set()})

for item in data:
    image_path = item["image_path"]
    # 以句号分隔并清理空字符串
    subjects = [s.strip() for s in item.get("subjects", "").split(".") if s.strip()]
    objects = [o.strip() for o in item.get("objects", "").split(".") if o.strip()]

    merged[image_path]["subjects"].update(subjects)
    merged[image_path]["objects"].update(objects)

# 最终输出结果列表
result = []
for image_path, content in merged.items():
    subject_set = sorted(content["subjects"])
    object_set = sorted(content["objects"].union(content["subjects"]))  # 保证object_set包含全部subject_set

    result.append({
        "image_path": image_path,
        "subject_set": ". ".join(subject_set) + ".",
        "object_set": ". ".join(object_set) + "."
    })

# 保存到文件
with open(path_args.project_path+"/knowledge/img_subj.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("合并1完成")

############################################################################################

def merge_scene_and_subjects(scene_path, subj_path, output_path):
    # Load scene descriptions
    with open(scene_path, 'r') as f:
        scenes = json.load(f)

    # Load subject lists
    with open(subj_path, 'r', encoding="utf-8") as f:
        subs = json.load(f)

    # Build a mapping from image filename (basename) to its subject info
    subj_map = {
        os.path.basename(item['image_path']): item
        for item in subs
    }

    # Merge scene + subject info
    combined = []
    for scene in scenes:
        img_name = scene['image_name']
        subj_info = subj_map.get(img_name, {})

        combined.append({
            'image_name':        img_name,
            'scene_scope':       scene.get('scene_scope'),
            'sub_scenes':   scene.get('top2_sub_scenes')[0],
            'subject_set':       subj_info.get('subject_set', []),
            'object_set':        subj_info.get('object_set', [])
        })

    # Write merged JSON
    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Merged {len(combined)} entries to {output_path}")

scene_path = path_args.project_path+"/knowledge/img_scene.json"
subj_path = path_args.project_path+"/knowledge/img_subj.json"
output_path = path_args.project_path+"/knowledge/img_scene_subj.json"

merge_scene_and_subjects(scene_path, subj_path, output_path)