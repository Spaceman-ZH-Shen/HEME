import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from config import get_args_parser

path_args = get_args_parser().parse_args()
model_path = os.path.join(path_args.model_dir, 'Florence-2-large')

# 模型加载
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 输入输出路径
input_json_path = path_args.project_path + "/baseline/VG-RS-question.json"
image_base_path = path_args.project_path + "/baseline/images"
output_json_path = path_args.project_path + "/baseline/baseline_results/baseline_florence2.json"

# 任务类型（可选：<REC>, <CAPTION_TO_PHRASE_GROUNDING>, <OD> 等）
task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

# 读取数据
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 处理数据
for item in tqdm(data):
    image_rel_path = item["image_path"].replace("\\", "/")
    if image_rel_path.startswith("images/"):
        image_rel_path = image_rel_path[len("images/"):]
    
    image_path = os.path.join(image_base_path, image_rel_path)

    if not os.path.exists(image_path):
        print(f"[警告] 找不到图像：{image_path}")
        item["result"] = None
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[错误] 图像打开失败：{image_path}, 错误：{e}")
        item["result"] = None
        continue

    # 拼接 task + question
    question = item.get("question", "")
    prompt = task_prompt + question

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            early_stopping=True
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    # 获取检测框
    if parsed and task_prompt in parsed:
        result_obj = parsed[task_prompt]
        if 'bboxes' in result_obj and len(result_obj['bboxes']) > 0:
            box = result_obj['bboxes'][0]
            point_box = [[box[0], box[1]], [box[2], box[3]]]
            item["result"] = point_box
        else:
            item["result"] = None
    else:
        item["result"] = None

# 保存结果
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ 检测完成，结果保存为：{output_json_path}")
