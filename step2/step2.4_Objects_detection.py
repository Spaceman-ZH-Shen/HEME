import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import (
    AutoProcessor as FlorenceProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Qwen2_5_VLForConditionalGeneration
)
from config import get_args_parser
from qwen_vl_utils import process_vision_info
path_args = get_args_parser().parse_args()

# ========== 路径与配置 ==========
florence_model_path = path_args.model_dir + "/Florence-2-large-ft"
dino_model_path = path_args.model_dir + "/grounding_dino"

input_json_path = path_args.project_path+"/knowledge/img_subj.json"
image_base_path = path_args.project_path + '/baseline/images'
output_json_path = path_args.project_path + "/knowledge/florence_dino_object_grounding.json"

task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ========== 加载模型 ==========
# Florence-2
florence_model = AutoModelForCausalLM.from_pretrained(
    florence_model_path, trust_remote_code=True, torch_dtype=torch_dtype
).to(device)
florence_processor = FlorenceProcessor.from_pretrained(
    florence_model_path, trust_remote_code=True
)

# Grounding DINO
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_path).to(device)
dino_processor = AutoProcessor.from_pretrained(dino_model_path)

# ========== 加载数据 ==========
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ========== 主处理 ==========
for item in tqdm(data, desc="Processing"):
    image_rel_path = item["image_path"].replace("\\", "/")
    if image_rel_path.startswith("images/"):
        image_rel_path = image_rel_path[len("images/"):]
    image_path = os.path.join(image_base_path, image_rel_path)

    if not os.path.exists(image_path):
        print(f"[警告] 图像不存在: {image_path}")
        item["florence_result"] = {}
        item["dino_result"] = {}
        continue

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[错误] 图像打开失败: {image_path} | 错误: {e}")
        item["florence_result"] = {}
        item["dino_result"] = {}
        continue

    object_list = [obj.strip().lower() for obj in item.get("subject_set", "").split(".") if obj.strip()]
    if not object_list:
        item["florence_result"] = {}
        item["dino_result"] = {}
        continue

    # ================= Florence Grounding =================
    florence_prompt = task_prompt + ". ".join(object_list) + "."
    florence_inputs = florence_processor(text=florence_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=florence_inputs["input_ids"],
            pixel_values=florence_inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            early_stopping=True
        )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = florence_processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    florence_result = {}
    if parsed and task_prompt in parsed:
        result_obj = parsed[task_prompt]
        for label, box in zip(result_obj.get("labels", []), result_obj.get("bboxes", [])):
            if label not in florence_result:
                florence_result[label] = []
            florence_result[label].append([[box[0], box[1]], [box[2], box[3]]])
    item["florence_result"] = florence_result



    # ================= Grounding DINO =================
    dino_result = {}
    for obj in object_list:
        dino_inputs = dino_processor(
            images=image,
            text=obj,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            dino_outputs = dino_model(**dino_inputs)

        detection_results = dino_processor.post_process_grounded_object_detection(
            dino_outputs,
            dino_inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]  # (H, W)
        )

        boxes = detection_results[0]["boxes"].tolist()
        formatted_boxes = [[[int(b[0]), int(b[1])], [int(b[2]), int(b[3])]] for b in boxes]
        dino_result[obj] = formatted_boxes

    item["dino_result"] = dino_result

# ========== 保存结果 ==========
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n✅ 完成！结果已保存至：{output_json_path}")

# ========== 结束 ==========
