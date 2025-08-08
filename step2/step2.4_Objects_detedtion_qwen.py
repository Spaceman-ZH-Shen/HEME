import os
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import get_args_parser
path_args = get_args_parser().parse_args()

# ========== 路径与配置 ==========
def qwen_find_subj_bbox(image_path, output_path):
    model_name = path_args.model_dir + '/basic_model/Qwen2.5-VL-72B-Instruct'
    input_json = path_args.project_path + "/baseline/question_subject.json"
    output_json = path_args.project_path + output_path
    start_index = 0  # 从第 0 条开始处理

    # 加载模型和处理器
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_name)

    # 加载输入数据
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 如果是从头开始处理就初始化输出文件
    if start_index == 0:
        with open(output_json, "w", encoding="utf-8") as f:
            f.write("[\n")
    else:
        print(f"从第 {start_index} 条数据开始处理，结果将追加写入 {output_json}")

    for idx, item in enumerate(tqdm(data, desc="Processing Forward")):
        if idx < start_index:
            continue  # 跳过前面处理过的样本

        image_name = item["image_path"].replace("\\", "/")
        image_path = item["image_path"].replace("images\\", path_args.project_path + image_path)
        subject = item.get("subjects", "").strip().rstrip(".")
        question = item.get("question", "")

        output_record = {
            "image_path": image_name,
            "question": question,
            "subjects": subject,
            "qwen_raw_output": ""
        }

        if subject:
            try:
                image = Image.open(image_path).convert("RGB")
                prompt_text = (
                    f"You are an expert in visual grounding.\n"
                    f"Locate and draw bounding boxes for all possible objects in the image that can be considered as: \"{subject}\".\n"
                    f"Be inclusive: for example, both a water bottle and a cup may be matched for “cup”.\n"
                    f"The same object may exist in multiple forms (such as a light that is on or off, etc.), so please be careful to distinguish.\n"
                    f"The identified object may be small or obscured, please identify it carefully."
                    f"Return the bounding boxes in JSON format as a list of [[x1, y1], [x2, y2]] pairs, one pair for each detected object."
                )

                messages = [{"role": "user",
                             "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                   return_tensors="pt").to("cuda")

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=256)
                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                                         zip(inputs.input_ids, generated_ids)]
                output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

                output_record["qwen_raw_output"] = output_text

            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # 追加写入当前结果
        with open(output_json, "a", encoding="utf-8") as f:
            json.dump(output_record, f, ensure_ascii=False, indent=2)
            if idx < len(data) - 1:
                f.write(",\n")
            else:
                f.write("\n")  # 最后一条不加逗号

    # 如果是最后一条，闭合 JSON 数组
    if start_index == 0 or start_index + 1 >= len(data):
        with open(output_json, "a", encoding="utf-8") as f:
            f.write("]\n")

    print(f"从第 {start_index} 条开始已处理完毕，结果写入：{output_json}")

qwen_find_subj_bbox(image_path='/baseline/Qwen_baseline/img_1120/images',output_path="/knowledge/qwen_subj_bbox_1120.json")
qwen_find_subj_bbox(image_path='/baseline/Qwen_baseline/img_1400/images',output_path="/knowledge/qwen_subj_bbox_1400.json")