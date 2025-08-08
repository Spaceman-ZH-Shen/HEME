import json
import torch
import os
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import get_args_parser
path_args = get_args_parser().parse_args()
# 模型路径与加载
model_path = path_args.model_dir+'/basic_model/Qwen2.5-VL-72B-Instruct'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_path)

# 输入输出路径
input_json = path_args.project_path + "/baseline/VG-RS-question.json"
output_json = path_args.project_path + "/knowledge/question_zh_translated.json"

# 如果存在部分结果，先加载它（增量处理）
if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    processed_set = set((item["image_path"], item["question"]) for item in saved_data)
else:
    saved_data = []
    processed_set = set()

# 加载原始数据
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in tqdm(data, desc="Translating questions"):
    image_path = item["image_path"].replace("images\\",path_args.project_path + '/baseline/images')
    question = item["question"]

    # 跳过已处理项
    if (image_path, question) in processed_set:
        continue

    try:
        # 构造 prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_path).convert("RGB"),
                    },
                    {
                        "type": "text",
                        "text": f"这句话是一个英语不太好的中国人写的：'{question}'，请帮我翻译成自然流畅的中文，用来帮助完成图像的目标识别任务。"
                                f"请注意，question中的目标物体在图中一定存在，可能被遮挡或者过于微小",
                    },
                ],
            }
        ]

        # 构建输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # 推理生成
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        # 保存当前结果
        item["question_zh"] = output_text.strip()
        saved_data.append(item)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(saved_data, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"❌ 处理失败: {image_path}, 问题: {question}, 错误: {str(e)}")
        continue

print(f"✅ 全部完成，共处理 {len(saved_data)} 条问题，结果保存在 {output_json}")
