import os
import re
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor as QwenProcessor
)
from qwen_vl_utils import process_vision_info
from config import get_args_parser
path_args = get_args_parser().parse_args()

# ========== Config ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 模型路径
qwen_path = path_args.model_dir+'/basic_model/Qwen2.5-VL-72B-Instruct/'

# 文件路径
input_json_path = path_args.project_path + "/baseline/VG-RS-question.json"
image_base_path = path_args.project_path + '/baseline/images'

step1_output_path = path_args.project_path + "/knowledge/question_subject.json"

# ========== Load Qwen ==========
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_path, torch_dtype=torch_dtype, device_map="auto", attn_implementation="flash_attention_2"
).eval()
qwen_processor = QwenProcessor.from_pretrained(qwen_path)

with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

step1_results = []

for item in tqdm(data, desc="Extracting Subjects+Objects", ncols=100):
    question = item.get("question", "")
    image_rel_path = item["image_path"]
    image_name = os.path.basename(image_rel_path)
    image_path = os.path.join(image_base_path, image_name)

    result = {
        "image_path": image_rel_path,
        "question": question,
        "subjects": "",
        "objects": "",
        "ans": ""
    }

    prompt = (
    "You are an expert in visual scene understanding.\n"
    "Given a natural language question, extract:\n"
    "1. The main visual subject(s) to be grounded (detected). Return only the **core noun(s)** of the subject. "
    "Remove any positional, directional, or order-related words (e.g., 'left', 'second', 'middle'). "
    "Convert plural forms to singular.\n"
    "2. All mentioned physical objects or elements in the question, including background elements. "
    "Also remove any directional or positional phrases.\n\n"
    f"Question: {question}\n\n"
    "Return your answer strictly in this JSON format:\n"
    "{\n"
    "  \"subjects\": \"subject1. subject2.\",\n"
    "  \"mentioned_objects\": \"object1. object2.\"\n"
    "}\n"
    "Ensure the output is in valid JSON format, and each object ends with a period."
)



    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = qwen_processor(text=[text_input], return_tensors="pt").to(qwen_model.device)

        with torch.no_grad():
            outputs = qwen_model.generate(**inputs, max_new_tokens=128)

        output_text = qwen_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        # 保存完整回答文本
        result["ans"] = output_text

        # 使用正则提取第一个花括号包围的 JSON 字符串
        json_matches = re.findall(r"\{.*?\}", output_text, re.DOTALL)
        if json_matches:
            json_str = json_matches[-1]  # 取最后一个匹配的 JSON 字符串
            parsed = json.loads(json_str)
            result["subjects"] = parsed.get("subjects", "")
            result["objects"] = parsed.get("mentioned_objects", "")
        else:
            print(f"[Warning] No JSON found in model output for {image_path}")
            result["subjects"] = ""
            result["objects"] = ""

    except Exception as e:
        print(f"[Warning] Failed to parse result for {image_path}, error: {e}")
        print(f"Raw output: {output_text if 'output_text' in locals() else 'N/A'}")

    step1_results.append(result)

with open(step1_output_path, "w", encoding="utf-8") as f:
    json.dump(step1_results, f, ensure_ascii=False, indent=2)

print(f"✅ Step 1 complete. Output saved to {step1_output_path}")
