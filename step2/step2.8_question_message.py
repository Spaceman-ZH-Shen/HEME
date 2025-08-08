import json
from collections import defaultdict
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from config import get_args_parser
path_args = get_args_parser().parse_args()
# 模型路径
model_name = path_args.model_dir+'/basic_model/Qwen2.5-VL-72B-Instruct'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_name)

# 输入输出路径
input_path = path_args.project_path + "/baseline/VG-RS-question.json"
output_path = path_args.project_path + "/knowledge/question_message.json"

# 加载输入数据
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 聚合每张图的所有问题
grouped_questions = defaultdict(list)
for item in data:
    grouped_questions[item["image_path"]].append(item["question"])


# 提示词构造函数
def build_prompt(image_path, questions):
    question_str = " ".join(q.strip().rstrip(".") + "." for q in questions)
    prompt = f"""You are given a list of object descriptions extracted from a single image, as follows:
"{question_str}"

Based on these descriptions, infer what objects are likely present in the image and describe their spatial relationships in a concise, structured way.

Your response should:
- Clearly list the objects mentioned.
- Indicate how they are positioned relative to one another (e.g., "on the left", "next to", "from left to right").
- If there are multiple objects of the same type and there is a spatial relationship between them, please clearly state the total number of objects of this type and their spatial relationship.
- Avoid assumptions or visual details that are not mentioned in the input.
- Write in a factual and neutral tone, suitable for object detection or grounding tasks.

Begin your description with "The image contains..." and make the result complete and self-contained."""

    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]


# 推理生成
results = []
for image_path, questions in tqdm(grouped_questions.items(), desc="Generating summaries"):
    messages = build_prompt(image_path, questions)

    # 构造输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt", padding=True).to("cuda")

    # 模型生成
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[-1]:]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # 保存结果
    results.append({
        "image_path": image_path,
        "description": output_text
    })

# 写入输出 JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ 完成生成，总共处理 {len(results)} 张图片，结果已保存至：{output_path}")