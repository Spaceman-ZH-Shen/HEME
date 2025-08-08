import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor as QwenProcessor
from qwen_vl_utils import process_vision_info
from config import get_args_parser
path_args = get_args_parser().parse_args()

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 模型路径
qwen_path = path_args.model_dir+'/basic_model/Qwen2.5-VL-72B-Instruct'

# 加载 Qwen2.5-VL 模型
print("Loading Qwen2.5-VL model...")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_path, torch_dtype=torch.float16, device_map="auto",attn_implementation="flash_attention_2"
).eval()
qwen_processor = QwenProcessor.from_pretrained(qwen_path)

# 输入输出路径
input_json_path = path_args.project_path + "/knowledge/img_scene_subj.json"
image_base_path = path_args.project_path + '/baseline/images'
output_json_path = path_args.project_path + "/knowledge/image_caption.json"

def build_caption_messages(image_path: str, focus_question: str = None, scene = None):

    if focus_question:
        prompt = (
            f"You are now in the scene of {scene}, please provide a detailed description of this image. "
            f"Pay special attention to objects of: {focus_question}, these object might have multiple instances in the image.\n\n"
            f"Your description should be a complete paragraph starts with **SS** and end with **END**  include:\n"
            f"1. Main objects and their characteristics (color, size, position, etc.)\n"
            f"2. People and their actions/clothing if present\n"
            f"3. Spatial relationships between objects\n"
            f"4. Any text or signs visible in the image\n"
            f"Please be as descriptive and detailed as possible,also, some objects is very small and maybe blocked by other objects, please be careful"
        )
    else:
        prompt = (
            f"Please provide a detailed and comprehensive description of this image. "
            f"Your description should include:\n"
            f"1. Overall scene and setting\n"
            f"2. Main objects and their characteristics (color, size, position, etc.)\n"
            f"3. People and their actions/clothing if present\n"
            f"4. Spatial relationships between objects\n"
            f"5. Any text or signs visible in the image\n"
            f"Please be as descriptive and detailed as possible, also, some objects is very small and maybe blocked by other objects, please be careful "
        )
    
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]

# 读取输入数据
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# 推理流程
print("Starting caption generation...")
for item in tqdm(data, desc="生成图像描述", ncols=100):
    image_name = item["image_name"]
    #image_rel_path = item["image_path"].replace("\\", "/")
    #if image_rel_path.startswith("images/"):
    #    image_rel_path = image_rel_path[len("images/"):]
    #image_path = os.path.join(image_base_path, image_rel_path)
    image_path = os.path.join(image_base_path, image_name)
    if not os.path.exists(image_path):
        print(f"[跳过] 找不到图像：{image_path}")
        item["caption"] = None
        item["error"] = "Image not found"
        results.append(item)
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        
        # 可以选择是否使用问题作为引导
        use_question_guidance = True  # 设置为False则不使用问题引导
        
        if use_question_guidance:
            question = item.get("subject_set", "")
            scene_main = item.get('scene_scope',"")
            scene_spec = item.get('sub_scenes',"")
            scene = scene_main + '-' +scene_spec
            messages = build_caption_messages(image_path, question,scene)
        else:
            messages = build_caption_messages(image_path)

        # 使用Qwen2.5-VL生成详细描述
        text_input = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = qwen_processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True
        ).to("cuda")

        with torch.no_grad():
            outputs = qwen_model.generate(
                **inputs, 
                max_new_tokens=1024,  # 增加token数量以获得更详细的描述
                do_sample=True,
                temperature=0.3,  # 稍微降低温度以获得更稳定的输出
                top_p=0.9
            )

        caption = qwen_processor.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        print(caption)
        # 提取生成的描述部分（去掉输入提示）
        # 寻找assistant的回复部分
        if "assistant" in caption:
            caption = caption.split("assistant")[-1].strip()
        
        # 记录结果
        item["caption"] = caption
        item["image_size"] = f"{image.width}x{image.height}"
        item["error"] = None

    except Exception as e:
        print(f"❌ 处理失败：{image_path}，错误：{e}")
        item["caption"] = None
        item["error"] = str(e)

    results.append(item)

# 保存结果
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 描述生成完成，结果保存在：{output_json_path}")

