import os
import json
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor as QwenProcessor
from qwen_vl_utils import process_vision_info
from typing import Dict, List, Tuple, Optional
import tempfile
from config import get_args_parser
path_args = get_args_parser().parse_args()

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32
if device == "cuda":
    if torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        print("Using bfloat16 precision (supported by hardware)")
    else:
        torch_dtype = torch.float16
        print("Using float16 precision (bfloat16 not supported)")
else:
    print("Using float32 precision (CPU)")

# 模型路径
qwen_path = path_args.model_dir + '/basic_model/Qwen2.5-VL-72B-Instruct'

# 加载模型
print("Loading Qwen2.5-VL model...")
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_path,
    torch_dtype=torch_dtype,
    device_map="auto",
    attn_implementation="flash_attention_2"
).eval()
qwen_processor = QwenProcessor.from_pretrained(qwen_path)

# 文件路径
INPUT_JSON_FILE = path_args.project_path + '/knowledge/step3_candidate_boxes.json'
IMAGE_BASE_PATH = path_args.project_path + '/baseline/Qwen_baseline/img_1120/images'
OUTPUT_JSON_PATH = path_args.project_path + "/knowledge/step3_output_1120_large_model.json"

PROMPTS_FULL_JSON = path_args.project_path + "/knowledge/prompts-full_1120.json"
SPATIAL_DESC_JSON = path_args.project_path + "/knowledge/question_message.json"

# 可选择的 bbox key 列表
BBOX_KEYS = [
    "qwen_72b_bbox",
    "step2_1120_bbox",
    "step2_1400_bbox",
    "step2_1400v2_bbox",
]

def load_json_file(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_additional_data():
    prompts_data = {}
    spatial_data = {}

    # 加载prompts-full.json
    if os.path.exists(PROMPTS_FULL_JSON):
        prompts_list = load_json_file(PROMPTS_FULL_JSON)
        for item in prompts_list:
            prompts_data[item['image_name']] = {
                'setting': item.get('setting', ''),
                'main_prompt': item.get('main_prompt', ''),
            }

    # 加载img_objs_spatial_description.json
    if os.path.exists(SPATIAL_DESC_JSON):
        spatial_list = load_json_file(SPATIAL_DESC_JSON)
        for item in spatial_list:
            image_path = item['image_path'].replace('\\', '/')
            image_name = os.path.basename(image_path)
            spatial_data[image_name] = item.get('description', '')

    return prompts_data, spatial_data

def normalize_bbox(bbox: List[List[int]]) -> List[List[int]]:
    (x1, y1), (x2, y2) = bbox
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [[x1, y1], [x2, y2]]

def draw_bounding_boxes_with_labels(image_path: str, bboxes: Dict[str, List]) -> str:
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    colors = {
        'qwen_72b_bbox': 'darkred',
        'step2_1120_bbox': 'blue',
        'step2_1400_bbox': 'orange',
        'step2_1400v2_bbox': 'brown',
    }
    model_names = {
        'qwen_72b_bbox': 'Qwen-72B',
        'step2_1120_bbox': 'Step2-1120',
        'step2_1400_bbox': 'Step2-1400',
        'step2_1400v2_bbox': 'Step2-1400v2',
    }
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for bbox_key, bbox in bboxes.items():
        if bbox and len(bbox) >= 2:
            bbox = normalize_bbox(bbox)
            x1, y1 = bbox[0]
            x2, y2 = bbox[1]
            color = colors.get(bbox_key, 'yellow')
            model_name = model_names.get(bbox_key, bbox_key)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            label = f"{model_name}"
            if font:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            else:
                text_width, text_height = 80, 20
            label_x = x1
            label_y = max(0, y1 - text_height - 5)
            draw.rectangle([label_x, label_y, label_x + text_width + 10, label_y + text_height + 5], fill=color, outline=color)
            draw.text((label_x + 5, label_y + 2), label, fill='white', font=font)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(temp_file.name, 'JPEG', quality=95)
    return temp_file.name

def build_bbox_selection_messages(image_path: str, question: str, available_bboxes: Dict[str, List],
                                  setting: str = "", main_prompt: str = "", spatial_desc: str = ""):
    """构建包含 bbox 坐标和额外上下文信息的消息"""
    color_map = {
        'qwen_72b_bbox': 'darkred',
        'step2_1120_bbox': 'blue',
        'step2_1400_bbox': 'orange',
        'step2_1400v2_bbox': 'brown',
    }
    name_map = {
        'qwen_72b_bbox': 'Qwen-72B',
        'step2_1120_bbox': 'Step2-1120',
        'step2_1400_bbox': 'Step2-1400',
        'step2_1400v2_bbox': 'Step2-1400v2',
    }

    bbox_descriptions = []
    for bbox_key, bbox in available_bboxes.items():
        if bbox:
            color = color_map.get(bbox_key, 'yellow')
            name = name_map.get(bbox_key, bbox_key)
            coord_str = f"[[{bbox[0][0]}, {bbox[0][1]}], [{bbox[1][0]}, {bbox[1][1]}]]"
            bbox_descriptions.append(f"- {name} (marked in {color}): {coord_str}")

    bbox_text = "\n".join(bbox_descriptions)

    # 组装上下文
    context_info = ""
    if setting:
        context_info += f"**Scene Setting:** {setting}\n\n"
    if main_prompt:
        context_info += f"**Detailed Image Description:** {main_prompt}\n\n"
    if spatial_desc:
        context_info += f"**Spatial Relationships:** {spatial_desc}\n\n"

    prompt = f"""You are given an image with bounding boxes from different models, each attempting to localize the object described in the question.

{context_info}**Question/Description:** \"{question}\"\n\n**Bounding Boxes from Different Models:**\n{bbox_text}\n\n**Your task:**\n1. **First**, carefully evaluate whether the bounding box from **Step2-1400** accurately and precisely localizes the target object described in the question.\n2. If **Step2-1400 is clearly correct**, select it as the final answer.\n3. If **Step2-1400 is incorrect or significantly less accurate**, compare the results from the other models (**Qwen-72B, Step2-1120, Step2-1400v2**) and choose the one that **most accurately** captures the target object.\n\n**Evaluation criteria:**\n- How well the bounding box encloses the correct object described in the question.\n- Whether the object is small, occluded, or spatially situated (e.g., \"on the left\", \"in the center\", etc.).\n- Precision of the bounding box in covering the object (not too loose or too tight).\n- Use the provided scene setting and spatial relationship information to better understand the context.\n- Avoid selecting placeholder or empty boxes.\n\n**Output format (choose only one):**\nSELECTED: [Model Name]  \n(e.g., SELECTED: Step2-1120)\n\nDo **not** explain your reasoning or include any additional text."""

    return [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]

def parse_selected_model(response: str) -> Optional[str]:
    try:
        for line in response.split('\n'):
            if 'SELECTED:' in line.upper():
                selected = line.split('SELECTED:')[-1].strip().upper()
                if 'QWEN-72B' in selected:
                    return 'qwen_72b_bbox'
                if 'STEP2-1400V2' in selected:
                    return 'step2_1400v2_bbox'
                if 'STEP2-1400' in selected:
                    return 'step2_1400_bbox'
                if 'STEP2-1120' in selected:
                    return 'step2_1120_bbox'
        # fallback: majority voting by keyword occurrence
        model_counts = {
            'qwen_72b_bbox': response.upper().count('QWEN-72B'),
            'step2_1400v2_bbox': response.upper().count('STEP2-1400V2'),
            'step2_1400_bbox': response.upper().count('STEP2-1400'),
            'step2_1120_bbox': response.upper().count('STEP2-1120'),
        }
        max_count = max(model_counts.values())
        return next((k for k, v in model_counts.items() if v == max_count), None)
    except Exception as e:
        print(f"解析选择模型时出错: {e}")
        return None
def select_best_bbox(image_with_boxes_path: str, question: str, available_bboxes: Dict[str, List],
                    setting: str = "", main_prompt: str = "", spatial_desc: str = "") -> Tuple[str, str]:
    """选择最佳边界框，包含额外上下文信息"""
    try:
        messages = build_bbox_selection_messages(
            image_with_boxes_path, question, available_bboxes, 
            setting, main_prompt, spatial_desc
        )
        text_input = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = qwen_processor(text=[text_input], images=image_inputs, videos=video_inputs, return_tensors="pt", padding=True).to("cuda")

        for key, tensor in inputs.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"Input tensor '{key}' contains NaN or Inf values.")

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=qwen_model.dtype):
                outputs = qwen_model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.1, top_p=0.9)

        response = qwen_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        selected_model = parse_selected_model(response)
        return response, selected_model
    except torch.cuda.CudaError as cuda_err:
        torch.cuda.empty_cache()
        return f"CUDA error: {str(cuda_err)}", None
    except Exception as e:
        return f"Error during evaluation: {str(e)}", None

def main():
    print("Loading bbox comparison data...")
    print(BBOX_KEYS)
    bbox_data = load_json_file(INPUT_JSON_FILE)
    
    # 加载额外数据
    print("Loading additional context data...")
    prompts_data, spatial_data = load_additional_data()
    print(f"Loaded prompts data for {len(prompts_data)} images")
    print(f"Loaded spatial data for {len(spatial_data)} images")
    
    open(OUTPUT_JSON_PATH, 'w', encoding='utf-8').close()
    results = []
    temp_files_to_cleanup = []

    try:
        for entry in tqdm(bbox_data, desc="选择最佳边界框", ncols=100):
            image_name = entry['image_name']
            question = entry['question']
            available_bboxes = {}
            for key in BBOX_KEYS:
                bbox = entry.get(key)
                if bbox and len(bbox) >= 2:
                    available_bboxes[key] = normalize_bbox(bbox)

            if len(available_bboxes) < 2:
                print(f"跳过 {image_name} - 边界框数量不足")
                continue

            try:
                full_image_path = os.path.join(IMAGE_BASE_PATH, image_name)
                if not os.path.exists(full_image_path):
                    print(f"图像不存在: {full_image_path}")
                    continue

                # 获取额外的上下文信息
                prompt_info = prompts_data.get(image_name, {})
                setting = prompt_info.get('setting', '')
                main_prompt = prompt_info.get('main_prompt', '')
                spatial_desc = spatial_data.get(image_name, '')


                image_with_boxes_path = draw_bounding_boxes_with_labels(full_image_path, available_bboxes)
                temp_files_to_cleanup.append(image_with_boxes_path)

                qwen_evaluation, selected_model_key = select_best_bbox(
                    image_with_boxes_path, question, available_bboxes,
                    setting, main_prompt, spatial_desc
                )

                if selected_model_key and selected_model_key in available_bboxes:
                    final_bbox = available_bboxes[selected_model_key]
                    selected_model_name = selected_model_key
                else:
                    selected_model_name = list(available_bboxes.keys())[0]
                    final_bbox = available_bboxes[selected_model_name]

                result_entry = {
                    "image_name": image_name,
                    "question": question,
                    "setting": setting,
                    "spatial_description": spatial_desc,
                    "selected_model": selected_model_name,
                    "final_bbox": final_bbox,
                    "available_models": list(available_bboxes.keys()),
                    "qwen_evaluation": qwen_evaluation,
                    "has_additional_context": bool(setting and main_prompt and spatial_desc)
                }
                results.append(result_entry)
                with open(OUTPUT_JSON_PATH, 'a', encoding='utf-8') as f_out:
                    f_out.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

            except Exception as e:
                print(f"处理 {image_name} 时出错: {str(e)}")

    finally:
        for temp_file in temp_files_to_cleanup:
            try:
                os.unlink(temp_file)
            except:
                pass

    # 统计信息
    total_processed = len(results)
    with_context = sum(1 for r in results if r['has_additional_context'])
    print(f"已完成 {total_processed} 条数据处理")
    print(f"结果保存在: {OUTPUT_JSON_PATH}")

    ##########################################
    input_file = path_args.project_path + "/knowledge/step3_output_1120_large_model.json"
    image_folder = 'images'
    output_json_path = path_args.project_path + "/knowledge/step3_output_1120_large_model.json"  # 你想保存的新文件路径

    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # 逐行解析为 JSON 对象
    data = [json.loads(line.strip()) for line in lines if line.strip()]

    # 转换数据格式
    converted = []
    for item in data:
        new_item = {
            "image_path": os.path.join(image_folder, item["image_name"]).replace("/", "\\"),
            "question": item["question"],
            "result": item["final_bbox"]
        }
        converted.append(new_item)

    # 保存为新 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2)

if __name__ == "__main__":
    main()