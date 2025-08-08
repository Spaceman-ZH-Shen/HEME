import argparse
import random
from pathlib import Path
import numpy as np
import os, cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import torch
import ast
import json
from tqdm import tqdm
from config import get_args_parser

path_args = get_args_parser().parse_args()

def get_args_parser():
    parser = argparse.ArgumentParser('Visual grounding', add_help=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model_dir', default=path_args.model_dir + '/Qwen2.5-VL-32B-Instruct/')
    parser.add_argument('--json_path', default=path_args.project_path + "/baseline/VG-RS-question.json")
    parser.add_argument('--json_save_path', default=path_args.project_path + '/knowledge/baseline_32b_1400.json')
    return parser


def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def read_json_and_extract_fields(file_path=path_args.project_path + "/baseline/VG-RS-question.json"):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="auto"
    ).eval()

    seed = args.seed
    print('seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    img_path = r'./images/'
    data_infer = read_json_and_extract_fields(args.json_path)
    batch_size = args.batch_size
    max_pixels = 2560 * 2560
    processor = AutoProcessor.from_pretrained(args.model_dir, max_pixels=max_pixels)

    content_list = []
    total_batches = len(data_infer) // batch_size

    for i in tqdm(range(total_batches), desc="Processing batches"):
        messages_list = []
        text_query_list = []
        image_name_list = []

        for i_batch in range(batch_size):
            image_name = data_infer[i * batch_size + i_batch].get('image_path').split('images\\')[1]
            image_name_list.append(image_name)
            text_query = data_infer[i * batch_size + i_batch].get('question')
            text_query_list.append(text_query)

            image_path = os.path.join(img_path, str(image_name.lower()))
            # === 获取图像原始尺寸 ===
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                print(f"[Warning] Failed to read image: {image_path}")
                continue
            height, width = image_cv.shape[:2]
            size_note = f"The original image size is (height={height}, width={width}). Note: The input image might have been resized by the model processor."

            # === 构建Prompt，插入原图尺寸信息 ===
            prompt_text = (
                f"{size_note}\n"
                f"Please provide the bounding box coordinate of the region this sentence describes: {text_query} "
                f"and output it in JSON format"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt_text}
                    ],
                }
            ]
            messages_list.append(messages)

        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            padding_side='left',
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for i_batch in range(batch_size):
            bounding_boxes = parse_json(output_text[i_batch])
            try:
                json_output = ast.literal_eval(bounding_boxes)
            except Exception as e:
                end_idx = bounding_boxes.rfind('"}') + len('"}')
                truncated_text = bounding_boxes[:end_idx] + "]"
                try:
                    json_output = ast.literal_eval(truncated_text)
                except Exception as e:
                    print(['[Unable to detect samples]', f'{i}'])
                    print(image_path)
                    continue

            for j_index, bounding_box in enumerate(json_output):
                if j_index >= 1:
                    continue
                try:
                    len(bounding_box["bbox_2d"]) != 4
                except (KeyError, TypeError):
                    continue
                try:
                    abs_y1 = bounding_box["bbox_2d"][1]
                    abs_x1 = bounding_box["bbox_2d"][0]
                    abs_y2 = bounding_box["bbox_2d"][3]
                    abs_x2 = bounding_box["bbox_2d"][2]
                except IndexError:
                    continue

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                content = {
                    "image_path": 'images\\' + image_name_list[i_batch],
                    'question': text_query_list[i_batch],
                    "result": [[abs_x1, abs_y1], [abs_x2, abs_y2]]
                }
                content_list.append(content)

    with open(args.json_save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(content_list, ensure_ascii=False, indent=2) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Infer result', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
