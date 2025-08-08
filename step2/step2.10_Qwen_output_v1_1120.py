import argparse
import random
import numpy as np
import os
import torch
import json
import re
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import get_args_parser
path_args = get_args_parser().parse_args()

def get_args_parser():
    parser = argparse.ArgumentParser('Qwen-VL Inference with enhanced caption only', add_help=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--model_dir', default=path_args.model_dir+'/basic_model/Qwen2.5-VL-72B-Instruct')
    parser.add_argument('--json_path', default=path_args.project_path + "/knowledge/prompts-full_1120.json")
    parser.add_argument('--json_save_path', default=path_args.project_path + '/knowledge/step2_output_v1_1120.json')
    parser.add_argument('--img_dir', default=path_args.project_path + '/baseline/Qwen_baseline/img_1120/images')
    parser.add_argument('--florence_path', default=path_args.project_path + "/knowledge/bbox_florence_1120.json")
    parser.add_argument('--gdino_path', default=path_args.project_path + "/knowledge/bbox_gdino_1120.json")
    parser.add_argument('--qwen32b_path', default=path_args.project_path + '/knowledge/baseline_32b_1120.json')
    parser.add_argument('--florence_ft_path', default=path_args.project_path + "/knowledge/bbox_florence_ft_1120.json")
    parser.add_argument('--llmdet_path', default=path_args.project_path + "/knowledge/bbox_llmdet_1120.json")
    parser.add_argument('--qwen72b_path', default=path_args.project_path + '/knowledge/baseline_72b_1120.json')
    parser.add_argument('--spatial_path', default=path_args.project_path + "/knowledge/question_message.json")
    parser.add_argument('--subject_bbox_json_path', default=path_args.project_path + "/knowledge/qwen_subj_bbox_1120.json")
    parser.add_argument('--question_zh_path', default=path_args.project_path + "/knowledge/question_zh_translated.json")
    return parser


def extract_bbox(qwen_output):
    try:
        qwen_output = qwen_output.replace("\n", "").strip("`json ").strip()
        content = re.search(r"\[.*\]", qwen_output)
        if not content:
            return None
        raw = json.loads(content.group(0))
        if not raw or raw == [[0, 0], [0, 0]]:
            return None
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
            raw = raw[0]
        if len(raw) >= 2 and all(isinstance(pt, list) for pt in raw):
            pts = []
            for pt in raw[:2]:
                if isinstance(pt, list) and len(pt) >= 2:
                    pts.append([int(round(pt[0])), int(round(pt[1]))])
            if len(pts) == 2 and pts != [[0, 0], [0, 0]]:
                return pts
        if isinstance(raw, list) and len(raw) == 4 and all(isinstance(x, (int, float)) for x in raw):
            return [[int(round(raw[0])), int(round(raw[1]))],
                    [int(round(raw[2])), int(round(raw[3]))]]
        for candidate in raw:
            if isinstance(candidate, list):
                if len(candidate) == 4 and all(isinstance(x, (int, float)) for x in candidate):
                    return [[int(round(candidate[0])), int(round(candidate[1]))],
                            [int(round(candidate[2])), int(round(candidate[3]))]]
                if len(candidate) >= 2 and all(isinstance(x, (int, float)) for x in candidate):
                    pt1 = candidate[:2]
                    idx = raw.index(candidate) + 1
                    if idx < len(raw) and isinstance(raw[idx], list):
                        pt2 = raw[idx][:2]
                        return [[int(round(pt1[0])), int(round(pt1[1]))],
                                [int(round(pt2[0]), int(round(pt2[1])))]]
    except Exception:
        return None
    return None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_ref_result_map(ref_data):
    return {item["image_path"] + "|" + item["question"]: item["result"] for item in ref_data}


def build_spatial_map(spatial_data):
    return {item["image_path"]: item["description"] for item in spatial_data}


def build_subject_bbox_map(subject_bbox_data):
    result = {}
    for item in subject_bbox_data:
        key = item["image_path"] + "|" + item["question"]
        subject = item.get("subjects", "")
        try:
            raw_text = item.get("qwen_raw_output", "").replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw_text)
            bboxes = parsed[0] if isinstance(parsed, list) and parsed and isinstance(parsed[0], list) else []
        except Exception:
            bboxes = []
        result[key] = {"subject": subject, "bboxes": bboxes}
    return result


def build_question_zh_map(data):
    result = {}
    for item in data:
        key = item["image_path"] + "|" + item["question"]
        result[key] = item.get("question_zh", "")
    return result


def load_existing_results(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_incremental_result(path, new_entry):
    results = load_existing_results(path)
    results.append(new_entry)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main(args):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, max_pixels=2560 * 2560)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = load_json(args.json_path)
    florence_map = build_ref_result_map(load_json(args.florence_path))
    gdino_map = build_ref_result_map(load_json(args.gdino_path))
    qwen32b_map = build_ref_result_map(load_json(args.qwen32b_path))
    florence_ft_map = build_ref_result_map(load_json(args.florence_ft_path))
    llmdet_map = build_ref_result_map(load_json(args.llmdet_path))
    qwen72b_map = build_ref_result_map(load_json(args.qwen72b_path))
    spatial_map = build_spatial_map(load_json(args.spatial_path))
    subject_bbox_map = build_subject_bbox_map(load_json(args.subject_bbox_json_path))
    question_zh_map = build_question_zh_map(load_json(args.question_zh_path))

    for sample in tqdm(data, desc="Processing with Chinese question input"):
        image_name = sample["image_name"]
        image_rel_path = "images\\" + image_name
        question = sample["question"]
        setting = sample.get("setting", "")
        main_prompt = sample.get("main_prompt", "")
        spatial_caption = spatial_map.get(image_rel_path, "")

        key = image_rel_path + "|" + question
        question_zh = question_zh_map.get(key, "")
        subject_info = subject_bbox_map.get(key, {"subject": "", "bboxes": []})
        subject = subject_info["subject"]
        bboxes = subject_info["bboxes"]

        florence_result = florence_map.get(key, [])
        gdino_result = gdino_map.get(key, [])
        qwen32b_result = qwen32b_map.get(key, [])
        florence_ft_result = florence_ft_map.get(key, [])
        llmdet_result = llmdet_map.get(key, [])
        qwen72b_result = qwen72b_map.get(key, [])

        prompt_text = (
            f"You are an image recognition expert, specializing in object detection tasks in images. "
            f"I will provide you with scene information and detailed interpretations of the pictures. These are all generated by an image recognition model and can assist you in understanding the images (the information may be inaccurate or incomplete, and you need to use it with caution).\n"
            f"Scene understanding information: {setting}\n\n"
            f"Detailed image caption (from scene analysis): {main_prompt}\n\n"
            f"Some specific items contained in the image and the spatial relationships among them (from object relationship analysis, possibly incomplete): {spatial_caption}\n\n"
            f"Based on the image, the scene context, and the question, please output exactly one bounding box in the format: [[x1, y1], [x2, y2]]. You must always return a valid bounding box.\n\n"
            f"Consider factors like:\n"
            f"1.  The target object might be small or partially covered by other objects\n"
            f"2.  Focus on how well the bounding box captures the target object itself\n"
            f"3.  Consider spatial descriptions in the question (e.g., \"on the left\", \"in the center\", etc.)\n"
            f"4.  Pay attention to the description of the target, such as color, shape or action (the color mentioned in general questions is accurate without error), but the corresponding noun of the target itself may be wrong. This might be a problem of translation by a Chinese person with poor English proficiency\n"
            f"5.  If the objects to be detected in the question are multiple or plural, please enclose them all with a bounding box. If the object to be detected is singular, please enclose the entire object\n"
            f"6.  If there are multiple objects in the picture that match the description, please select the one you think best fits (evaluation criteria: stronger spatial relationship, more appropriate shape, more consistent color).\n"
            f"7.  Look for the bounding box that best covers the complete target object\n"
            f"8.  Some objects may be hard to see, so examine carefully\n\n"
            f"Next I will ask you a question\n"
            f"Question: {question}\n\n"
            f"The Chinese meaning of the question may be (for reference): {question_zh}\n\n"
            f"We have done some analysis on the question and extracted the target subject of the problem: {subject}\n"
            f"There may be areas in the image that are the target subject (the bbox I provided may be inaccurate or incomplete): {bboxes}\n\n"
            f"Some of the results were detected by the image detection model, and their ACC@0.5 scores were approximately 0.40-0.60. You can use these results as a reference. If you think they are inaccurate, you can also output a bounding box yourself.\n"
            f"The result of the image detection model: {qwen72b_result}, {qwen32b_result}, {florence_ft_result}, {florence_result}, {llmdet_result}, {gdino_result}\n"
            f"Please output the bounding box in JSON format."
        )

        image_full_path = os.path.join(args.img_dir, image_name)
        retry_count = 0
        final_output = None
        final_bbox = None

        while retry_count < 3:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_full_path},
                    {"type": "text", "text": prompt_text}
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                padding_side='left',
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = generated_ids[0][len(inputs.input_ids[0]):]
            raw_output = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

            bbox = extract_bbox(raw_output)
            if bbox:
                final_output = raw_output
                final_bbox = bbox
                break
            else:
                retry_count += 1

        if final_bbox:
            content = {
                "image_path": image_rel_path,
                "question": question,
                # "question_zh": question_zh,
                # "subject": subject,
                # "qwen_output": final_output,
                "result": final_bbox
            }
            save_incremental_result(args.json_save_path, content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run Qwen2.5-VL for bbox inference with references', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
