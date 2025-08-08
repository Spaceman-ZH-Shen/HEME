import argparse
import os
import torch
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from config import get_args_parser
path_args = get_args_parser().parse_args()

def get_args_parser():
    parser = argparse.ArgumentParser('Visual grounding', add_help=False)  # 添加 add_help=False
    parser.add_argument('--model_dir', default=path_args.model_dir+'/basic_model/Qwen2.5-VL-72B-Instruct/')
    parser.add_argument('--image_dir', default=path_args.project_path+'/baseline/images')
    parser.add_argument('--json_save_path', default=path_args.project_path+'/knowledge/img_scene.json')
    return parser

def build_prompt():
    return (
        "You are an expert in visual scene understanding.\n\n"
        "Given an input image, your task is to:\n"
        "1. First determine the **scene scope** of the image. The scene scope is the broad category that describes the general environment in which the image was taken. Choose one from the following three options only:\n"
        "- \"city\": outdoor or public urban environments (e.g. roads, stations, restaurants)\n"
        "- \"home\": private indoor environments in a house or apartment\n"
        "- \"education\": places related to learning or school environments\n\n"
        "2. After deciding the scene scope, identify the **two most likely fine-grained scenes** (i.e., sub-scenes) from the following list associated with that scope:\n"
        "- If the scene scope is \"city\", choose from: \"street\", \"station\", \"airport\", \"restaurant\", \"scenic spot\"\n"
        "- If the scene scope is \"home\", choose from: \"living room\", \"bedroom\", \"bathroom\", \"kitchen\"\n"
        "- If the scene scope is \"education\", choose from: \"playground\", \"classroom\", \"library\"\n\n"
        "Please ensure the selected fine-grained scenes belong to the same scope.\n\n"
        "Respond only in JSON format like this:\n"
        "{\n"
        "  \"scene_scope\": \"city\",\n"
        "  \"top2_sub_scenes\": [\"station\", \"street\"]\n"
        "}"
    )



def parse_model_output(output_text):
    try:
        start = output_text.find("{")
        end = output_text.rfind("}") + 1
        json_str = output_text[start:end]
        return json.loads(json_str)
    except Exception as e:
        print("Failed to parse output:", output_text)
        return None


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_dir, max_pixels=2560 * 2560)

    image_list = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    results = []
    prompt = build_prompt()

    for image_name in tqdm(image_list, desc="Processing Images"):
        image_path = os.path.join(args.image_dir, image_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ],
            }
        ]

        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=text_input,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(device)

        outputs = model.generate(**inputs, max_new_tokens=512)
        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_model_output(response)
        print(image_name)
        print(parsed)

        if parsed:
            results.append({
                "image_name": image_name,
                "scene_scope": parsed.get("scene_scope", "unknown"),
                "top2_sub_scenes": parsed.get("top2_sub_scenes", [])
            })

    with open(args.json_save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {args.json_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene Scope Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
