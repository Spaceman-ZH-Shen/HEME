import json
import os
from PIL import Image
from config import get_args_parser
path_args = get_args_parser().parse_args()


def prep_ori():
    vg_rs_file = path_args.project_path + "/baseline/VG-RS-question.json"
    qwen_caption_file = path_args.project_path + "/knowledge/image_caption.json"
    florence_dino_file = path_args.project_path + "/knowledge/florence_dino_object_grounding.json"
    output_file = path_args.project_path + "/knowledge/prompts-full.json"

    files_to_check = [vg_rs_file, qwen_caption_file, florence_dino_file]
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return
    print("All files found. Starting combination process...")

    with open(vg_rs_file, 'r', encoding='utf-8') as f:
        vg_rs_data = json.load(f)
    with open(qwen_caption_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    with open(florence_dino_file, 'r', encoding='utf-8') as f:
        florence_data = json.load(f)

    qwen_lookup = {item['image_name']: item for item in qwen_data}
    florence_lookup = {os.path.basename(item['image_path']): item for item in florence_data}

    combined_data = []

    for vg_item in vg_rs_data:
        image_path = vg_item['image_path']
        image_name = os.path.basename(image_path)
        question = vg_item['question']
        qwen_item = qwen_lookup.get(image_name)
        florence_item = florence_lookup.get(image_name)

        if qwen_item and florence_item:
            main_scene = qwen_item['scene_scope']
            specific_setting = qwen_item['sub_scenes']
            setting = f"The scene of the picture is {main_scene} {specific_setting}. Assume that the image represents your first-person perspective."

            florence_result = florence_item['florence_result']
            bbox_descriptions = []

            for obj_name, bbox_list in florence_result.items():
                if bbox_list:  # Only include objects that have bounding boxes
                    for bbox in bbox_list:
                        if len(bbox) == 2 and len(bbox[0]) == 2 and len(bbox[1]) == 2:
                            x1, y1 = bbox[0]
                            x2, y2 = bbox[1]
                            bbox_descriptions.append(f"{obj_name}: [{x1:.1f},{y1:.1f}], [{x2:.1f},{y2:.1f}]")

            bbox_text = "; ".join(bbox_descriptions) if bbox_descriptions else "No bounding boxes available"

            caption = qwen_item['caption']
            if caption.startswith("SS\n\n"):
                caption = caption[4:]

            main_prompt = f"The detailed description of the image is: {caption}. Note that the bounding box([x1,y1](top left), [x2,y2](bottom right)) for some main objects are: {bbox_text}."

            combined_item = {
                'image_name': image_name,
                'question': question,
                'setting': setting,
                'main_prompt': main_prompt
            }

            combined_data.append(combined_item)
        else:
            print(f"Warning: Missing data for image {image_name}")
            partial_item = {
                'image_name': image_name,
                'question': question,
                'setting': "The scene of the picture is unknown. Assume that the image represents your first-person perspective.",
                'main_prompt': "Image description and bounding box information not available."
            }

            if qwen_item:
                main_scene = qwen_item['scene_scope']
                specific_setting = qwen_item['sub_scenes']
                partial_item[
                    'setting'] = f"The scene of the picture is {main_scene} {specific_setting}. Assume that the image represents your first-person perspective."

                caption = qwen_item['caption']
                if caption.startswith("SS\n\n"):
                    caption = caption[4:]
                partial_item[
                    'main_prompt'] = f"The detailed description of the image is: {caption}. Note that bounding box information is not available."

            combined_data.append(partial_item)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Combined data saved to {output_file}")
    print(f"✓ Total items processed: {len(combined_data)}")

    # Print a sample to verify format
    if combined_data:
        print("\n" + "=" * 50)
        print("SAMPLE OUTPUT:")
        print("=" * 50)
        sample = combined_data[0]
        print(f"Image Name: {sample['image_name']}")
        print(f"Question: {sample['question']}")
        print(f"Setting: {sample['setting']}")
        print(f"Main Prompt: {sample['main_prompt'][:200]}...")  # First 200 chars
        print("=" * 50)



def prep_1120():
    vg_rs_file = path_args.project_path + "/baseline/VG-RS-question.json"
    qwen_caption_file = path_args.project_path + "/knowledge/image_caption.json"
    florence_dino_file = path_args.project_path + "/knowledge/florence_dino_object_grounding.json"
    output_file = path_args.project_path + "/knowledge/prompts-full_1120.json"
    TARGET_SIZE          = (1120, 1120)

    for file_path in (vg_rs_file, qwen_caption_file, florence_dino_file):
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found!")
            return
    print("All files found. Starting combination process...")

    with open(vg_rs_file, 'r', encoding='utf-8') as f:
        vg_rs_data = json.load(f)
    with open(qwen_caption_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    with open(florence_dino_file, 'r', encoding='utf-8') as f:
        florence_data = json.load(f)

    qwen_lookup      = {item['image_name']: item for item in qwen_data}
    florence_lookup = {os.path.basename(item['image_path']): item for item in florence_data}

    combined_data = []

    for vg_item in vg_rs_data:
        image_path   = vg_item['image_path']
        image_name   = os.path.basename(image_path)
        question     = vg_item['question']
        qwen_item    = qwen_lookup.get(image_name)
        florence_item = florence_lookup.get(image_name)

        # compute scale factors for this image ↓↓↓
        with Image.open(image_path) as img:
            orig_w, orig_h = img.size
        scale_x = TARGET_SIZE[0] / orig_w  # ← added
        scale_y = TARGET_SIZE[1] / orig_h  # ← added

        if qwen_item and florence_item:
            main_scene       = qwen_item['scene_scope']
            specific_setting = qwen_item['sub_scenes']
            setting = (
                f"The scene of the picture is {main_scene} {specific_setting}. "
                "Assume that the image represents your first-person perspective."
            )

            florence_result = florence_item['florence_result']
            bbox_descriptions = []

            for obj_name, bbox_list in florence_result.items():
                if not bbox_list:
                    continue
                for bbox in bbox_list:
                    # original coords
                    x1, y1 = bbox[0]
                    x2, y2 = bbox[1]
                    # resize them ↓↓↓
                    xr1 = x1 * scale_x  # ← added
                    yr1 = y1 * scale_y  # ← added
                    xr2 = x2 * scale_x  # ← added
                    yr2 = y2 * scale_y  # ← added

                    bbox_descriptions.append(
                        f"{obj_name}: "
                        f"[{xr1:.1f},{yr1:.1f}], [{xr2:.1f},{yr2:.1f}]"
                    )

            bbox_text = "; ".join(bbox_descriptions) \
                        if bbox_descriptions else "No bounding boxes available"

            caption = qwen_item['caption']
            if caption.startswith("SS\n\n"):
                caption = caption[4:]

            main_prompt = (
                f"The detailed description of the image is: {caption}. "
                f"Note that (after resizing to {TARGET_SIZE[0]}×{TARGET_SIZE[1]}) "
                f"the bounding box([x1,y1](top-left), [x2,y2](bottom-right)) "
                f"for some main objects are: {bbox_text}."
            )

            combined_item = {
                'image_name': image_name,
                'question':    question,
                'setting':     setting,
                'main_prompt': main_prompt
            }

            combined_data.append(combined_item)

        else:
            print(f"Warning: Missing data for image {image_name}")
            # ... (your existing partial_item logic) ...
            # you can also choose to apply scaling there if you have any florence boxes

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Combined data saved to {output_file}")
    print(f"✓ Total items processed: {len(combined_data)}")
    if combined_data:
        print("\n" + "="*50)
        print("SAMPLE OUTPUT:")
        print("="*50)
        sample = combined_data[0]
        print(f"Image Name: {sample['image_name']}")
        print(f"Question:   {sample['question']}")
        print(f"Setting:    {sample['setting']}")
        print(f"Main Prompt (first 200 chars):\n{sample['main_prompt'][:200]}…")
        print("="*50)



def prep_1400():
    vg_rs_file = path_args.project_path + "/baseline/VG-RS-question.json"
    qwen_caption_file = path_args.project_path + "/knowledge/image_caption.json"
    florence_dino_file = path_args.project_path + "/knowledge/florence_dino_object_grounding.json"
    output_file = path_args.project_path + "/knowledge/prompts-full_1400.json"
    TARGET             = 1400  # final canvas size

    # 1) sanity check
    for p in (vg_rs_file, qwen_caption_file, florence_dino_file):
        if not os.path.exists(p):
            print(f"Error: '{p}' not found!")
            return
    print("All files found. Combining...")

    # 2) load inputs
    with open(vg_rs_file, 'r', encoding='utf-8') as f:
        vg_rs_data = json.load(f)
    with open(qwen_caption_file, 'r', encoding='utf-8') as f:
        qwen_data = json.load(f)
    with open(florence_dino_file, 'r', encoding='utf-8') as f:
        florence_data = json.load(f)

    # 3) build lookups
    qwen_lookup      = {item['image_name']: item for item in qwen_data}
    florence_lookup  = {
        os.path.basename(item['image_path']): item
        for item in florence_data
    }

    combined_data = []

    # 4) main loop
    for vg_item in vg_rs_data:
        img_path      = vg_item['image_path']
        img_name      = os.path.basename(img_path)
        question      = vg_item['question']
        qwen_item     = qwen_lookup.get(img_name)
        florence_item = florence_lookup.get(img_name)

        # --- compute uniform scale ---
        with Image.open(img_path) as img:
            orig_w, orig_h = img.size
        scale = TARGET / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        # pad_right = TARGET - new_w  # if you need it for actual image padding
        # pad_bot   = TARGET - new_h

        if qwen_item and florence_item:
            # build the “setting” string
            main_scene       = qwen_item['scene_scope']
            specific_setting = qwen_item['sub_scenes']
            setting = (
                f"The scene of the picture is {main_scene} {specific_setting}. "
                "Assume that the image represents your first-person perspective."
            )

            # scale each florence bbox
            florence_result = florence_item['florence_result']
            bbox_descs = []
            for obj_name, bbox_list in florence_result.items():
                if not bbox_list:
                    continue
                for bbox in bbox_list:
                    # ensure [[x1,y1],[x2,y2]]
                    if len(bbox) == 2 and len(bbox[0]) == 2 and len(bbox[1]) == 2:
                        x1, y1 = bbox[0]
                        x2, y2 = bbox[1]
                        xr1, yr1 = x1 * scale, y1 * scale
                        xr2, yr2 = x2 * scale, y2 * scale
                        bbox_descs.append(
                            f"{obj_name}: [{xr1:.1f},{yr1:.1f}], [{xr2:.1f},{yr2:.1f}]"
                        )

            bbox_text = (
                "; ".join(bbox_descs)
                if bbox_descs else "No bounding boxes available"
            )

            # clean up caption
            caption = qwen_item['caption']
            if caption.startswith("SS\n\n"):
                caption = caption[4:]

            # assemble prompt
            main_prompt = (
                f"The detailed description of the image is: {caption}. "
                f"Note that after aspect-ratio-preserving resize (longest side→{TARGET}px) "
                f"and padding to {TARGET}×{TARGET}, the bounding box "
                f"([x1,y1] top-left, [x2,y2] bottom-right) for some main objects are: "
                f"{bbox_text}."
            )

            combined_data.append({
                'image_name':  img_name,
                'question':    question,
                'setting':     setting,
                'main_prompt': main_prompt
            })

        else:
            # fallback when missing data
            print(f"Warning: Missing data for {img_name}")
            partial = {
                'image_name': img_name,
                'question':   question,
                'setting':    "The scene of the picture is unknown. Assume first-person perspective.",
                'main_prompt': "Image description and bounding box information not available."
            }
            if qwen_item:
                # you can similarly scale any boxes here if you have them
                cap = qwen_item['caption']
                if cap.startswith("SS\n\n"):
                    cap = cap[4:]
                partial['setting'] = (
                    f"The scene of the picture is {qwen_item['scene_scope']} "
                    f"{qwen_item['sub_scenes']}. Assume first-person perspective."
                )
                partial['main_prompt'] = (
                    f"The detailed description of the image is: {cap}. "
                    "Note that bounding box info is not available."
                )

            combined_data.append(partial)

    # 5) write out
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {len(combined_data)} items to {output_file}")


prep_1120()
prep_1400()
