import os
from PIL import Image
from tqdm import tqdm
from config import get_args_parser
path_args = get_args_parser().parse_args()



def images_resize_1400(input_dir, output_dir):
    """将 input_dir 中的图片调整为最长边 1400，并保存到 output_dir"""
    target_size = 1400

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片扩展名
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for fname in tqdm(os.listdir(input_dir), desc="Processing images"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue  # 跳过非图片文件

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                w, h = img.size

                # 计算缩放比例，使最长边变为 1400
                scale = target_size / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_img = img.resize((new_w, new_h), Image.BILINEAR)

                # 创建1400×1400的白底图像，并将缩放后的图像粘贴进去
                new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
                new_img.paste(resized_img, (0, 0))  # 左上角对齐，右/下填充空白

                new_img.save(output_path, quality=100)

        except Exception as e:
            print(f"[Error] Failed to process {fname}: {e}")


