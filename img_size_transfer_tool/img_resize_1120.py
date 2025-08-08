import os
from PIL import Image
from tqdm import tqdm
from config import get_args_parser
path_args = get_args_parser().parse_args()


def images_resize_1120(input_dir, output_dir):
    """将 input_dir 中的图片调整为 1120x1120，并保存到 output_dir"""
    target_size = (1120, 1120)

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 支持的图片扩展名
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # 遍历图片并处理
    for fname in tqdm(os.listdir(input_dir), desc="Resizing images"):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_exts:
            continue  # 跳过非图片文件

        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")  # 避免PNG带alpha通道
                img_resized = img.resize(target_size, Image.BILINEAR)
                img_resized.save(output_path, quality=100)
        except Exception as e:
            print(f"[Error] Failed to process {fname}: {e}")


