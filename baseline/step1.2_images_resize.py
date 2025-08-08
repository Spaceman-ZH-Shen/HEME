import os
from PIL import Image
from tqdm import tqdm
from config import get_args_parser
path_args = get_args_parser().parse_args()

from img_size_transfer_tool.img_resize_1120 import images_resize_1120
from img_size_transfer_tool.img_resize_1400 import images_resize_1400

input_dir = path_args.project_path + "/baseline/images"
output_dir_1120 = path_args.project_path + "/baseline/Qwen_baseline/img_1120/images"
output_dir_1400 = path_args.project_path + "/baseline/Qwen_baseline/img_1400/images"

images_resize_1120(input_dir,output_dir_1120)
images_resize_1400(input_dir,output_dir_1400)