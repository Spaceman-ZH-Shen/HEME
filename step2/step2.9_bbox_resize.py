import json
from config import get_args_parser
path_args = get_args_parser().parse_args()
from img_size_transfer_tool.bbox_size_converter import transfer_ori_to_1120, transfer_ori_to_1400, transfer_1120_to_ori, transfer_1400_to_ori

florence_path = path_args.project_path + "/baseline/baseline_results/baseline_florence2.json"
florence_ft_path = path_args.project_path + "/baseline/baseline_results/baseline_florence2_ft.json"
g_dino_path = path_args.project_path + "/baseline/baseline_results/baseline_gdino.json"
llmdet_path = path_args.project_path + "/baseline/baseline_results/baseline_llmdet.json"

qwen_32_1120_path = path_args.project_path + '/knowledge/baseline_32b_1120.json'
qwen_32_1400_path = path_args.project_path + '/knowledge/baseline_32b_1400.json'
qwen_72_1120_path = path_args.project_path + '/knowledge/baseline_72b_1120.json'
qwen_72_1400_path = path_args.project_path + '/knowledge/baseline_72b_1400.json'

transfer_1120_to_ori(bbox_input_path=qwen_32_1120_path, output_path=path_args.project_path + '/knowledge/baseline_32b_1120_ori.json')
transfer_1120_to_ori(bbox_input_path=qwen_72_1120_path, output_path=path_args.project_path + '/knowledge/baseline_72b_1120_ori.json')
transfer_1400_to_ori(bbox_input_path=qwen_32_1400_path, output_path=path_args.project_path + '/knowledge/baseline_32b_1400_ori.json')
transfer_1400_to_ori(bbox_input_path=qwen_72_1400_path, output_path=path_args.project_path + '/knowledge/baseline_72b_1400_ori.json')

transfer_ori_to_1120(bbox_input_path=florence_path, output_path=path_args.project_path + "/knowledge/bbox_florence_1120.json")
transfer_ori_to_1120(bbox_input_path=florence_ft_path, output_path=path_args.project_path + "/knowledge/bbox_florence_ft_1120.json")
transfer_ori_to_1120(bbox_input_path=g_dino_path, output_path=path_args.project_path + "/knowledge/bbox_gdino_1120.json")
transfer_ori_to_1120(bbox_input_path=llmdet_path, output_path=path_args.project_path + "/knowledge/bbox_llmdet_1120.json")
transfer_ori_to_1120(bbox_input_path=llmdet_path, output_path=path_args.project_path + "/knowledge/bbox_llmdet_1120.json")
transfer_ori_to_1120(bbox_input_path=path_args.project_path + '/knowledge/baseline_32b_1400_ori.json', output_path=path_args.project_path + '/knowledge/baseline_32b_1400_1120.json')
transfer_ori_to_1120(bbox_input_path=path_args.project_path + '/knowledge/baseline_72b_1400_ori.json', output_path=path_args.project_path + '/knowledge/baseline_72b_1400_1120.json')

transfer_ori_to_1400(bbox_input_path=florence_path, output_path=path_args.project_path + "/knowledge/bbox_florence_1400.json")
transfer_ori_to_1400(bbox_input_path=florence_ft_path, output_path=path_args.project_path + "/knowledge/bbox_florence_ft_1400.json")
transfer_ori_to_1400(bbox_input_path=g_dino_path, output_path=path_args.project_path + "/knowledge/bbox_gdino_1400.json")
transfer_ori_to_1400(bbox_input_path=llmdet_path, output_path=path_args.project_path + "/knowledge/bbox_llmdet_1400.json")
transfer_ori_to_1400(bbox_input_path=path_args.project_path + '/knowledge/baseline_32b_1120_ori.json', output_path=path_args.project_path + '/knowledge/baseline_32b_1120_1400.json')
transfer_ori_to_1400(bbox_input_path=path_args.project_path + '/knowledge/baseline_72b_1120_ori.json', output_path=path_args.project_path + '/knowledge/baseline_72b_1120_1400.json')

