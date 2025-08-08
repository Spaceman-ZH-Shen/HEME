import argparse
import os


def get_args_parser():
    parser = argparse.ArgumentParser('Sup_model_path', add_help=False)
    parser.add_argument('--model_dir', default=r'/model_path')
    parser.add_argument('--project_path', default=os.path.dirname(__file__))
    return parser


