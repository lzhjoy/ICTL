import argparse
import os
import sys
import pdb
import torch

from accelerate import Accelerator

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ICTL_ROOT_PATH)

import src.dataset as md
import src.method as method
from src.utils import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_few_shot.py', help='path to config file')
    parser.add_argument('--src_datasets', type=str, help="string of src_datasets", default="financial_phrasebank")
    parser.add_argument('--tar_datasets', type=str, help="string of tar_datasets", default="agnews")
    parser.add_argument('--shot_num', type=int, help="number of shot", default=2)
    parser.add_argument('--shot_method', type=str, help="method of shot", default="dpp")
    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = get_args()
        
    # load config
    config = utils.load_config(args.config_path)
    src_datasets = args.src_datasets
    tar_datasets = args.tar_datasets
    shot_num = args.shot_num
    shot_method = args.shot_method
    config['src_datasets'] = src_datasets
    config['tar_datasets'] = tar_datasets
    config['shot_num'] = shot_num
    config['shot_method'] = shot_method
    
    model_path = utils.transformmodel_name2model_path(config['model_name'])
    config['model_path'] = model_path
    accelerator = Accelerator()
    device = accelerator.device
    my_method = method.get_method(method_name=config['method'], config=config, accelerator=accelerator)
    
    my_method.run(src_dataset_name=src_datasets, tar_dataset_name=tar_datasets)