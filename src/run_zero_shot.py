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
    parser.add_argument('--config_path', type=str, default='configs/config_zero_shot.py', help='path to config file')
    parser.add_argument('--datasets', type=str, help="string of datasets", default="financial_phrasebank")
    return parser.parse_args()

if __name__ == "__main__":
    # get args
    args = get_args()
        
    # load config
    config = utils.load_config(args.config_path)
    datasets = args.datasets
    config['datasets'] = datasets
    
    model_path = utils.transformmodel_name2model_path(config['model_name'])
    config['model_path'] = model_path
    accelerator = Accelerator()
    device = accelerator.device
    my_method = method.get_method(method_name=config['method'], config=config, accelerator=accelerator)
    
    my_method.run(src_dataset_name=datasets)