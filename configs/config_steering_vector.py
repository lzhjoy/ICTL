import sys
import os
import argparse

ICTL_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ICTL_ROOT_PATH)

import src.dataset as md

# 单向量注入配置
config = {
    'module': 'hidden',
    'layer': 'mid',
    'inject_method': 'add',
    'inject_pos': 'all',
    'strength': 1, # 固定注入强度 
    'tok_pos': 'last',
    'post_fuse_method': 'mean',
    'init_value': 0.1, # 可学习注入强度初始值
}

# general
config["method"] = "ours"

config["domain"] = "cross_task_data"


config["gpus"] = ["0"]

config["model_name"] = "llama3.1-8b"

# config["exp_name"] = "exps/few_shot_ins-debug/" + config["shot_method"]
config['exp_name'] = 'exps/steering_vector-debug/'

config["bs"] = 1
config["load_in_8bit"] = False
config["use_cache"] = False

config["test_num"] = 500
# config['use_instruction'] = False
config["use_instruction"] = True