import sys
import os
import argparse

ICTL_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ICTL_ROOT_PATH)

import src.dataset as md

config = {}
# general
config["method"] = "few_shot"

# config["shot_method"] = "random"
# config['shot_method'] = 'dpp'
config['shot_method'] = 'topk'

config["domain"] = "cross_task_data"

config["gpus"] = ["0"]

config["model_name"] = "llama3.1-8b"

# config["exp_name"] = "exps/few_shot_ins-debug/" + config["shot_method"]
config['exp_name'] = 'exps/few_shot-debug/'

config["bs"] = 1
config["load_in_8bit"] = False
config["use_cache"] = False

config["shot_num"] = 4
config["test_num"] = 500
config['use_instruction'] = False
# config["use_instruction"] = True
