import sys
import os
import argparse

ICTL_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ICTL_ROOT_PATH)

import src.dataset as md

config = {}
# general
config["method"] = "few_shot"

config["domain"] = "cross_task_data"

config["gpus"] = ["0"]

config["model_name"] = "llama3.1-8b"

config["exp_name"] = "exps/few_shot-debug/"
# config['exp_name'] = 'exps/few_shot_ins-debug/'

config["bs"] = 1
config["load_in_8bit"] = False
config["use_cache"] = False

config["test_num"] = 500
# config['use_instruction'] = False
config["use_instruction"] = True
