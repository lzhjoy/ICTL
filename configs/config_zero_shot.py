import sys
import os
import argparse

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ICTL_ROOT_PATH)

import src.dataset as md

config = {}
# general
config['exp_name'] = 'exps/zero_shot_ins-debug'
config['method'] = 'zero_shot'
config['domain'] = 'cross_task_data'
config['gpus'] = ['0']
config['model_name'] = 'llama3.1-8b'

config['bs'] = 1
config['load_in_8bit'] = False
config['use_cache'] = True

config['shot_num'] = 0
config['test_num'] = 500
config['use_instruction'] = True