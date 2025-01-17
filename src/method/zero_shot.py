import sys
import os
import json

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.method.basemethod import BaseMethod
from src.utils import utils

class ZeroShot(BaseMethod):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        self.device = self.accelerator.device
        
        test_zeroshot_result = self.test_evaluator.evaluate(self.tokenizer, self.model, use_demonstration=False)
        self.result_dict['test_result']['method'].append(test_zeroshot_result)
        print(f'Test zero-shot result: {test_zeroshot_result}\n')
        
        with open(self.save_dir + '/result_dict.json', 'w') as f:
            json.dump(self.result_dict, f, indent=4)