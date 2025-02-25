import sys
import os
import json

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.method.basemethod import BaseMethod
from src.utils import utils

class FewShot(BaseMethod):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        self.device = self.accelerator.device

        # print("running evaluation")
        test_fewshot_result = self.test_evaluator.evaluate(self.tokenizer, self.model, use_demonstration=True,)
        self.result_dict['test_result']['few_shot'].append(test_fewshot_result)
        print(f'Test few-shot result: {test_fewshot_result}\n')
        
        with open(self.save_dir + '/result_dict.json', 'w') as f:
            json.dump(self.result_dict, f, indent=4)