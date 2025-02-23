import sys
import os
import json

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.method.basemethod import BaseMethod
from src.utils import utils
from src.utils import wrapper
class SteeringVector(BaseMethod):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        self.device = self.accelerator.device
  
        # print("running evaluation")
        test_steering_result = self.test_evaluator.evaluate(self.tokenizer, self.model, use_demonstration=False,)
        self.result_dict['test_result']['ours'].append(test_steering_result)
        print(f'Test steering result: {test_steering_result}\n')
        
        with open(self.save_dir + '/result_dict.json', 'w') as f:
            json.dump(self.result_dict, f, indent=4)
    
    def construct_input_with_demonstration(self, question, demonstration, use_instruction=False):
        pass

    def construct_input_without_demonstration(self, question, use_instruction=False):
        pass

    def extract_steering_vector(self):
        pass

    def inject_steering_vector(self):
        pass