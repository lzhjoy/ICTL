from datasets import load_dataset
import random
import os
import sys
from tqdm import tqdm

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.dataset.basetask import BaseTask
from src.utils.utils import write_jsonl

class ARC_CHALLENGE(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Question: {sentence}\nAnswer:',
            'ans': '{label}',
            'options': ["A", "B", "C", "D"],
            'format': ['Question:', 'Answer:'],
            'instruction': 'Given a question answering task from the 3rd to 9th-grade science exam. The question contains four options "A.", "B.", "C." and "D." Select the most appropriate choice that answers the question'
        }
        return template
    
    def apply_template(self, data):
        """
        PS: label should always be an integer and can be used to index the options
        """
        template = self.get_dmonstration_template()
        input_template = template['input']
        ans_template = template['ans']
        options = template['options']
        input_str = input_template.replace("{sentence}", data["sentence"])
        # answers can have multiple options and is a list
        answer_str = [ans_template.replace("{label}", options[i]) for i in range(len(options))]
        label = data["label"]
        return input_str, answer_str, label
    
    def download(self):
        random.seed(42)
        target_k = 500
        target_cross_task_save_dir = "data/cross_task_data/target"
        os.makedirs(target_cross_task_save_dir, exist_ok=True)
        arc_challenge_dataset_name = 'ARC-Challenge'
        arc_challenge_dataset = load_dataset('ai2_arc', arc_challenge_dataset_name)['test']

        arc_challenge_label2text = {
            'A':'A',
            'B':'B',
            "C":"C",
            "D":"D",
            '1':'A',
            '2':'B',
            "3":"C",
            "4":"D",
        }

        arc_challenge_data = []

        for d in tqdm(arc_challenge_dataset, desc="arc_challenge"):
            data = {}
            if len(d['choices']['text']) != 4:
                continue
            data['id'] = d['id']
            data['sentence'] = d['question']+'\nA. '+d['choices']['text'][0]+'\nB. '+d['choices']['text'][1]+'\nC. '+d['choices']['text'][2]+'\nD. '+d['choices']['text'][3]
            data['label'] = arc_challenge_label2text[d['answerKey']]
            arc_challenge_data.append(data)
            
        arc_challenge_data_sampled = random.sample(arc_challenge_data, target_k)
        write_jsonl(arc_challenge_data_sampled, os.path.join(target_cross_task_save_dir, 'arc_challenge.jsonl'))