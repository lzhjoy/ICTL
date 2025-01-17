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

class QQP(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Question1: {s1}\nQuestion2: {s2}\nLabel:',
            'ans': '{label}',
            'options': ["entailment", "neutral", "contradiction"],
            'format': ['Question1:', 'Question2:', 'Label:'],
            'instruction': 'Given two question pairs do text classification based on whether they are duplicates or not. The questions are mined from the popular online discussion forum Quora. As duplicate quetion might be present on Quora, the task is to label two identical questions as "duplicate" if they ask the same query else label the pair as "not duplicate".'
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
        input_str = input_template.replace("{s1}", data["s1"]).replace("{s2}", data["s2"])
        # answers can have multiple options and is a list
        answer_str = [ans_template.replace("{label}", options[i]) for i in range(len(options))]
        label = data["label"]
        return input_str, answer_str, label
    
    def download(self):
        random.seed(42)
        k = 1000
        source_cross_task_save_dir = "data/cross_task_data/source"
        qqp_dataset_name = 'qqp'
        qqp_dataset = load_dataset('glue', qqp_dataset_name)['train']

        qqp_label2text = {
            1: 'duplicate',
            0: 'not duplicate'
        }

        qqp_data = []

        for d in tqdm(qqp_dataset, desc='qqp'):
            data = {}
            data['id'] = d['idx']
            data['s1'] = d['question1']
            data['s2'] = d['question2']
            data['label'] = qqp_label2text[d['label']]
            qqp_data.append(data)
            
        qqp_data_sampled = random.sample(qqp_data, k)
        write_jsonl(qqp_data_sampled, os.path.join(source_cross_task_save_dir, 'qqp.jsonl'))