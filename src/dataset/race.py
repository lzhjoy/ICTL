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

class RACE(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Context: {context}\nQuestion: {sentence}\nLabel:',
            'ans': '{label}',
            'options': ["Context", "False"],
            'format': ['Sentence:', 'Question:', 'Label:'],
            'instruction': 'Given a context and a question do binary true and false type text classification. You are given a passage as context and a question related to the passage that can be answered as "True" or "False". Based on the context, question and your reasoning ability answer in a "True" and "False".'
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
        input_str = input_template.replace("{sentence}", data["sentence"]).replace("{context}", data["context"])
        # answers can have multiple options and is a list
        answer_str = [ans_template.replace("{label}", options[i]) for i in range(len(options))]
        label = data["label"]
        return input_str, answer_str, label
    
    def download(self):
        random.seed(42)
        k = 1000
        source_cross_task_save_dir = "data/cross_task_data/source"
        race_dataset_name = 'race'
        race_dataset = load_dataset('race', 'all')['train']
        race_data = []

        for d in tqdm(race_dataset, desc="race"):
            data = {}
            q = d['question']+' \nA. '+d['options'][0]+' \nB. '+d['options'][1]+' \nC. '+d['options'][2]+' \nD. '+d['options'][3]

            data['id'] = d['example_id']
            data['sentence'] = q
            data['context'] = d['article']
            data['label'] = d['answer']
            race_data.append(data)

        race_data_sampled = random.sample(race_data, k)
        write_jsonl(race_data_sampled, os.path.join(source_cross_task_save_dir, 'race.jsonl'))