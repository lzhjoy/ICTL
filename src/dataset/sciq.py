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

class SCIQ(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Question: {sentence}\nAnswer:',
            'ans': '{label}',
            'options': ["A", "B", "C", "D"],
            'format': ['Question:', 'Answer:'],
            'instruction': 'Given a question from a scientific exam about Physics, Chemistry, and Biology, among others. The question is in multiple choice format with four answer options "A.", "B.", "C." and "D.". Using your knowledge about the scientific fields answer the question and provide the label "A", "B", "C" and "D" as answer'
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
        sciq_dataset_name = 'sciq'
        sciq_dataset = load_dataset(sciq_dataset_name)['test']
        id = 1
        text2label = {
            'A':'distractor3',
            'B':'distractor1',
            'C':'distractor2',
            'D':'correct_answer'
        }

        sciq_data = []

        for d in tqdm(sciq_dataset, desc="sciq"):
            data = {}
            k=['A','B','C','D']
            k_=['A','B','C','D']
            random.shuffle(k)
            q = d['question']   
            for i, l in zip(k, k_):
                op=text2label[i]
                if op == 'correct_answer':
                    data['label'] = l

                q += f'\n{l}. '+ d[op]
                            
            data['id'] = id
            id += 1
            data['sentence'] = q
            sciq_data.append(data)

        sciq_data_sampled = random.sample(sciq_data, target_k)
        write_jsonl(sciq_data_sampled, os.path.join(target_cross_task_save_dir, 'sciq.jsonl'))