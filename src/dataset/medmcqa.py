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

class MEDMCQA(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Question: {sentence}\nAnswer:',
            'ans': '{label}',
            'options': ["A", "B", "C", "D"],
            'format': ['Question:', 'Answer:'],
            'instruction': 'Given a multiple choice question containing four options "A.", "B.", "C." and "D." from a medical entrance exam. The question is related to a sub-field of medical science like Microbiology, Radiology, Ophthalmology, Surgery, Human anatomy, etc. Based on the question, the option and your knowledge of the medical field select the most appropriate answer from the provided choices "A.", "B.", "C." and "D.".'
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
        medmcqa_dataset_name = 'medmcqa'
        medmcqa_dataset = load_dataset(medmcqa_dataset_name)['validation']

        medmcqa_text2label = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        medmcqa_data = []

        for d in tqdm(medmcqa_dataset, desc="medmcqa"):
            data = {}
            data['id'] = d['id']
            data['label'] = medmcqa_text2label[d['cop']]

            q = d['question']+' \nA. '+d['opa']+' \nB. '+d['opb']+' \nC. '+d['opc']+' \nD. '+d['opd']

            data['sentence'] = q
            medmcqa_data.append(data)

        medmcqa_data_sampled = random.sample(medmcqa_data, target_k)
        write_jsonl(medmcqa_data_sampled, os.path.join(target_cross_task_save_dir, 'medmcqa.jsonl'))