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

class ARC_EASY(BaseTask):
    
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
        k = 1000
        source_cross_task_save_dir = "data/cross_task_data/source"
        arceasy_dataset_name = 'ARC-Easy'
        arceasy_dataset = load_dataset('ai2_arc', arceasy_dataset_name)['train']
        id=1

        arceasy_label2text = {
            'A':'A',
            'B':'B',
            "C":"C",
            "D":"D",
            '1':'A',
            '2':'B',
            "3":"C",
            "4":"D",
        }

        arceasy_data = []

        for d in tqdm(arceasy_dataset, desc="arc_easy"):
            data = {}
            if len(d['choices']['text'])!=4:
                continue
            data['id'] = id
            id += 1
            data['sentence'] = d['question']+'\nA. '+d['choices']['text'][0]+'\nB. '+d['choices']['text'][1]+'\nC. '+d['choices']['text'][2]+'\nD. '+d['choices']['text'][3]
            data['label'] = arceasy_label2text[d['answerKey']]
            arceasy_data.append(data)
            
        arceasy_data_sampled = random.sample(arceasy_data, k)
        write_jsonl(arceasy_data_sampled, os.path.join(source_cross_task_save_dir, 'arc_easy.jsonl'))