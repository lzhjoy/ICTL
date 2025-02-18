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

class Commensenseqa(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Question: {sentence}\nAnswer:',
            'ans': '{label}',
            'options': ["A", "B", "C", "D", "E"],
            'format': ['Question:', 'Answer:'],
            'instruction': 'The following task relates to commonsense reasoning. It consists of a question that can be easily solved using logical abilities and reasoning, a set of five options  "A.", "B.", "C.", "D." and "E." are also provided along with the question, one of these options answers the question logically. Use your reasoning ability to select the most appropriate answer from the provided choices "A.", "B.", "C.", "D." and "E." and assign these choices (i.e  "A.", "B.", "C.", "D." and "E.") as the label'
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
        cqa_dataset_name= 'commonsense_qa'
        cqa_dataset= load_dataset('commonsense_qa')['validation']

        cqa_text2label={
            'A':0,
            'B':1,
            'C':2,
            'D':3,
            'E':4
        }

        cqa_label2text={
            0:'\nA. ',
            1:'\nB. ',
            2:'\nC. ',
            3:'\nD. ',
            4:'\nE. '
        }

        cqa_data = []

        for d in tqdm(cqa_dataset, desc="commonsenseqa"):
            data = {}
            data['id'] = d['id']
            data['label'] = d['answerKey']

            q = d['question']
            for i, a in enumerate(d['choices']['text']):
                q += ' ' + cqa_label2text[i] + a

            data['sentence'] = q
            cqa_data.append(data)
            
        cqa_data_sampled = random.sample(cqa_data, k)
        write_jsonl(cqa_data_sampled, os.path.join(source_cross_task_save_dir, 'commensense_qa.jsonl'))