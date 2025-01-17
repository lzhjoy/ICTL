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

class SOCIALQA(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Context: {context}\nQuestion: {sentence}\nLabel:',
            'ans': '{label}',
            'options': ["A", "B", "C"],
            'format': ['Context:', 'Question:', 'Label:'],
            'instruction': 'Given an action as the context and a related question, you are to answer the question based on the context using your social intelligence. The question is of multiple choice form with three options "A", "B" and "C". Select the most appropriate answer from the provided choices "A", "B" and "C".'
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
        target_k = 500
        target_cross_task_save_dir = "data/cross_task_data/target"
        os.makedirs(target_cross_task_save_dir, exist_ok=True)
        social_qa_dataset_name = 'social_i_qa'
        social_qa_dataset = load_dataset(social_qa_dataset_name, trust_remote_code=True)['validation']
        id=1

        social_qa_text2label={
            '1':'A',
            '2':'B',
            '3':'C'
        }
        social_qa_label2text={
            1:'answerA',
            2:'answerB',
            3:'answerC'
        }

        social_qa_data = []

        for d in social_qa_dataset:
            data = {}
            data['id'] = id
            id += 1

            data['label'] = social_qa_text2label[d['label']]
            data['context'] = d['context']

            q = d['question']+' \nA. '+d['answerA']+' \nB. '+d['answerB']+' \nC. '+d['answerC']
            data['sentence'] = q
            social_qa_data.append(data)

        social_qa_data_sampled = random.sample(social_qa_data, target_k)
        write_jsonl(social_qa_data_sampled, os.path.join(target_cross_task_save_dir, 'social_i_qa.jsonl'))