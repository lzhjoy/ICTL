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

class SST2(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Sentence: {sentence}\nLabel:',
            'ans': '{label}',
            'options': ["positive", "negative"],
            'format': ['Sentence:', 'Label:'],
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
        k = 1000
        source_cross_task_save_dir = "data/cross_task_data/source"
        sst2_dataset_name = 'sst2'
        sst2_dataset = load_dataset('glue','sst2')['train']
        sst2_label2text = {
            1: 'positive',
            0: 'negative'
        }

        sst2_data = []

        for d in tqdm(sst2_dataset, desc="sst2"):
            data = {}
            data['id'] = d['idx']
            data['sentence'] = d['sentence']
            data['label'] = sst2_label2text[d['label']]
            sst2_data.append(data)
            
        sst2_data_sampled = random.sample(sst2_data, k)
        write_jsonl(sst2_data_sampled, os.path.join(source_cross_task_save_dir, 'sst2.jsonl'))