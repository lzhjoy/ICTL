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

class MNLI(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Sentence1: {s1}\nSentence2: {s2}\nLabel:',
            'ans': '{label}',
            'options': ["entailment", "neutral", "contradiction"],
            'format': ['Sentence1:', 'Sentence2:', 'Label:'],
            'instruction': 'Given Sentence 1 which is a premise and Sentence 2 which is a hypothesis do natural language inference on the pair. In natural language inference we mark whether the premise and hypothesis are "neutral", "contradiction" or "entailment". The pair are said to be "entailed" if the premise justifies/supports the hypothesis, if the pair contradict each other we label them as "contradiction" and label them "neutral" in all other cases".'
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
        mnli_dataset_name = 'mnli'
        mnli_dataset = load_dataset('glue', mnli_dataset_name)['train']
        mnli_label2text = {
            2: 'contradiction',
            1: 'neutral',
            0: 'entailment'
        }

        mnli_data = []

        for d in tqdm(mnli_dataset, desc="mnli"):
            data = {}
            data['id'] = d['idx']
            data['s1'] = d['premise']
            data['s2'] = d['hypothesis']
            data['label'] = mnli_label2text[d['label']]
            mnli_data.append(data)
            
        mnli_data_sampled = random.sample(mnli_data, k)
        write_jsonl(mnli_data_sampled, os.path.join(source_cross_task_save_dir, 'mnli.jsonl'))
