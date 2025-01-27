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
# 继承自BaseTask
class AGNews(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Sentence: {sentence}\nLabel:',
            'ans': '{label}',
            'options': ["sports", "technology", "world", "business"],
            'format': ['Sentence:', 'Label:'],
            'instruction': 'Given a sentence do text classification, the sentence is a clipping from a news article that may be either related to sports, business, technology, or world news. You are to recognize the category of the sentence and label them as "sports", "business", "technology" or "world" news'
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
        os.makedirs(source_cross_task_save_dir, exist_ok=True)
        agnews_dataset_name = 'ag_news'
        agnews_dataset= load_dataset(agnews_dataset_name)['train']
        id = 1 # 数据id
        agnews_label2text={
            0: 'world',
            1: 'sports',
            2: 'business',
            3: 'technology'
        }

        agnews_data = [] # 数据列表，每一条数据是一个字典

        for d in tqdm(agnews_dataset, desc="agnews"):
            data = {}
            data['id'] = id
            id += 1
            data['sentence'] = d['text']
            data['label'] = agnews_label2text[d['label']]
            agnews_data.append(data)

        agnews_data_sampled = random.sample(agnews_data, k) # 随机采样k条数据
        write_jsonl(agnews_data_sampled, os.path.join(source_cross_task_save_dir, 'agnews.jsonl'))