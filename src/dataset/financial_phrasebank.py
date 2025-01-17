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

class FINANCIAL_PHRASEBANK(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def get_dmonstration_template(self):
        template = {
            'input': 'Sentence: {sentence}\nLabel:',
            'ans': '{label}',
            'options': ["negative", "positive", "neutral"],
            'format': ['Sentence:', 'Label:'],
            'instruction': 'Given a sentence mined from a financial news article, you are to determine the sentiment polarity of the sentence. The task deals with financial sentiment analysis. Based on the sentiment conveyed by the sentence, label the sentence as "negative", "positive" or "neutral"'
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
        fp_dataset_name = 'financial_phrasebank'
        fp_dataset = load_dataset(fp_dataset_name, 'sentences_allagree', trust_remote_code=True)['train']
        fp_dataset = fp_dataset.train_test_split(test_size=0.2, stratify_by_column="label")
        dataset = fp_dataset['train']
        id=1

        fp_text2label={
            1:'neutral',
            2:'positive',
            0:'negative'
        }

        fp_data = []

        for d in dataset:
            data = {}
            data['id'] = id
            id+=1
            data['sentence'] = d['sentence']
            data['label'] = fp_text2label[d['label']]
            fp_data.append(data)

        fp_data_sampled = random.sample(fp_data, target_k)
        write_jsonl(fp_data_sampled, os.path.join(target_cross_task_save_dir, 'financial_phrasebank.jsonl'))