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

class Multilingual_Amazon_Reviews_Corpus(BaseTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"
    
    def download(self):
        
        random.seed(42)
        k = 1000
        target_k = 500

        cross_lingual_save_dir = "data/cross_lingual_data"
        os.makedirs(cross_lingual_save_dir, exist_ok=True)
        
        amaz_save_dir = f"{cross_lingual_save_dir}/amazon_reviews_corpus"
        os.makedirs(amaz_save_dir, exist_ok=True)

        languages = ['de', 'en', 'es', 'fr', 'ja', 'zh']

        amaz_t={
            'de':'Rezension: {s} Bewertung:',
            'ja':'レビュー: {s} 評価:',
            'es':'Revisar: {s} Clasificación:',
            'fr':'Examen: {s} Évaluation:',
            'zh':'审查: {s} 评分:',
            'en':'Review: {s} Rating:'
            
        }

        amaz_bi_v={
            'en':{
                'negative':'bad',
                'positive':'good'
            },
            'fr':{
                'negative':'mal',
                'positive':'bien'
            },
            'es':{
                'negative':'malo',
                'positive':'bueno'
            },
            'ja':{
                'negative':'悪い',
                'positive':'良い'
            },
            'zh':{
                'negative':'坏的',
                'positive':'好的'
            },
            'de':{
                'negative':'Schlecht',
                'positive':'gut'
            }
        }


        for language in tqdm(languages, desc="language"):
            
            train_marc = load_dataset('mteb/amazon_reviews_multi', language)['train']

            amaz_train_data = []

            for d in train_marc:
                data = {}
                data['language'] = language

                if d['label'] == 2:
                    continue
                elif d['label'] in [0, 1]:
                    data['output'] = amaz_bi_v[language]['negative']
                elif d['label'] in [3, 4]:
                    data['output'] = amaz_bi_v[language]['positive']
                
                data['input'] = amaz_t[language].format(s = d['text'])
                data['id'] = d['id']
                amaz_train_data.append(data)

            amaz_train_data = random.sample(amaz_train_data, k)
            
            language_amaz_path = os.path.join(amaz_save_dir, language)
            os.makedirs(language_amaz_path, exist_ok=True)
            
            write_jsonl(amaz_train_data, os.path.join(language_amaz_path, f'train.jsonl'))


            test_marc = load_dataset('mteb/amazon_reviews_multi', language)['test']

            amaz_test_data = []

            for d in test_marc:
                data = {}
                data['language'] = language

                if d['label'] == 2:
                    continue
                elif d['label'] in [0, 1]:
                    data['output'] = amaz_bi_v[language]['negative']
                elif d['label'] in [3, 4]:
                    data['output'] = amaz_bi_v[language]['positive']
                
                data['input'] = amaz_t[language].format(s=d['text'])
                data['id'] = d['id']
                amaz_test_data.append(data)
                
            amaz_test_data = random.sample(amaz_test_data, target_k)

            write_jsonl(amaz_test_data, os.path.join(language_amaz_path, f'test.jsonl'))