import json
import random
from tqdm import tqdm
import os
import sys
from datasets import load_dataset
import pandas as pd

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.utils.utils import write_jsonl

random.seed(42)
k = 1000
target_k = 500

cross_lingual_save_dir = "data/cross_lingual_data"
os.makedirs(cross_lingual_save_dir, exist_ok=True)


# =========== Multilingual Amazon Reviews Corpus ===========

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


for language in tqdm(languages):
    
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


# =========== CLS ===========

# cls_save_dir = f"{cross_lingual_save_dir}/cls"
# os.makedirs(cls_save_dir, exist_ok=True)

# langs = ['en', 'de', 'fr', 'jp']
# topics = ['books', 'dvd', 'music']
# df_train = pd.DataFrame()
# df_test = pd.DataFrame()

# for l in langs:
#     for t in topics:
#         df1 = pd.read_xml(f'data/raw/cls-acl10-unprocessed/{l}/{t}/train.review')
#         df2 = pd.read_xml(f'data/raw/cls-acl10-unprocessed/{l}/{t}/test.review')
        
#         df1['topic_number']=[t]*len(df1)
#         df2['topic_number']=[t]*len(df2)

#         df1['language']=[l]*len(df1)
#         df2['language']=[l]*len(df2)

        

#         df_train=pd.concat([df_train,df1])
#         df_test=pd.concat([df_test,df2])
        
# cls_v = {
#     'en':{
#         'negative':'bad',
#         'positive':'good'
#     },
#     'fr':{
#         'negative':'mal',
#         'positive':'bien'
#     },
#     'jp':{
#         'negative':'悪い',
#         'positive':'良い'
#     },
#     'de':{
#         'negative':'Schlecht',
#         'positive':'gut'
#     }
# }

# def verbiliser(l,i):
#     if i in [1.0, 2.0]:
#         return cls_v[l]['negative']
#     elif i in [4.0, 5.0]:
#         return cls_v[l]['positive']
#     else:
#         return

# df_train['output'] = df_train.apply(lambda x: verbiliser(x['language'], x['rating']),axis=1)
# df_test['output'] = df_test.apply(lambda x: verbiliser(x['language'], x['rating']),axis=1)

# cls_temp = {
#     'fr':{
#         'Review':'Examen: ',
#         'Rating':' Évaluation:'

#     },
#     'en':{
#         'Review':'Review: ',
#         'Rating':' Rating:'

#     },
#     'jp':{
#         'Review':'レビュー: ',
#         'Rating':' 評価:'
#     },
#     'de':{
#         'Review':'Rezension: ',
#         'Rating':'Bewertung:'
#     }
# }

# def temp(l,s):
#     return cls_temp[l]['Review']+str(s)+cls_temp[l]['Rating']
  
# df_train['input'] = df_train.apply(lambda x: temp(x['language'],x['text']), axis=1)
# df_train.to_csv(f'{cls_save_dir}/train.csv')

# df_test['input'] = df_test.apply(lambda x: temp(x['language'],x['text']), axis=1)
# df_test.to_csv(f'{cls_save_dir}/test.csv')