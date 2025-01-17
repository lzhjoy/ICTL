import json
import random
from tqdm import tqdm
import os
import sys
from datasets import load_dataset

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.utils.utils import write_jsonl

random.seed(42)
k = 1000
target_k = 500

source_cross_task_save_dir = "data/cross_task_data/source"
target_cross_task_save_dir = "data/cross_task_data/target"
os.makedirs(source_cross_task_save_dir, exist_ok=True)
os.makedirs(target_cross_task_save_dir, exist_ok=True)


# Source Data

# =========== mnli ===========

mnli_dataset_name = 'mnli'
mnli_dataset = load_dataset('glue', mnli_dataset_name)['train']
mnli_label2text = {
    2: 'contradiction',
    1: 'neutral',
    0: 'entailment'
}

mnli_data = []

for d in mnli_dataset:
    data = {}
    data['id'] = d['idx']
    data['s1'] = d['premise']
    data['s2'] = d['hypothesis']
    data['label'] = mnli_label2text[d['label']]
    mnli_data.append(data)
    
mnli_data_sampled = random.sample(mnli_data, k)
write_jsonl(mnli_data_sampled, os.path.join(source_cross_task_save_dir, 'mnli.jsonl'))

# =========== qqp ===========

qqp_dataset_name = 'qqp'
qqp_dataset = load_dataset('glue', qqp_dataset_name)['train']

qqp_label2text = {
    1: 'duplicate',
    0: 'not duplicate'
}

qqp_data = []

for d in tqdm(qqp_dataset, desc='qqp'):
    data = {}
    data['id'] = d['idx']
    data['s1'] = d['question1']
    data['s2'] = d['question2']
    data['label'] = qqp_label2text[d['label']]
    qqp_data.append(data)
    
qqp_data_sampled = random.sample(qqp_data, k)
write_jsonl(qqp_data_sampled, os.path.join(source_cross_task_save_dir, 'qqp.jsonl'))


# # =========== boolq ===========

boolq_dataset_name = 'boolq'
boolq_dataset = load_dataset(boolq_dataset_name)['train']
id = 1

boolq_data = []

for d in tqdm(boolq_dataset, desc='boolq'):
    data = {}
    id += 1
    data['id'] = id
    data['sentence'] = d['question']
    data['context'] = d['passage']
    data['label'] = str(d['answer'])
    boolq_data.append(data)

boolq_data_sampled = random.sample(boolq_data, k)
write_jsonl(boolq_data_sampled, os.path.join(source_cross_task_save_dir, 'boolq.jsonl'))

# # =========== conll ===========

conll_dataset_name = 'conll2003'
conll_dataset = load_dataset(conll_dataset_name)['train']

conll_ner_label2text = {
    0: 'O', 
    1: 'B-PER', 
    2: 'I-PER',
    3: 'B-ORG',
    4: 'I-ORG',
    5: 'B-LOC',
    6: 'I-LOC',
    7: 'B-MISC',
    8: 'I-MISC'
}

conll_ner_data = []

for d in conll_dataset:
    data = {}
    data['id'] = d['id']
    l = []
    for i in d['ner_tags']:
        l.append(conll_ner_label2text[i])
    
    data['sentence'] = ' '.join(d['tokens'])
    data['label'] = ' '.join(l)
    conll_ner_data.append(data)
    
conll_ner_data_sampled = random.sample(conll_ner_data, k)
write_jsonl(conll_ner_data_sampled, os.path.join(source_cross_task_save_dir, 'conll_ner.jsonl'))

# =========== conll ===========

conll_dataset_name = 'conll2003'
conll_dataset = load_dataset(conll_dataset_name)['train']
conll_pos_label = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12, 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23, 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33, 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43, 'WP': 44, 'WP$': 45, 'WRB': 46}


conll_pos_label2text = dict([(v, k) for k, v in conll_pos_label.items()])
conll_pos_data = []

for d in conll_dataset:
    data = {}
    data['id'] = d['id']
    for i in d['pos_tags']:
        l.append(conll_pos_label2text[i])
    data['sentence'] = ' '.join(d['tokens'])
    data['label'] = ' '.join(l)
    conll_pos_data.append(data)
    
conll_pos_data_sampled = random.sample(conll_pos_data, k)
write_jsonl(conll_pos_data_sampled, os.path.join(source_cross_task_save_dir, 'conll_pos.jsonl'))


# =========== commensense_qa ===========

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

for d in cqa_dataset:
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


# =========== ARC_Easy ===========

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

for d in arceasy_dataset:
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


# =========== race ===========

race_dataset_name = 'race'
race_dataset = load_dataset('race', 'all')['train']
race_data = []

for d in race_dataset:
    data = {}
    q = d['question']+' \nA. '+d['options'][0]+' \nB. '+d['options'][1]+' \nC. '+d['options'][2]+' \nD. '+d['options'][3]

    data['id'] = d['example_id']
    data['sentence'] = q
    data['context'] = d['article']
    data['label'] = d['answer']
    race_data.append(data)

race_data_sampled = random.sample(race_data, k)
write_jsonl(race_data_sampled, os.path.join(source_cross_task_save_dir, 'race.jsonl'))

# =========== agnews ===========

agnews_dataset_name = 'ag_news'
agnews_dataset= load_dataset(agnews_dataset_name)['train']
id = 1
agnews_label2text={
    0: 'world',
    1: 'sports',
    2: 'business',
    3: 'technology'
}

agnews_data = []

for d in agnews_dataset:
    data = {}
    data['id'] = id
    id += 1
    data['sentence'] = d['text']
    data['label'] = agnews_label2text[d['label']]
    agnews_data.append(data)

agnews_data_sampled = random.sample(agnews_data, k)
write_jsonl(agnews_data_sampled, os.path.join(source_cross_task_save_dir, 'agnews.jsonl'))


# =========== sst2 ===========

sst2_dataset_name = 'sst2'
sst2_dataset= load_dataset('glue','sst2')['train']
sst2_label2text={
    1: 'positive',
    0: 'negative'
}

sst2_data = []

for d in sst2_dataset:
    data = {}
    data['id'] = d['idx']
    data['sentence'] = d['sentence']
    data['label'] = sst2_label2text[d['label']]
    sst2_data.append(data)
    
sst2_data_sampled = random.sample(sst2_data, k)
write_jsonl(sst2_data_sampled, os.path.join(source_cross_task_save_dir, 'sst2.jsonl'))



# Target Data

# =========== medmcqa ===========

medmcqa_dataset_name = 'medmcqa'
medmcqa_dataset = load_dataset(medmcqa_dataset_name)['validation']

medmcqa_text2label = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

medmcqa_data = []

for d in medmcqa_dataset:
    data = {}
    data['id'] = d['id']
    data['label'] = medmcqa_text2label[d['cop']]

    q = d['question']+' \nA. '+d['opa']+' \nB. '+d['opb']+' \nC. '+d['opc']+' \nD. '+d['opd']

    data['sentence'] = q
    medmcqa_data.append(data)

medmcqa_data_sampled = random.sample(medmcqa_data, target_k)
write_jsonl(medmcqa_data_sampled, os.path.join(target_cross_task_save_dir, 'medmcqa.jsonl'))


# =========== sciq ===========

sciq_dataset_name = 'sciq'
sciq_dataset = load_dataset(sciq_dataset_name)['test']
id = 1
text2label = {
    'A':'distractor3',
    'B':'distractor1',
    'C':'distractor2',
    'D':'correct_answer'
}

sciq_data = []

for d in sciq_dataset:
    data = {}
    k=['A','B','C','D']
    k_=['A','B','C','D']
    random.shuffle(k)
    q = d['question']   
    for i, l in zip(k, k_):
        op=text2label[i]
        if op == 'correct_answer':
            data['label'] = l

        q += f'\n{l}. '+ d[op]
                    
    data['id'] = id
    id += 1
    data['sentence'] = q
    sciq_data.append(data)

sciq_data_sampled = random.sample(sciq_data, target_k)
write_jsonl(sciq_data_sampled, os.path.join(target_cross_task_save_dir, 'sciq.jsonl'))


# =========== ARC-Challenge ===========

arc_challenge_dataset_name = 'ARC-Challenge'
arc_challenge_dataset = load_dataset('ai2_arc', arc_challenge_dataset_name)['test']

arc_challenge_label2text = {
    'A':'A',
    'B':'B',
    "C":"C",
    "D":"D",
    '1':'A',
    '2':'B',
    "3":"C",
    "4":"D",
}

arc_challenge_data = []

for d in arc_challenge_dataset:
    data = {}
    if len(d['choices']['text']) != 4:
        continue
    data['id'] = d['id']
    data['sentence'] = d['question']+'\nA. '+d['choices']['text'][0]+'\nB. '+d['choices']['text'][1]+'\nC. '+d['choices']['text'][2]+'\nD. '+d['choices']['text'][3]
    data['label'] = arc_challenge_label2text[d['answerKey']]
    arc_challenge_data.append(data)
    
arc_challenge_data_sampled = random.sample(arc_challenge_data, target_k)
write_jsonl(arc_challenge_data_sampled, os.path.join(target_cross_task_save_dir, 'arc_challenge.jsonl'))


# =========== social_i_qa ===========

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


# =========== financial_phrasebank ===========

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