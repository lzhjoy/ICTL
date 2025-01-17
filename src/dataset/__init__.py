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

from src.dataset.basetask import BaseTask
from src.dataset.agnews import AGNews
from src.dataset.arc_challenge import ARC_CHALLENGE
from src.dataset.arc_easy import ARC_EASY
from src.dataset.boolq import BOOLQ
from src.dataset.commense_qa import Commonsenseqa
from src.dataset.financial_phrasebank import FINANCIAL_PHRASEBANK
from src.dataset.medmcqa import MEDMCQA
from src.dataset.mnli import MNLI
from src.dataset.qqp import QQP
from src.dataset.race import RACE
from src.dataset.sciq import SCIQ
from src.dataset.social_qa import SOCIALQA
from src.dataset.sst2 import SST2
from src.dataset.multi_amazon import Multilingual_Amazon_Reviews_Corpus

datasets = {
    # cross task transfer learning
    "agnews": AGNews,
    "arc_challenge": ARC_CHALLENGE,
    "arc_easy": ARC_EASY,
    "boolq": BOOLQ,
    "commonsenseqa": Commonsenseqa,
    "financial_phrasebank": FINANCIAL_PHRASEBANK,
    "medmcqa": MEDMCQA,
    "mnli": MNLI,
    "qqp": QQP,
    "race": RACE,
    "sciq": SCIQ,
    "social_i_qa": SOCIALQA,
    "sst2": SST2,
    # cross lingual transfer learning
    "multi_amazon": Multilingual_Amazon_Reviews_Corpus,
}

def get_dataset(task_name, *args, **kwargs) -> BaseTask:
    return datasets[task_name](task_name=task_name, *args, **kwargs)