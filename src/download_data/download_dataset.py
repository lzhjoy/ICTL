import os
import sys

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.dataset import datasets, get_dataset

dataset_names = datasets.keys()

for dataset_name in dataset_names:
    ds_class = get_dataset(dataset_name)
    ds_class.download()