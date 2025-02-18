import sys
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

import src.utils.utils as utils
import src.dataset as ds
import src.utils.evaluator as ev

class BaseMethod:
    
    def __init__(self, method_name, config, accelerator) -> None:
        self.method_name = method_name
        self.config = config
        self.accelerator = accelerator
        self.device = self.accelerator.device
        
    def init_exp_path(self, dataset_name):
        self.save_dir = utils.init_exp_path(self.config, dataset_name)
        
    def load_model_tokenizer(self):
        self.model, self.tokenizer, self.model_config, self.MODEL_CONFIG = utils.load_model_tokenizer(self.config, self.accelerator)
    
    # 示例来自于源任务
    def load_demonstration_list(self, dataset_name):
        demon_path = f"data/{self.config['domain']}/source"
        self.demon_data = utils.read_jsonl(f"{demon_path}/{dataset_name}.jsonl")
    
    def get_embedding(self):
        self.sentence_model = SentenceTransformer("/mnt/tangxinyu/huggingface/models/BAAI--bge-base-en-v1.5/").to(self.device)
        self.demon_info = []
        for demon in tqdm(self.demon_data, desc="embed demon"):
            demon_str, _, label = self.tar_ds_class.apply_template(demon)
            demon_embed = self.sentence_model.encode([demon_str], convert_to_tensor=True)
            self.demon_embeddings = demon_embed
            self.demon_info.append({'demon': demon_str, 'label': label, 'embed': demon_embed})
            
    
    def load_test_dataset(self, dataset_name):
        if self.config['domain'] == 'cross_task_data':
            test_path = f"data/{self.config['domain']}/target"
            self.test_data = utils.read_jsonl(f"{test_path}/{dataset_name}.jsonl")
            self.test_data = self.test_data[:self.config['test_num']]
        elif self.config['domain'] == 'cross_lingual_data':
            # TODO: 需要完善
            pass
    
    def get_evaluator(self):
        self.test_evaluator = ev.Evaluator(config=self.config, sentence_model=self.sentence_model, src_ds_class=self.src_ds_class, tar_ds_class=self.tar_ds_class, demon_info=self.demon_info, dataset=self.test_data, batch_size=self.config['bs'], accelerator=self.accelerator)
        self.result_dict = {'demon': {}, 'dev_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 'test_result': {'zero_shot': [], 'few_shot': [], 'ours': []}, 'time': {'train': [], 'evaluate': []}, 'best_replace_layer': {}}
    
    # 在这个类里，src_dataset_name 是目标任务，tar_dataset_name 是源任务
    def run(self, src_dataset_name, tar_dataset_name=None):
        self.src_dataset_name = src_dataset_name

        # src_ds_class 和 tar_ds_class 是数据集的类
        self.src_ds_class = ds.datasets[src_dataset_name](task_name=src_dataset_name)
        if tar_dataset_name is not None:
            self.tar_dataset_name = tar_dataset_name
            self.tar_ds_class = ds.datasets[tar_dataset_name](task_name=tar_dataset_name)
        else:
            self.tar_dataset_name = None
            self.tar_ds_class = None

            
        # 初始化实验路径
        if self.tar_dataset_name is not None:
            self.init_exp_path(f"{src_dataset_name}_{tar_dataset_name}")
        else:
            self.init_exp_path(src_dataset_name)

        
        # 加载模型和tokenizer
        self.load_model_tokenizer()
        # 加载测试数据集
        self.load_test_dataset(src_dataset_name)
        # 如果目源据集存在，则加载源数据集的演示数据
        if self.tar_dataset_name is not None:
            self.load_demonstration_list(tar_dataset_name)
            self.get_embedding()
        else:
            self.sentence_model = None
            self.demon_info = None
            self.demon_embeddings = None
        self.get_evaluator()