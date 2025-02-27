import sys
import os
import json
import src.dataset as ds
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.method.basemethod import BaseMethod
from src.utils import utils
from src.utils import wrapper
class SteeringVector(BaseMethod):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tar_dataset_name = None
        self.src_dataset_name = None
        
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        self.device = self.accelerator.device
        self.tar_dataset_name = kwargs['tar_dataset_name']
        self.src_dataset_name = kwargs['src_dataset_name']

        self.construct_demon_index(num_vector=200, tar_dataset_name=self.tar_dataset_name, src_dataset_name=self.src_dataset_name)

        prompts_with_demonstration = self.construct_input_with_demonstration()
        prompts_without_demonstration = self.construct_input_without_demonstration()
        # print(prompts_with_demonstration[0]+'\n\n\n')
        # print(prompts_without_demonstration[0]+'\n\n\n')

        self.extract_steering_vector(prompts_with_demonstration, prompts_without_demonstration)

        # print("running evaluation")
        test_steering_result = self.test_evaluator.evaluate(self.tokenizer, self.model, use_demonstration=False,)
        self.result_dict['test_result']['ours'].append(test_steering_result)
        print(f'Test steering result: {test_steering_result}\n')
        
        with open(self.save_dir + '/result_dict.json', 'w') as f:
            json.dump(self.result_dict, f, indent=4)
    
    def construct_demon_index(self, num_vector, tar_dataset_name, src_dataset_name):
        # 创建保存路径
        save_dir = f"data/processed/demon_index/{self.config['shot_num']}_shot/{self.config['shot_method']}/{self.config['model_name']}/{tar_dataset_name}_{src_dataset_name}"
        save_path = os.path.join(save_dir, "demon_indices.json")
        
        # 检查文件是否已存在
        if os.path.exists(save_path):
            print(f"Demon indices file already exists at {save_path}, skipping generation...")
            return
        
        # 如果文件不存在，创建目录并继续处理
        os.makedirs(save_dir, exist_ok=True)
        
        # 用于存储所有样本的demon索引
        demon_indices_dict = {}
        
        for idx, data in tqdm(enumerate(self.test_data[:num_vector]), desc="generate demon indices"):
            ques_str, _, label = self.src_ds_class.apply_template(data)
            k = self.config['shot_num']
            ques_embed = self.sentence_model.encode([ques_str], convert_to_tensor=True)
            demon_embed = [demon['embed'] for demon in self.demon_info]
            
            if self.config['shot_method'] == 'random':
                # 随机选择k+1个示例的索引
                selected_indices = random.sample(range(len(self.demon_info)), k+1)
                
            elif self.config['shot_method'] == 'topk':
                demon_embed = torch.stack(demon_embed).squeeze(1)
                # 计算问题与所有示例的余弦相似度
                similarities = F.cosine_similarity(
                    ques_embed, 
                    demon_embed,
                    dim=1
                )
                # 获取相似度最高的k+1个示例的索引
                _, selected_indices = torch.topk(similarities, k+1)
                selected_indices = selected_indices.tolist()
                
            elif self.config['shot_method'] == 'dpp':
                # 计算质量分数（使用余弦相似度）
                # [num_demons,]
                demon_embed = torch.stack(demon_embed).squeeze(1)
                quality_scores = F.cosine_similarity(
                    ques_embed,  # [1, embedding_dim]
                    demon_embed,  # [num_demons, embedding_dim]
                    dim=1
                )
                # 确保质量分数为正值（因为余弦相似度范围是[-1,1]）
                quality_scores = (quality_scores + 1) / 2  # 将范围映射到[0,1]
                
                # 构建核矩阵
                # [num_demons, num_demons]
                similarity_matrix = torch.matmul(demon_embed, demon_embed.t())
                
                # 计算核矩阵 L
                # [num_demons, num_demons]
                L = similarity_matrix * quality_scores.unsqueeze(0) * quality_scores.unsqueeze(1)
                # print(L.shape)
                
                # DPP采样
                selected_indices = self.dpp_sample(L, k+1)
                
            
            # 将选中的索引存入字典
            demon_indices_dict[str(idx)] = {
                "question": ques_str,
                "selected_indices": selected_indices
            }
        
        # 保存为JSON文件
        with open(save_path, 'w') as f:
            json.dump(demon_indices_dict, f, indent=4)
        
        print(f"Demon indices saved to {save_path}")

    def construct_input_with_demonstration(self):
        # 读取保存的demon索引
        index_dir = f"data/processed/demon_index/{self.config['shot_num']}_shot/{self.config['shot_method']}/{self.config['model_name']}/{self.tar_dataset_name}_{self.src_dataset_name}/demon_indices.json"
        
        with open(index_dir, 'r') as f:
            demon_indices = json.load(f)

        # 获取任务的instruction
        if self.tar_ds_class != None:
            tar_instruction = self.tar_ds_class.get_dmonstration_template()['instruction']
        else:
            tar_instruction = ""
        
        prompts = []
        for idx in demon_indices:
            # 获取选中的示例索引
            selected_indices = demon_indices[idx]['selected_indices']
            
            # 构建demonstration字符串
            demonstrations = []
            # 使用除第一个索引外的所有索引构建示例
            for demon_idx in selected_indices[1:]:
                demon = self.demon_info[demon_idx]
                # print(demon['demon']+'\n\n\n')
                      
                demonstrations.append(f"{demon['demon']}{demon['label']}")
            
            # 将所有示例组合成一个字符串，用换行符分隔
            demo_str = "\n".join(demonstrations)
            
            # 构建最终的prompt，将示例放在问题之前
            question = self.demon_info[selected_indices[0]]['demon']
            final_prompt = f"{demo_str}\n{question}"

            if self.config['use_instruction']:
                final_prompt = f"Definition: {tar_instruction}\n{final_prompt}"
            else:
                final_prompt = f"{final_prompt}"
            
            prompts.append(final_prompt)
        
        return prompts

    def construct_input_without_demonstration(self):
        # 读取保存的demon索引
        index_dir = f"data/processed/demon_index/{self.config['shot_num']}_shot/{self.config['shot_method']}/{self.config['model_name']}/{self.tar_dataset_name}_{self.src_dataset_name}/demon_indices.json"
        
        with open(index_dir, 'r') as f:
            demon_indices = json.load(f)

        # 获取任务的instruction
        if self.tar_ds_class != None:
            tar_instruction = self.tar_ds_class.get_dmonstration_template()['instruction']
        else:
            tar_instruction = ""
        
        prompts = []
        for idx in demon_indices:
            # 获取选中的示例索引
            selected_indices = demon_indices[idx]['selected_indices']
            
            # 构建最终的prompt，将示例放在问题之前
            question = self.demon_info[selected_indices[0]]['demon']
            final_prompt = f"{question}"

            if self.config['use_instruction']:
                final_prompt = f"Definition: {tar_instruction}\n{final_prompt}"
            else:
                final_prompt = f"{final_prompt}"
            
            prompts.append(final_prompt)
        
        return prompts

    def extract_steering_vector(self, prompts_with_demonstration: list, prompts_without_demonstration: list):

        """提取steering vector并保存为json文件"""
        # 创建保存路径
        save_dir = f"data/processed/steering_vector/{self.config['shot_num']}_shot/{self.config['shot_method']}/{self.config['model_name']}/{self.src_dataset_name}_{self.tar_dataset_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        with_demo_path = os.path.join(save_dir, "hidden_states_with_demo.json")
        without_demo_path = os.path.join(save_dir, "hidden_states_without_demo.json")
        steering_vector_path = os.path.join(save_dir, "steering_vectors.json")
        
        # 检查文件是否已存在
        if os.path.exists(steering_vector_path):
            print(f"Steering vector file already exists at {steering_vector_path}, skipping generation...")
            return
        
        # 存储隐藏状态的字典
        hidden_states_with_demo = {}
        hidden_states_without_demo = {}
        steering_vectors = {}
        
        # 使用模型包装器提取隐藏状态
        model_wrapper = wrapper.LlamaWrapper(self.model, self.tokenizer, self.model_config, self.device)
        
        print("Extracting hidden states with demonstrations...")
        for idx, prompt in enumerate(tqdm(prompts_with_demonstration)):
            # 对输入进行编码
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 提取隐藏状态
            with model_wrapper.extract_latent():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 获取隐藏状态
                hidden_states = model_wrapper.get_context_vector(
                    [model_wrapper.latent_dict],
                    config={
                        'module': self.config['module'],
                        'tok_pos': self.config['tok_pos'],
                        'post_fuse_method': self.config['post_fuse_method']
                    }
                )
                hidden_states_with_demo[str(idx)] = {
                    str(layer): {
                        str(module): tensor.tolist() 
                        for module, tensor in layer_dict.items()
                    }
                    for layer, layer_dict in hidden_states.items()
                }
        
        print("Extracting hidden states without demonstrations...")
        for idx, prompt in enumerate(tqdm(prompts_without_demonstration)):
            # 对输入进行编码
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # 提取隐藏状态
            with model_wrapper.extract_latent():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 获取隐藏状态
                hidden_states = model_wrapper.get_context_vector(
                    [model_wrapper.latent_dict],
                    config={
                        'module': self.config['module'],
                        'tok_pos': self.config['tok_pos'],
                        'post_fuse_method': self.config['post_fuse_method']
                    }
                )
                hidden_states_without_demo[str(idx)] = {
                    str(layer): {
                        str(module): tensor.tolist() 
                        for module, tensor in layer_dict.items()
                    }
                    for layer, layer_dict in hidden_states.items()
                }
        
        print("Computing steering vectors...")
        # 计算steering vectors
        # hidden_states_with_demo: {str(idx): {str(layer): {str(module): [list]}}
        # hidden_states_without_demo: {str(idx): {str(layer): {str(module): [list]}}
        for idx in tqdm(hidden_states_with_demo.keys()):
            steering_vectors[idx] = {
                str(layer): {
                    self.config['module']: [  # 移除 str(module)，直接使用 module 字符串
                        a - b for a, b in zip(
                            hidden_states_with_demo[idx][str(layer)][self.config['module']],
                            hidden_states_without_demo[idx][str(layer)][self.config['module']]
                        )
                    ]
                }
                for layer in range(model_wrapper.num_layers)
            }
        
        # 保存结果
        print("Saving results...")
        with open(with_demo_path, 'w') as f:
            json.dump(hidden_states_with_demo, f, indent=4)
        
        with open(without_demo_path, 'w') as f:
            json.dump(hidden_states_without_demo, f, indent=4)
        
        with open(steering_vector_path, 'w') as f:
            json.dump(steering_vectors, f, indent=4)
        
        print(f"Results saved to {save_dir}")

    def inject_steering_vector(self):
        pass

    def dpp_sample(self, L, k):
        """
        使用贪心算法近似DPP进行采样，同时批量化处理
        L: 核矩阵
        k: 需要选择的示例数量
        """
        N = L.shape[0]
        device = L.device
        
        selected_indices = []
        remaining = torch.arange(N, device=device)
        
        for _ in range(k):
            if len(remaining) == 0:
                break
            
            # 批量计算边际增益
            if len(selected_indices) == 0:
                gains = torch.diagonal(L)
            else:
                prev_selected = torch.tensor(selected_indices, device=device)
                sub_L = L[prev_selected][:, prev_selected]
                V = L[remaining][:, prev_selected]
                gains = torch.diagonal(L)[remaining] - torch.sum(V * torch.linalg.solve(sub_L, V.T).T, dim=1)
            
            # 找到最大增益的索引
            max_gain_idx = torch.argmax(gains)
            selected_indices.append(remaining[max_gain_idx].item())
            remaining = torch.cat([remaining[:max_gain_idx], remaining[max_gain_idx+1:]])
        
        return selected_indices