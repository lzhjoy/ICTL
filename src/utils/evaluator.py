import sys
import os
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
import random

ICTL_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, ICTL_ROOT_PATH)

from src.utils import utils
import src.utils.global_vars as gv
from tqdm import tqdm

class Evaluator(nn.Module):

    def __init__(self, config, sentence_model, src_ds_class, tar_ds_class, demon_info, dataset, batch_size, accelerator):
        super().__init__()
        self.config = config
        self.sentence_model = sentence_model
        self.src_ds_class = src_ds_class
        self.tar_ds_class = tar_ds_class
        self.demon_info = demon_info
        self.dataset = dataset
        self.batch_size = batch_size
        self.accelerator = accelerator

    def evaluate(self, tokenizer, model, use_demonstration=False, use_logits=True):
        
        # TODO: 不用cache，few-shot设置下，来一个query直接检索示例
        
        # 前向传播得到label
        if use_logits:
            
            # 这种映射在处理分类任务时很重要，因为：
            # 模型输出的是token ID  
            # 评估需要数字标签
            # 最终展示需要可读的文本答案
            # prepare label dict
            label_map = {}
            label2id = {}
            ans_txt_list = self.src_ds_class.get_dmonstration_template()['options']
            for label, ans_txt in enumerate(ans_txt_list):
                ans_tok = tokenizer.encode(ans_txt, add_special_tokens=False)[0]  # use the first token if more than one token
                print(f"ans_txt: {ans_txt}, ans_tok: {ans_tok}")
                label_map[ans_tok] = ans_txt  # index is the label
                label2id[ans_txt] = label
            print(f"label_map: {label_map}")

            # prepare all test data
            
            if self.config['use_instruction']:
                src_instruction = self.src_ds_class.get_dmonstration_template()['instruction']
                if self.tar_ds_class != None:
                    tar_instruction = self.tar_ds_class.get_dmonstration_template()['instruction']
                else:
                    tar_instruction = ""
            
            all_pred_labels = []
            all_inputs, all_labels = [], []
            for data in tqdm(self.dataset, desc="generate inputs and labels"):
                ques_str, _, label = self.src_ds_class.apply_template(data)
                if not use_demonstration:
                    if self.config['use_instruction']:
                        context = f"Definition: {src_instruction}\n{ques_str}"
                    else:
                        context = ques_str
                else:
                    k = self.config['shot_num']
                    ques_embed = self.sentence_model.encode([ques_str], convert_to_tensor=True)
                    demon_embed = [demon['embed'] for demon in self.demon_info]
                    if self.config['shot_method'] == 'random':
                        # 随机选择k个示例
                        selected_demons = random.sample(self.demon_info, k)
                        # 构建示例字符串
                        demonstrations = []
                        for demon in selected_demons:
                            demon_str = f"{demon['demon']}{demon['label']}\n"
                            demonstrations.append(demon_str)
                        # 将所有示例连接成一个字符串
                        demonstration = "".join(demonstrations) 
                    elif self.config['shot_method'] == 'topk':
                        demon_embed = torch.stack(demon_embed).squeeze(1)
                        # 计算问题与所有示例的余弦相似度
                        similarities = F.cosine_similarity(
                            ques_embed, 
                            demon_embed,
                            dim=1
                        )
                        # print(torch.tensor(demon_embed).shape)
                        # print(torch.stack(demon_embed).squeeze(1).shape)
                        # print(ques_embed.shape)
                        # print(similarities.shape)

                        # 获取相似度最高的k个示例的索引
                        _, top_k_indices = torch.topk(similarities, k)
                        # print(f"top_k_indices: {top_k_indices}")
                        # print(type(top_k_indices))
                        # print(top_k_indices.shape)
                        # 构建示例字符串
                        demonstrations = []
                        for idx in top_k_indices:
                            demon = self.demon_info[idx]
                            demon_str = f"{demon['demon']}{demon['label']}\n"
                            demonstrations.append(demon_str)
                        # 将所有示例连接成一个字符串
                        demonstration = "".join(demonstrations)
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
                        selected_indices = self.dpp_sample(L, k)
                        
                        # 构建示例字符串
                        demonstrations = []
                        for idx in selected_indices:
                            demon = self.demon_info[idx]
                            demon_str = f"{demon['demon']}{demon['label']}\n"
                            demonstrations.append(demon_str)
                        demonstration = "".join(demonstrations)
                    
                    if self.config['use_instruction']:
                        context = f"Definition: {tar_instruction}\n{demonstration}\nDefinition: {src_instruction}\n{ques_str}"
                    else:
                        context = demonstration + "\n" + ques_str
                # print(f"context: {context}")

                all_inputs.append(context)
                all_labels.append(label2id[label])
            
            # loop over all data
            with torch.no_grad():
                for i in tqdm(range(0, len(all_inputs), self.batch_size), desc="evaluate"):
                    # 批量处理输入数据
                    cur_inputs = all_inputs[i:i+self.batch_size]
                    
                    # 对输入进行tokenization
                    input_tok = tokenizer(cur_inputs, return_tensors="pt", padding=True)
                    input_ids = input_tok['input_ids'].to(self.accelerator.device)
                    # 获取注意力掩码，用于处理输入序列中的填充标记
                    attn_mask = input_tok['attention_mask'].to(self.accelerator.device)
                    # 获取注意力掩码中最后一个1的位置，用于处理输入序列中的填充标记
                    pred_loc = utils.last_one_indices(attn_mask).to(self.accelerator.device)
                    # 设置全局变量
                    gv.ATTN_MASK_START = torch.zeros_like(pred_loc)
                    gv.ATTN_MASK_END = pred_loc
                    # 前向传播
                    output = model(input_ids=input_ids, attention_mask=attn_mask)
                    
                    # (batch_size, seq_len, vocab_size)
                    logits = output.logits

                    # get prediction logits，the last token is the prediction
                    pred_logits = logits[torch.arange(logits.size(0)), pred_loc]
                    # get prediction labels
                    interest_index = list(label_map.keys())
                    pred_logits = pred_logits[:, interest_index]
                    probs = F.softmax(pred_logits, dim=-1)
                    pred_labels = probs.argmax(dim=-1)
                    # save results
                    all_pred_labels.extend(pred_labels.cpu().numpy().tolist())
                    
        else:
            pass
                    
        # 进行评估            
        assert len(all_pred_labels) == len(all_labels)
        acc = []
            
        num_classes = len(self.src_ds_class.get_dmonstration_template()['options'])
        TP = [0] * num_classes
        FP = [0] * num_classes
        FN = [0] * num_classes
        
        for i, true_label in enumerate(all_labels):
            pred_label = all_pred_labels[i]
            pred = (pred_label == true_label)
            acc.append(pred)
            # Update TP, FP, FN
            if pred:
                TP[true_label] += 1
            else:
                FP[pred_label] += 1
                FN[true_label] += 1
        # Calculate precision, recall, F1 for each class and macro F1
        
        precision = [0] * num_classes
        recall = [0] * num_classes
        f1 = [0] * num_classes
        # 使用宏观平均，首先计算每个类的precision和recall，最后进行平均化
        for i in range(num_classes):
            precision[i] = TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0
            recall[i] = TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        macro_f1 = sum(f1) / num_classes
        acc = sum(acc) / len(acc)
            
            
        return {'acc': acc, 'macro_f1': macro_f1}

    # def dpp_sample(self, L, k):
    #     """
    #     使用DPP进行采样
    #     L: 核矩阵
    #     k: 需要选择的示例数量
    #     """
    #     N = L.shape[0]
    #     # print(L.device)
        
    #     # 特征分解
    #     eigenvalues, eigenvectors = torch.linalg.eigh(L)
        
    #     # 计算每个特征向量被选中的概率
    #     probs = eigenvalues / (eigenvalues + 1)
        
    #     # 第一阶段：确定要选择多少个特征向量
    #     selected_eigenvectors = []
    #     for i in range(N-1, -1, -1):
    #         if len(selected_eigenvectors) < k and torch.rand(1).to(L.device) < probs[i]:
    #             selected_eigenvectors.append(eigenvectors[:, i])
        
    #     # 第二阶段：从选中的特征向量确定具体的样本
    #     selected_indices = []
    #     remaining = list(range(N))
        
    #     while len(selected_indices) < k and remaining:
    #         # 计算条件概率
    #         probs = []
    #         for i in remaining:
    #             if len(selected_indices) == 0:
    #                 # 对于第一个选择，直接使用对角线元素
    #                 prob = L[i, i]
    #             else:
    #                 # 计算条件概率
    #                 prev_selected = torch.tensor(selected_indices)
    #                 sub_L = L[prev_selected][:, prev_selected]
    #                 v = L[i, prev_selected]
    #                 prob = L[i, i] - torch.dot(v, torch.linalg.solve(sub_L, v))
    #             probs.append(max(0, prob.item()))
            
    #         # 归一化概率
    #         probs = torch.tensor(probs)
    #         if probs.sum() == 0:
    #             break
    #         probs = probs / probs.sum()
            
    #         # 采样下一个索引
    #         idx = torch.multinomial(probs, 1).item()
    #         selected_indices.append(remaining[idx])
    #         remaining.pop(idx)
        
    #     return selected_indices

    

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