import sys
import os
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F

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
                tar_instruction = self.tar_ds_class.get_dmonstration_template()['instruction']
            
            all_pred_labels = []
            all_inputs, all_labels = [], []
            for data in self.dataset:
                ques_str, _, label = self.src_ds_class.apply_template(data)
                if not use_demonstration:
                    if self.config['use_instruction']:
                        context = f"Definition: {src_instruction}\n{ques_str}"
                    else:
                        context = ques_str
                else:
                    # TODO：需要完善一下使用demonstration的逻辑
                    ques_embed = self.sentence_model.encode([ques_str], convert_to_tensor=True)
                    demon_embed = [demon['embed'] for demon in self.demon_info]
                    if self.config['use_instruction']:
                        context = f"Definition: {tar_instruction}\n{demonstration}\nDefinition: {src_instruction}\n{ques_str}"
                    else:
                        context = demonstration + "\n" + ques_str
                    
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