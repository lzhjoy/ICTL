import json
import re
import os
import sys
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np
from collections import defaultdict
from copy import copy
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from accelerate import Accelerator

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, RE4R_ROOT_PATH)

RE4R_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
sys.path.insert(0, RE4R_ROOT_PATH)

from src.utils.utils import read_jsonl, write_jsonl, transformodel_name2model_path, load_model_tokenizer, get_model_wrapper
from src.utils.evaluator import MATHEvaluator

step_pattern = r'(Step \d+:.*?)(?=Step \d+:|$)'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()

    # ==== model & dataset ====
    parser.add_argument('--model_name', type=str, default='llama3.1-8b-instruct', help='path to config file')
    parser.add_argument('--datasets', type=str, help="string of datasets", default="mathoai")
    parser.add_argument('--max_generated_token', type=int, default=3000, help="max generated token")
    
    # ==== intervention ====
    parser.add_argument('--module', type=str, default="hidden", help="inject vector to which module, attn / mlp / hidden")
    parser.add_argument('--extract_pos', type=str, default="last", help="extract vector from which position, first / last / random")
    parser.add_argument('--inject_method', type=str, default="linear", help="inject method, linear / add / balance")
    parser.add_argument('--inject_pos', type=str, default="first", help="inject vector to which position, first / last / random")
    parser.add_argument('--layer', type=int, default=0, help="layer to inject")
    parser.add_argument('--strength', type=str, default="0,1", help="strength to inject")
    parser.add_argument("--step_level_sample_num", type=int, default=1, help="number of similar steps to sample")

    # ==== evluation ====
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--sample', action='store_true', help='sample mode')
    return parser.parse_args()


def sub_vector(dict1, dict2):
    new_dict = {}
    for key in dict1.keys():
        new_dict[key] = {}
        for module in dict1[key].keys():
            new_dict[key][module] = dict1[key][module] - dict2[key][module]
    return new_dict


def get_step_level_thought_bank(step_level_data, tokenizer, model, model_wrapper, device, instruction, knowledge_vec_config, layer):
    
    step_level_thought_bank_list = []
    # problem: input_str, step: current_steo, context: history_step, last_step_id: last_step_id, step_id: current_step_id, step_token_id: current_step_token_id

    for case_id, d in tqdm(enumerate(step_level_data), desc="cache vector"):

        problem = d['problem']
        input_str = problem + instruction
        input_str = input_str.strip()

        step_level_thought = d['step_level']
        step_list = step_level_thought.split("\n\n---\n\n")

        # empty thought
        step_thought_list = [
            {'role': 'user', 'content': input_str},
            {'role': 'assistant', 'content': ""}
        ]
        step_thought_str = tokenizer.apply_chat_template(step_thought_list, tokenize=False)
        tokenized_replacement = tokenizer.encode(step_thought_str, add_special_tokens=False)
        current_step_token_id = copy(len(tokenized_replacement))
        
        demon_step_all_latent_dicts = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                tokenized_replacement = tokenizer(step_thought_str, return_tensors='pt').to(device)
                current_step_token_id = copy(len(tokenized_replacement['input_ids'][0]))
                _ = model(**tokenized_replacement)
            demon_step_all_latent_dicts.append(model_wrapper.latent_dict)
        demon_step_context_vector_dict = model_wrapper.get_context_vector(demon_step_all_latent_dicts, knowledge_vec_config)
        demon_step_context_vector_dict = {key: value for key, value in demon_step_context_vector_dict.items() if int(key) == layer}

        last_demon_step_context_vector_dict = copy(demon_step_context_vector_dict)

        for i, step in enumerate(step_list):
            context = copy(step_thought_str)
            last_step_token_id = copy(current_step_token_id)
            step = step.strip()
            step_thought_str += step
            tokenized_replacement = tokenizer.encode(step_thought_str, add_special_tokens=False)

            demon_step_all_latent_dicts = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    tokenized_replacement = tokenizer(step_thought_str, return_tensors='pt').to(device)
                    current_step_token_id = copy(len(tokenized_replacement['input_ids'][0]))
                    _ = model(**tokenized_replacement)
                demon_step_all_latent_dicts.append(model_wrapper.latent_dict)

            demon_step_context_vector_dict = model_wrapper.get_context_vector(demon_step_all_latent_dicts, knowledge_vec_config)
            demon_step_context_vector_dict = {key: value for key, value in demon_step_context_vector_dict.items() if int(key) == layer}
            diff_vector = sub_vector(demon_step_context_vector_dict, last_demon_step_context_vector_dict)

            step_level_thought_bank_list.append({
                "case_id": case_id,
                "step_id": i+1,
                "step": step,
                "context": context,
                "step_token_id": current_step_token_id,
                "last_step_token_id": last_step_token_id,
                "vector": copy(diff_vector),
            })

            last_demon_step_context_vector_dict = copy(demon_step_context_vector_dict)

            if i != len(step_list) - 1:
                step_thought_str += "\n\n"

    return step_level_thought_bank_list


def main():
    
    # ========== initialize ==========
    
    # get args
    args = get_args()
    accelerator = Accelerator()
    device = accelerator.device
    math_evaluator = MATHEvaluator()
    
    model_name = args.model_name
    step_level_sample_num = args.step_level_sample_num
    model_path = transformodel_name2model_path(model_name)
    model, tokenizer, model_config, MODEL_CONFIG = load_model_tokenizer(model_path, accelerator, output_hidden_states=True, load_in_8bit=False)
    max_generated_token = args.max_generated_token
    model_wrapper = get_model_wrapper(model_name, model, tokenizer, model_config, accelerator)
    
    step_level_thought_path = "data/math/longcot/step_level_thought.jsonl"
    data_path = "data/math/mathoai/mathoai.jsonl"
    data = read_jsonl(data_path)
    all_problems, all_inputs = [], []
    all_labels = []
    all_levels, all_unique_ids = [], []
    layer = args.layer
    strength = args.strength.split(',')
    strength = [float(s) for s in strength]

    knowledge_vec_config = {}

    knowledge_vec_config['module'] = args.module
    knowledge_vec_config['tok_pos'] = args.extract_pos
    knowledge_vec_config['inject_method'] = args.inject_method
    knowledge_vec_config['inject_pos'] = args.inject_pos
    strength = args.strength.split(',')
    strength = [float(s) for s in strength]
    knowledge_vec_config['strength'] = strength
    
    instruction = "Answer the following question step by step and put the final answer in \\boxed{}:\n"
    
    # ========== process test dataset ==========

    if args.debug:
        data = data[:10]
    if args.sample:
        data = data[:100]

    
    for d in data:
        problem = d['problem']
        answer = d['solution']
        level = d['level']
        unique_id = d['unique_id']
        input_str = instruction + problem
        input_str = input_str.strip()
        all_problems.append(problem)
        all_inputs.append(input_str)
        all_labels.append(answer)
        all_levels.append(level)
        all_unique_ids.append(unique_id)


    # ========== get step level thought bank ==========
    step_level_data = read_jsonl(step_level_thought_path)
    if args.debug:
        step_level_data = step_level_data[:2]
    step_level_thought_bank_list = get_step_level_thought_bank(step_level_data, tokenizer, model, model_wrapper, device, instruction, knowledge_vec_config, layer)

    # TODO: 改到这边，等回来再弄相似度向量的注入 (不需要前面向量的注入，直接逐个Step，搜索k个向量，沿着特定方向进行注入)
    # ========== inject step-level thought vector ==========
    all_pred_labels, accuracies = [], []

    eos_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")

    for sample_id, (input, label) in tqdm(enumerate(zip(all_inputs, all_labels)), desc="inject step-level vector"):
        
        case_info = []

        input_list = [
            {"role": "user", "content": input},
            {"role": "assistant", "content": ""}
        ]
        input_str = tokenizer.apply_chat_template(input_list, tokenize=False)
        last_step_response = ""

        last_token_list = []
        with torch.no_grad():
            with model_wrapper.extract_latent():
                input_token = tokenizer(input_str, return_tensors='pt').to(device)
                _ = model(**input_token)
            last_token_list.append(model_wrapper.latent_dict)

        last_token_vector_dict = model_wrapper.get_context_vector(last_token_list, knowledge_vec_config)
        last_token_vector_dict = {key: value for key, value in last_token_vector_dict.items() if int(key) == layer}

        while True:

            input_tok = tokenizer([input_str], return_tensors="pt")
            input_ids = input_tok['input_ids'].to(device)

            # 如果有kv_cache，则传递给generate方法
            generate_kwargs = {
                'max_new_tokens': 1024,
                'pad_token_id': tokenizer.eos_token_id,
                'num_return_sequences': 1,
                'stop_strings': ["Step"],
                'tokenizer': tokenizer,
            }

            step_response_ids = model.generate(input_ids, **generate_kwargs)
            try_step_response = tokenizer.decode(step_response_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            if try_step_response.endswith("Step"):
                try_step_response = try_step_response[:-len("Step")].rstrip()

            try_step_all_str = input_str + "\n\n" + try_step_response
            
            try_step_all_list = []
            with torch.no_grad():
                with model_wrapper.extract_latent():
                    try_step_token = tokenizer(try_step_all_str, return_tensors='pt').to(device)
                    _ = model(**try_step_token)
                try_step_all_list.append(model_wrapper.latent_dict)
            
            try_step_context_vector_dict = model_wrapper.get_context_vector(try_step_all_list, knowledge_vec_config)
            try_step_context_vector_dict = {key: value for key, value in try_step_context_vector_dict.items() if int(key) == layer}

            current_vector_dict = sub_vector(last_token_vector_dict, try_step_context_vector_dict)

            # ========== find step-level thought demon & vector ==========
            sim_step_level_thought_list = []
            step_level_vec = [step_level['vector'] for step_level in step_level_thought_bank_list]
            step_level_vector = [res[layer][knowledge_vec_config['module']] for res in step_level_vec]
            step_level_embed = torch.stack(step_level_vector, dim=0)
            # 确保 try_step_context_vector 是二维张量
            current_vectors = current_vector_dict[layer][knowledge_vec_config['module']]
            current_vectors = current_vectors.unsqueeze(0)

            # 计算余弦相似度
            cos_sim = F.cosine_similarity(step_level_embed, current_vectors)
            top_k_similarities, top_k_indices = torch.topk(cos_sim, step_level_sample_num)

            # TODO: average 多个向量注入 (待实现)
            for sim_id in top_k_indices:
                sim_step_level_thought_list.append(step_level_vector[sim_id.item()])

            step_step_list = []
            for sim_id in top_k_indices:
                step_step_list.append(step_level_thought_bank_list[sim_id.item()]['step'])

            sim_step_level_thought_tensor = torch.stack(sim_step_level_thought_list, dim=0)  # 将向量堆叠成一个 tensor
            mean_vector = sim_step_level_thought_tensor.mean(dim=0)  # 沿着第 0 维计算均值

            # 存储结果为字典
            mean_vector_dict = {layer: {knowledge_vec_config['module']: mean_vector}}
            
            # ========== inject step-level thought vector ==========

            with model_wrapper.inject_latent(mean_vector_dict, [layer], knowledge_vec_config):
                knowledge_augmented_step_response_ids = model.generate(input_ids, **generate_kwargs)
            
            knowledge_augmented_step_response = tokenizer.decode(knowledge_augmented_step_response_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            accumulated_token = len(knowledge_augmented_step_response_ids[0])

            case_info.append(
                {
                    'top_k_similarities': top_k_similarities.tolist(),
                    'top_k_indices': top_k_indices.tolist(),
                    'input_str': input_str,
                    'sim_step_list': step_step_list,
                    'try_step_response': try_step_response,
                    'injected_step_response': knowledge_augmented_step_response
                }
            )

            input_str += copy(knowledge_augmented_step_response)
            if accumulated_token > max_generated_token or eos_token_id in knowledge_augmented_step_response_ids[0] or knowledge_augmented_step_response == last_step_response:
                del mean_vector_dict
                break
            
            last_step_response = copy(knowledge_augmented_step_response)
            del mean_vector_dict

        final_pred_ans = input_str
        all_pred_labels.append(final_pred_ans)
        sample_acc = math_evaluator.score(final_pred_ans, label)
        accuracies.append(sample_acc)

        # ========== save results ==========
        save_dir = f"info/method/re_step_vector_{step_level_sample_num}_{args.inject_pos}_{layer}_{'-'.join([str(s) for s in strength])}"
        os.makedirs(save_dir, exist_ok=True)
        write_jsonl(case_info, f"{save_dir}/case_info_{sample_id}.jsonl")



    results = []
    print('layer', layer, 'strength', strength, "acc", sum(accuracies)/len(accuracies))
    for problem, pred_label, label, level, unique_id, acc in zip(all_problems, all_pred_labels, all_labels, all_levels, all_unique_ids, accuracies):
        results.append({'problem': problem, 'pred_label': pred_label, 'label': label, 'level': level, 'unique_id': unique_id, 'acc': acc})
    
    save_dir = f"exp/method/re_step_vector"
    os.makedirs(save_dir, exist_ok=True)
    write_jsonl(results, f"{save_dir}/results_{step_level_sample_num}_{args.inject_pos}_{layer}_{'-'.join([str(s) for s in strength])}.jsonl")
    
if __name__ == "__main__":
    set_seed(42)
    main()