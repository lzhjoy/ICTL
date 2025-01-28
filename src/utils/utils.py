import json
import os
import torch
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import wrapper

def read_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            
def transformmodel_name2model_path(model_name):
    model_name2model_path = {
        "llama3.1-8b": "/mnt/tangxinyu/huggingface/models/meta-llama/Meta-Llama-3.1-8B/",
        "llama2-7b": "/data/models/Meta-Llama-2-7B/"
    }
    model_path = model_name2model_path[model_name]
    return model_path

def init_exp_path(config, dataset_name):
    save_dir = os.path.join(config['exp_name'], config['model_name'], dataset_name)
    if os.path.exists(save_dir) and 'debug' not in config['exp_name']:
        raise ValueError(f"Experiment {config['exp_name']} already exists! please delete it or change the name!")
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    return save_dir


def load_model_tokenizer(config, accelerator, output_hidden_states=True, load_in_8bit=False):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # model
    model = AutoModelForCausalLM.from_pretrained(config['model_path'], device_map="auto", output_hidden_states=output_hidden_states, load_in_8bit=load_in_8bit, torch_dtype=torch.float32, trust_remote_code=True).eval()
    model = accelerator.prepare(model)
    # config
    model_config = AutoConfig.from_pretrained(config['model_path'])
    
    if "llama" in config['model_path'].lower():
        MODEL_CONFIG = {
            "head_num": model.config.num_attention_heads,
            "layer_num": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            # 注意力输出投影层的 hook 名称列表
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            # 每一层的 hook 名称列表
            "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "prepend_bos":True
        }
    
    return model, tokenizer, model_config, MODEL_CONFIG

def get_model_wrapper(config, model, tokenizer, model_config, accelerator):
    
    device = accelerator.device
    if 'llama' in config['model_name']:
        model_wrapper = wrapper.LlamaWrapper(model, tokenizer, model_config, device)
    elif 'gpt' in config['model_name']:
        model_wrapper = wrapper.GPTWrapper(model, tokenizer, model_config, device)
    else:
        raise ValueError("only support llama or gpt!")
    return model_wrapper

def load_config(file_path):
    if not file_path:
        raise ValueError("No file path provided")
    file_dir = os.path.dirname(file_path)
    if file_dir not in sys.path:
        sys.path.append(file_dir)
    file_name = os.path.basename(file_path)
    module_name = os.path.splitext(file_name)[0]
    module = __import__(module_name)
    try:
        my_variable = getattr(module, 'config')
        print(my_variable)
        return my_variable
    except AttributeError:
        print(f"The module does not have a variable named 'config'")

def last_one_indices(tensor):
    # 找到一个二维张量中每一行最后一个 1 的索引，若全为0则返回-1
    """
    Finds the index of the last 1 in each row of a 2D tensor.

    Args:
      tensor (torch.Tensor): A 2D tensor of size (N, M) containing only 0 and 1 entries.

    Returns:
      torch.Tensor: A tensor of size N containing the index of the last 1 in each row.
                    If a row contains only 0s, the index will be set to -1 (or a sentinel value of your choice).
    """
    # Reverse each row to find the last occurrence of 1 (which becomes the first in the reversed row)
    reversed_tensor = torch.flip(tensor, [1])
    # Check for rows containing only zeros in the reversed tensor
    is_all_zero = reversed_tensor.sum(dim=1) == 0
    # Get the index of the first occurrence of the maximum value (1) along each row in the reversed tensor
    indices = reversed_tensor.argmax(dim=1) 
    # Adjust the indices for the original order of each row
    indices = tensor.size(1) - 1 - indices
    # Handle rows with all zeros
    indices[is_all_zero] = -1  # Set to -1 to indicate no '1' found in these rows
    return indices