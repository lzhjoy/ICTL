import json
import numpy as np

datasets = ["arc_challenge", "financial_phrasebank", "medmcqa",  "sciq", "social_i_qa"]

method = "zero_shot_ins-debug"
use_baselines = True

# method = "task_vector"
# use_baselines = False

# method = "function_vector_inter_best"
# use_baselines = False

model = "llama3.1-8b"
# model = "llama2-7b"

all_res_path = f"exps/{method}/{model}"

for dataset in datasets:
    
    print(dataset)
    
    results_path = f"{all_res_path}/{dataset}/result_dict.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    test_result = results["test_result"]
            
    method_results = test_result["zero_shot"]
    method_acc_res = []
    method_f1_res = []
    for method_result in method_results:
        method_acc_res.append(method_result["acc"])
        method_f1_res.append(method_result["macro_f1"])
        
    method_acc_res = [round(a * 100, 2) for a in method_acc_res]
    method_f1_res = [round(a * 100, 2) for a in method_f1_res]
    
    method_acc_avg = round(np.mean(method_acc_res), 2)
    method_acc_std = round(np.std(method_acc_res), 2)
    method_f1_avg = round(np.mean(method_f1_res), 2)
    method_f1_std = round(np.std(method_f1_res), 2)
    
    print(method)
    print(f'ACC: avg: {method_acc_avg}, std: {method_acc_std}.')
    print(f'F1: avg: {method_f1_avg}, std: {method_f1_std}.')
    
    print()