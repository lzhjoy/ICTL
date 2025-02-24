import json
import pandas as pd
from pathlib import Path
import itertools

# 定义参数组合
shot_num = [1, 2, 3, 4, 5, 6]
shot_method = ['random', 'topk', 'dpp']
model_name = ['llama3.1-8b']
tar_data_name = ["arc_challenge", "financial_phrasebank"]
src_data_name = ["arc_easy", "commensense_qa", "sst2"]

# 创建存储结果的列表
results = []

# 遍历所有组合
for shot, method, model, tar_data, src_data in itertools.product(
    shot_num, shot_method, model_name, tar_data_name, src_data_name
):
    # 构建文件路径
    file_path = f"exps/few_shot-debug/{shot}_shot/{method}/{model}/{tar_data}_{src_data}/result_dict.json"
    
    try:
        # 读取JSON文件
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 获取few_shot的结果
        if data['test_result']['few_shot']:
            acc = data['test_result']['few_shot'][0]['acc']
            macro_f1 = data['test_result']['few_shot'][0]['macro_f1']
            
            # 添加到结果列表
            results.append({
                'Shot': shot,
                'Method': method,
                'Model': model,
                'Target Dataset': tar_data,
                'Source Dataset': src_data,
                'Accuracy': acc,
                'Macro F1': macro_f1
            })
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

# 创建DataFrame
df = pd.DataFrame(results)

# 为每个目标数据集创建单独的CSV文件
for tar_data in tar_data_name:
    tar_df = df[df['Target Dataset'] == tar_data]
    
    # 为每个source dataset创建单独的CSV文件
    for src_data in src_data_name:
        sheet_df = tar_df[tar_df['Source Dataset'] == src_data]
        if not sheet_df.empty:
            # 数据透视表：行为shot number，列为method
            pivot_df = sheet_df.pivot_table(
                values=['Accuracy', 'Macro F1'],
                index='Shot',
                columns='Method',
                aggfunc='first'
            )
            # 保存为CSV文件
            csv_path = f"output/analysis/few_shot/analysis_results_{tar_data}_{src_data}.csv"
            pivot_df.to_csv(csv_path)

print("Analysis complete. Results saved to CSV files.")











