#!/bin/bash

# 数据集列表
datasets=("arc_challenge" "financial_phrasebank" "medmcqa" "sciq" "social_i_qa")

# 循环每个数据集并运行
for dataset in "${datasets[@]}"
do
    CUDA_VISIBLE_DEVICES=7 /home/tangxinyu/anaconda3/envs/ictl/bin/python src/run_few_shot.py --datasets $dataset
done

echo "All datasets have been processed."