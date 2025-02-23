#!/bin/bash

# 数据集列表

# all
# src_datasets=("arc_challenge" "financial_phrasebank" "medmcqa" "sciq" "social_i_qa")
# tar_datasets=("agnews" "arc_easy" "boolq" "commensense_qa" "mnli" "qqp" "race" "sst2")


# part
src_datasets=("arc_challenge" "financial_phrasebank")
tar_datasets=("arc_easy" "commensense_qa" "sst2")

# 循环每个数据集并运行
for src_dataset in "${src_datasets[@]}"
do
    for tar_dataset in "${tar_datasets[@]}"
    do
        CUDA_VISIBLE_DEVICES=7,8 /home/tangxinyu/.conda/envs/re4r/bin/python src/run_few_shot.py --src_datasets $src_dataset --tar_datasets $tar_dataset
    done
done

echo "All datasets have been processed."