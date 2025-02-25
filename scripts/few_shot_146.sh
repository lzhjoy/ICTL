#!/bin/bash

# 数据集列表

# all
# src_datasets=("arc_challenge" "financial_phrasebank" "medmcqa" "sciq" "social_i_qa")
# tar_datasets=("agnews" "arc_easy" "boolq" "commensense_qa" "mnli" "qqp" "race" "sst2")


# part
src_datasets=("arc_challenge" "financial_phrasebank")
tar_datasets=("arc_easy" "commensense_qa" "sst2")
shot_nums=(1 2 3 4 5)
shot_methods=("dpp" "topk" "random")

# 循环每个数据集并运行
for src_dataset in "${src_datasets[@]}"
do
    for tar_dataset in "${tar_datasets[@]}"
    do
        for shot_num in "${shot_nums[@]}"
        do
            for shot_method in "${shot_methods[@]}"
            do
                CUDA_VISIBLE_DEVICES=7,8 /home/tangxinyu/.conda/envs/re4r/bin/python src/run_few_shot.py --src_datasets $src_dataset --tar_datasets $tar_dataset --shot_num $shot_num --shot_method $shot_method
            done
        done
    done
done

echo "All datasets have been processed."