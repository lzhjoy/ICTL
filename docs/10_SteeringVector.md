# Steering Vector
## 1 核心思想
利用数据集A的丰富样本提取某种引导向量（steering vector），在不进行微调的情况下，提升数据集B在样本稀缺场景下的性能。

## 2 实现细节
### 2.1 设计理念

#### 2.1.1 demon_index的选取

对于tar_data中的m个样本，我们使用相应的shot_method，从src_data中选取shot_num + 1个样本，这shot_num + 1个样本的索引即为我们需要的demon_index，保存在data/processed/demon_index/{shot_num}/{shot_method}/{model_name}/{tar_data_name}_{src_data_name}路径下，和结果路径结构相同。


#### 2.1.2 引导向量的提取

利用demon_index，我们可以从src_data中提取出对应的shot_num + 1个样本，其中选择shot_num个样本作为demonstration，一个样本作为question_str，分别构造m个使用demonstration的prompt和不使用demonstration的prompt，使用模型计算出steering vector。









