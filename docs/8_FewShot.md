# Few-shot 

## 1. evaluate中参数的更改

添加了四个参数：
- use_random: 是否使用随机选择
- use_topk: 是否使用topk选择
- use_dpp: 是否使用dpp选择
- k: 选择示例的数量

## 2. 重点使用的参数

- self.demon_info: 示例信息，包括示例的文本、标签和embedding，一个列表，列表中的每一个元素是一个字典，字典中包含示例的文本、标签和embedding
- self.demon_data: 示例数据，用于计算embedding，一个列表，列表中的每一个元素是一个字典，字典中包含示例的文本、标签和id
- self.test_data: 测试数据，用于计算embedding，一个列表，列表中的每一个元素是一个字典，字典中包含query的文本、标签和id
- self.result_dict: 用于维护结果

## 3. 各种抽样方法

### 3.1 随机抽样
```python
if use_random:
    # 随机选择k个示例
    selected_demons = random.sample(self.demon_info, k)
    # 构建示例字符串
    demonstrations = []
    for demon in selected_demons:
        demon_str = f"Question: {demon['demon']}\nAnswer: {demon['label']}\n"
        demonstrations.append(demon_str)
    # 将所有示例连接成一个字符串
    demonstration = "".join(demonstrations)
```

### 3.2 topk抽样
```python
elif use_topk:
    # 计算问题与所有示例的余弦相似度
    similarities = F.cosine_similarity(
        ques_embed, 
        torch.stack(demon_embed),
        dim=1
    )
    # 获取相似度最高的k个示例的索引
    _, top_k_indices = torch.topk(similarities, k)
    # 构建示例字符串
    demonstrations = []
    for idx in top_k_indices:
        demon = self.demon_info[idx]
        demon_str = f"Question: {demon['demon']}\nAnswer: {demon['label']}\n"
        demonstrations.append(demon_str)
    # 将所有示例连接成一个字符串
    demonstration = "".join(demonstrations)
```
### 3.3 dpp抽样

**行列式点过程（Determinantal Point Process, DPP）** 是一种概率模型，用于从候选集合中选择具有高质量（相关性）和多样性（差异性）的子集。其核心思想是通过核矩阵（Kernel Matrix）的数学性质，量化样本之间的相似性，并为多样性高的子集赋予更高的概率。以下是DPP的详细解析：

---

#### 1. **基本概念**
- **目标**：从候选集 $ \mathcal{Y} = \{y_1, y_2, \dots, y_N\} $ 中选择子集 $ S \subseteq \mathcal{Y} $，使得 $ S $ 中的元素既与任务相关，又彼此不同。
- **核心公式**：  
  子集 $ S $ 的概率与其核矩阵 $ L $ 的行列式成正比：  
  $$
  P(S) \propto \det(L_S)
  $$  
  其中，$ L_S $ 是核矩阵 $ L $ 中对应子集 $ S $ 的块矩阵。

---

#### 2. **核矩阵的设计**
核矩阵 $ L $ 的构造是关键，通常分解为两个部分：  
$$
L_{i,j} = q_i \cdot \phi(y_i)^\top \phi(y_j) \cdot q_j
$$  
- **质量项（Quality Term）$ q_i $**：衡量单个样本 $ y_i $ 的与当前任务的相关性（例如，与输入问题的语义相似度）。  
- **相似度项（Similarity Term）$ \phi(y_i)^\top \phi(y_j) $**：衡量两个样本 $ y_i $ 和 $ y_j $ 的相似性（例如，余弦相似度）。  

通过这种分解，DPP 可以同时优化相关性和多样性。

---

#### 3. **数学性质与多样性**
- **行列式的意义**：  
  $\det(L_S)$ 的值与子集 $ S $ 中样本的“体积”（即多样性）相关。若样本间高度相似，核矩阵的行列式值较小，概率降低；若样本差异显著，行列式值较大，概率升高。  
- **排斥性（Repulsiveness）**：  
  DPP 倾向于选择彼此差异大的样本，避免冗余。

---

#### 4. dpp采样
该部分代码位于evaluator.py中的dpp_sample函数中。
```python
# 第二阶段：从选中的特征向量确定具体的样本
selected_indices = []
remaining = list(range(N))

while len(selected_indices) < k and remaining:
  # 计算条件概率
  probs = []
  for i in remaining:
      if len(selected_indices) == 0:
          # 对于第一个选择，直接使用对角线元素
          prob = L[i, i]
      else:
          # 计算条件概率
          prev_selected = torch.tensor(selected_indices)
          sub_L = L[prev_selected][:, prev_selected]
          v = L[i, prev_selected]
          prob = L[i, i] - torch.dot(v, torch.linalg.solve(sub_L, v))
      probs.append(max(0, prob.item()))
  
  # 归一化概率
  probs = torch.tensor(probs)
  if probs.sum() == 0:
      break
  probs = probs / probs.sum()
  
  # 采样下一个索引
  idx = torch.multinomial(probs, 1).item()
  selected_indices.append(remaining[idx])
  remaining.pop(idx)
```

代码通过 **贪心逐次选择** 样本，每次选择时最大化当前条件概率。这一过程利用了 DPP 的 **链式法则**，将联合概率分解为逐步条件概率的乘积。

##### **1. 首次选择（空集起始）**
- **条件概率**：当无已选样本时，每个样本被选中的概率正比于其 **自身质量**（即核矩阵对角线元素 $ L[i,i] $）。
- **数学依据**：$ P(i \in S) \propto L[i,i] $，反映单个样本的重要性。

##### **2. 后续选择（条件概率更新）**
假设已选子集 $ S $，下一个样本 $ i $ 的条件概率为：
$$
P(i \in S \mid 当前已选 S) = L[i,i] - L[i, S] (L_S)^{-1} L[S, i].
$$
- **几何解释**：从 $ i $ 的自身质量中 **减去与已选样本的冗余性**（投影到已选子空间的部分）。
- **矩阵意义**：此为 **Schur补** 形式，衡量 $ i $ 在已选子集 $ S $ 的补空间中的能量。

---




##### **3. 与 DPP 精确采样对比**
| **特性**       | **序列采样（本文代码）**          | **精确 DPP 采样**               |
|----------------|----------------------------------|----------------------------------|
| 采样方式       | 贪心逐次选择                     | 通过特征分解一次性生成           |
| 计算复杂度     | $ O(Nk^3) $                    | $ O(N^3) $                     |
| 适用场景       | 大规模、需实时性                 | 小规模、理论精确性要求高         |
| 多样性保证     | 近似最优                         | 精确最优                         |


## 4. 源任务应该如何选择

该部分需要在定义完evaluator后需要考虑。