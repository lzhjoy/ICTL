# 数据集构建

## basetask.py

这是一个名为 `BaseTask` 的基础任务类，它继承自 PyTorch 的 `Dataset` 类。这个类主要用于创建数据集的基础结构。让我们逐部分分析：

1. 类的定义和初始化：
```python
class BaseTask(Dataset):
    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name
        self.task_type = None
        self.all_data = None
```
- 类继承自 `torch.utils.data.Dataset`
- 初始化时需要传入一个 `task_name` 参数
- 设置了三个初始属性：
  - `task_name`：任务名称
  - `task_type`：任务类型（初始为None）
  - `all_data`：存储所有数据（初始为None）

2. 数据下载方法：
```python
def download(self):
    raise NotImplementedError
```
- 这是一个抽象方法，需要在子类中实现
- 用于下载或加载数据集

3. 必要的Dataset方法：
```python
def __len__(self):
    return len(self.all_data)

def __getitem__(self, index):
    return self.all_data[index]
```
- `__len__`：返回数据集的长度
- `__getitem__`：允许通过索引访问数据集中的元素

这个类是一个抽象基类，主要用于：
1. 定义了创建数据集的基本结构
2. 提供了一个统一的接口，其他具体的任务类可以继承这个基类
3. 确保了所有子类都必须实现必要的数据集操作方法

要使用这个类，需要创建一个子类并实现 `download` 方法，同时确保正确设置 `all_data`。


## dataset类（以agnews.py为例）

1. **类的基本结构**：
```python
class AGNews(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_type = "classification"  # 设置任务类型为分类任务
```
这个类继承自之前的`BaseTask`，专门用于处理AG News数据集（一个新闻文本分类数据集）。

2. **模板定义**：
```python
def get_dmonstration_template(self):
    template = {
        'input': 'Sentence: {sentence}\nLabel:',
        'ans': '{label}',
        'options': ["sports", "technology", "world", "business"],
        'format': ['Sentence:', 'Label:'],
        'instruction': '...'  # 任务说明
    }
    return template
```
定义了数据处理的模板，包括：
- 输入格式
- 答案格式
- 可选的标签类别
- 任务说明

3. **模板应用**：
```python
def apply_template(self, data):
    # 将原始数据转换为指定格式
    template = self.get_dmonstration_template()
    input_str = input_template.replace("{sentence}", data["sentence"])
    answer_str = [ans_template.replace("{label}", options[i]) for i in range(len(options))]
    return input_str, answer_str, label
```

4. **数据下载和处理**：
```python
def download(self):
    random.seed(42)  # 设置随机种子确保可重复性
    k = 1000  # 采样数量
    
    # 加载AG News数据集
    agnews_dataset = load_dataset('ag_news')['train']
    
    # 标签映射字典
    agnews_label2text = {
        0: 'world',
        1: 'sports',
        2: 'business',
        3: 'technology'
    }
    
    # 处理数据
    agnews_data = []
    for d in tqdm(agnews_dataset, desc="agnews"):
        data = {
            'id': id,
            'sentence': d['text'],
            'label': agnews_label2text[d['label']]
        }
        agnews_data.append(data)
    
    # 随机采样并保存
    agnews_data_sampled = random.sample(agnews_data, k)
    write_jsonl(agnews_data_sampled, os.path.join(source_cross_task_save_dir, 'agnews.jsonl'))
```

主要功能：
1. 从Hugging Face的datasets库加载AG News数据集
2. 将数字标签转换为文本标签（如0→'world'）
3. 为每条数据添加唯一ID
4. 随机采样1000条数据
5. 将处理后的数据保存为JSONL格式

这个类的主要用途是：
- 下载和预处理AG News数据集
- 将原始数据转换为统一的格式
- 提供模板化的数据处理方法
- 支持文本分类任务的数据准备

这是一个典型的数据处理管道，用于准备机器学习模型的训练数据。


## __init__.py

这个程序为所有的数据集提供了一个统一的接口，通过`get_dataset`函数可以获取到对应的数据集类。