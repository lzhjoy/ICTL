# wrapper

## 阅读顺序

1. 首先了解整体架构：
```python
class ModelWrapper(nn.Module):  # 基类
    def __init__(self, model, tokenizer, model_config, device)
    
class LlamaWrapper(ModelWrapper):  # 具体实现类
class QwenWrapper(ModelWrapper):
class GPTWrapper(ModelWrapper):
```
这是一个典型的面向对象设计，基类定义通用接口，子类实现具体细节。

2. 按功能模块阅读：

a) 基础设施部分：
```python
def _get_nested_attr(self, attr_path)  # 访问模型内部属性
def _get_layer_num(self)               # 获取模型层数
def _get_arribute_path(self, layer_idx, target_module)  # 获取特定层的路径
```

b) 核心功能按此顺序阅读：
```python
def extract_latent(self)      # 特征提取
def inject_latent(self)       # 特征注入
def replace_latent(self)      # 特征替换
def get_context_vector(self)  # 获取上下文向量
```

## ModelWrapper


1. `_get_nested_attr(self, attr_path)`:
- 这是一个工具函数，用于访问模型中的嵌套属性
- 接受一个点分隔的字符串路径，如 'transformer.h' 或 'model.layers'
- 使用 `reduce` 和 `getattr` 递归地访问对象的嵌套属性
- 举例：
```python
# 如果有这样的模型结构：
model = Model()
model.transformer = Transformer()
model.transformer.layers = Layers()

# 可以这样访问：
attr_path = 'transformer.layers'
result = _get_nested_attr(attr_path)  # 等同于 model.transformer.layers
```

2. `_get_layer_num(self)`:
- 这是一个抽象方法，需要在子类中实现
- 用于获取模型中层的数量
- 不同模型架构（如BERT、GPT等）的层数获取方式可能不同，所以需要具体实现
- 示例实现可能是：
```python
def _get_layer_num(self):
    return len(self.model.transformer.layers)  # 具体实现依赖于模型结构
```

3. `_get_arribute_path(self, layer_idx, target_module)`:
- 这也是一个抽象方法，需要在子类中实现
- 用于获取特定层和模块的属性路径
- 参数：
  - layer_idx: 层的索引
  - target_module: 目标模块名称
- 示例实现可能是：
```python
def _get_arribute_path(self, layer_idx, target_module):
    return f"transformer.layers.{layer_idx}.{target_module}"
```

4. `extract_hook_func(self, layer_idx, target_module)`

- 参数说明
```python
def extract_hook_func(self, layer_idx, target_module):
    """
    Args:
        layer_idx: 层的索引，用于标识是哪一层
        target_module: 目标模块名称，如'attention'、'mlp'等
    """
```

- 函数实现细节：
```python
# 首先确保该层的字典存在
if layer_idx not in self.latent_dict:
    self.latent_dict[layer_idx] = {}

# 定义实际的hook函数
def hook_func(module, inputs, outputs):
    # 处理输出是元组的情况（某些模块可能返回多个输出）
    if type(outputs) is tuple:
        outputs = outputs[0]
    
    # 存储处理后的输出
    # detach(): 分离计算图
    # cpu(): 将张量移到CPU
    self.latent_dict[layer_idx][target_module] = outputs.detach().cpu()

return hook_func
```

- 使用示例：
```python
# 假设有一个模型和包装器
model = MyModel()
wrapper = ModelWrapper(model)

# 为特定层注册hook
layer = wrapper._get_nested_attr(wrapper._get_arribute_path(1, 'attention'))
hook = layer.register_forward_hook(wrapper.extract_hook_func(1, 'attention'))

# 前向传播后
outputs = model(inputs)
# self.latent_dict 现在包含了第1层attention模块的输出
# 可以这样访问：
attention_output = wrapper.latent_dict[1]['attention']

# 使用完后记得移除hook
hook.remove()
```
5. `extract_latent(self)`

这个 `extract_latent` 是一个上下文管理器（context manager）函数，用于批量管理模型中间层特征的提取。

1. 装饰器和初始化：
```python
@contextmanager  # 使函数成为上下文管理器，可以使用 with 语句
def extract_latent(self):
    handles = []  # 存储所有hook的句柄，用于后续移除
    self.latent_dict = defaultdict(dict)  # 重置特征存储字典
```

2. 主要功能：
- 为模型的每一层的三个关键模块（attention、mlp、hidden）注册hook
- 使用之前定义的 `extract_hook_func` 来捕获这些模块的输出
- 自动管理hook的注册和移除

3. 使用示例：
```python
model = MyModel()
wrapper = ModelWrapper(model)

# 使用上下文管理器自动处理hook的注册和移除
with wrapper.extract_latent():
    outputs = model(inputs)
    # 此时 wrapper.latent_dict 包含了所有层的 attn、mlp、hidden 输出
    
# with 语句结束后，所有hook自动被移除
```

4. 代码结构解析：
```python
try:
    # 为每一层注册三个hook
    for layer_idx in range(self.num_layers):
        # 注册 attention 模块的hook
        handles.append(
            self._get_nested_attr(self._get_arribute_path(layer_idx, 'attn'))
            .register_forward_hook(self.extract_hook_func(layer_idx, 'attn')))
        
        # 注册 mlp 模块的hook
        handles.append(
            self._get_nested_attr(self._get_arribute_path(layer_idx, 'mlp'))
            .register_forward_hook(self.extract_hook_func(layer_idx, 'mlp')))
        
        # 注册 hidden 状态的hook
        handles.append(
            self._get_nested_attr(self._get_arribute_path(layer_idx, 'hidden'))
            .register_forward_hook(self.extract_hook_func(layer_idx, 'hidden')))
    
    yield  # 暂停执行，允许with语句块中的代码执行
    
finally:
    # 确保所有hook都被移除，即使发生异常
    for handle in handles:
        handle.remove()
```

5. 优点：
- 自动化管理：自动处理hook的注册和清理
- 安全性：使用 try-finally 确保hook被正确移除
- 易用性：通过 with 语句提供简洁的使用方式
- 完整性：同时捕获多个关键模块的输出

6. 数据组织：
- latent_dict 的结构：
```python
{
    layer_idx: {
        'attn': attention_output,
        'mlp': mlp_output,
        'hidden': hidden_state
    }
    # ... 对每一层都有类似的结构
}
```
