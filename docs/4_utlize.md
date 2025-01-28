1. **`read_jsonl(filename)`**
   - 读取 JSONL 格式文件（每行一个 JSON 对象）
   - 将每行解析为 Python 对象，存入列表并返回

2. **`write_jsonl(data, path)`**
   - 将数据写入 JSONL 格式文件
   - 每个数据项转换为 JSON 字符串，写入单独的一行

3. **`transformmodel_name2model_path(model_name)`**
   - 将模型名称映射到对应的模型路径
   - 维护了一个模型名称到实际存储路径的字典映射
   - 目前支持 "llama3.1-8b" 和 "llama2-7b" 两个模型

4. **`init_exp_path(config, dataset_name)`**
   - 初始化实验目录结构
   - 创建保存路径：`{exp_name}/{model_name}/{dataset_name}`
   - 保存实验配置到 config.json
   - 防止非调试实验被意外覆盖(借鉴！！！)

5. **`load_model_tokenizer(config, accelerator, output_hidden_states=True, load_in_8bit=False)`**
   - 加载模型和分词器
   - 设置模型配置，包括 tokenizer 的填充设置
   - 特别处理 LLaMA 模型的配置信息
   - 返回模型、分词器、模型配置和特定模型配置

6. **`get_model_wrapper(config, model, tokenizer, model_config, accelerator)`**
   - 根据模型类型获取对应的模型包装器
   - 支持 LLaMA 和 GPT 系列模型
   - 返回相应的模型包装器实例

7. **`load_config(file_path)`**
   - 动态加载配置文件
   - 将配置文件所在目录添加到系统路径
   - 导入配置模块并获取其中的 config 变量

8. **`last_one_indices(tensor)`**
   - 在二维张量中找到每行最后一个 1 的索引位置
   - 如果某行全为 0，则返回 -1
   - 主要用于处理注意力机制或掩码相关的操作

这些函数主要用于：
- 数据处理（JSONL 文件的读写）
- 模型管理（加载和初始化模型）
- 实验管理（创建实验目录和保存配置）
- 模型包装（提供统一的接口）
- 配置管理（加载配置文件）
- 张量操作（处理特定的张量计算需求）

这个工具文件提供了项目中常用的基础功能，使得主要代码可以更清晰地关注核心逻辑。
