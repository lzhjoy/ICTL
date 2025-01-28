# ZeroShot方法介绍


1. **类的继承结构**：
- `ZeroShot` 类继承自 `BaseMethod` 类
- 通过 `super().__init__()` 调用父类的初始化方法

2. **主要方法**：
- `run()` 方法是该类的核心功能实现
- 首先调用父类的 `run()` 方法
- 设置设备（device）为加速器（accelerator）指定的设备

3. **核心功能**：
- 执行零样本测试评估：
  ```python
  test_zeroshot_result = self.test_evaluator.evaluate(
      self.tokenizer, 
      self.model, 
      use_demonstration=False
  )
  ```
- 评估时不使用示例（`use_demonstration=False`）
- 将测试结果存储在 `result_dict` 中
- 打印测试结果
- 最后将结果保存到 JSON 文件中
