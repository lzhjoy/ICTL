
# 数据集
## 多任务数据集

### **一、Source 数据集**
#### 1. **agnews.jsonl**  
- **全称**：AG News  
- **领域**：新闻文本分类  
- **内容**：包含 120 万条新闻标题和摘要，分为 4 类：`World`, `Sports`, `Business`, `Sci/Tech`。  
- **任务**：文本分类（四分类），常用于测试模型对新闻主题的识别能力。

#### 2. **arc_easy.jsonl**  
- **全称**：AI2 Reasoning Challenge (ARC) - Easy  
- **领域**：科学问答  
- **内容**：面向小学科学考试的问答数据集，问题需基于科学知识推理，但难度较低。  
- **任务**：开放域问答，常用于评估模型的基础推理能力。

#### 3. **boolq.jsonl**  
- **全称**：Boolean Questions (BoolQ)  
- **领域**：阅读理解  
- **内容**：二元（Yes/No）问答对，每个问题需通过阅读给定段落回答。  
- **任务**：问答与文本蕴含，测试模型对段落逻辑的理解。

#### 4. **commonsense_qa.jsonl**  
- **全称**：CommonsenseQA  
- **领域**：常识推理  
- **内容**：需依赖常识（如物理、社会常识）回答的选择题，例如“水结冰后会变成什么？”  
- **任务**：多选问答，评估模型的常识推理能力。

#### 5. **mnli.jsonl**  
- **全称**：MultiNLI (Multi-Genre Natural Language Inference)  
- **领域**：自然语言推理  
- **内容**：句子对标注为 `entailment`, `contradiction`, 或 `neutral`。  
- **任务**：文本对关系判断，是NLP中的经典基准任务。

#### 6. **qqp.jsonl**  
- **全称**：Quora Question Pairs  
- **领域**：语义相似度  
- **内容**：来自 Quora 的 40 万对问题，标注是否语义相同。  
- **任务**：二分类（相似/不相似），用于训练语义匹配模型。

#### 7. **race.jsonl**  
- **全称**：RACE (ReAding Comprehension from Examinations)  
- **领域**：阅读理解  
- **内容**：中国初高中英语考试中的文章和问题，答案需从多选项中选出。  
- **任务**：机器阅读理解，测试长文本理解能力。

#### 8. **sst2.jsonl**  
- **全称**：Stanford Sentiment Treebank (SST-2)  
- **领域**：情感分析  
- **内容**：电影评论句子，标注为积极（Positive）或消极（Negative）。  
- **任务**：二分类情感分析，是情感分析的标准基准。

---

### **二、Target 数据集**
#### 1. **arc_challenge.jsonl**  
- **全称**：AI2 Reasoning Challenge (ARC) - Challenge  
- **领域**：科学问答（高难度）  
- **内容**：ARC 的困难版本，问题需复杂推理和跨学科知识。  
- **任务**：开放域问答，用于测试模型的深度推理能力。

#### 2. **financial_phrasebank.jsonl**  
- **全称**：Financial Phrasebank  
- **领域**：金融情感分析  
- **内容**：金融新闻句子，标注为 `positive`, `negative`, 或 `neutral`。  
- **任务**：三分类情感分析，评估模型在金融领域的应用效果。

#### 3. **medmcqa.jsonl**  
- **全称**：MedMCQA  
- **领域**：医学问答  
- **内容**：印度医学考试中的多选题，涵盖解剖学、药理学等。  
- **任务**：多选问答，测试模型在专业医学领域的知识。

#### 4. **sciq.jsonl**  
- **全称**：SciQ  
- **领域**：科学问答  
- **内容**：基础科学问题（物理、化学等），含正确答案和干扰项。  
- **任务**：多选问答，用于评估科学知识掌握程度。

#### 5. **social_i_qa.jsonl**  
- **全称**：SocialIQA  
- **领域**：社交常识推理  
- **内容**：关于社交情境的问题，需推断人物动机或行为后果。  
- **任务**：多选问答，测试模型对社交语境的理解能力。
## 多语言数据集

 **Amazon Multilingual Reviews** 是一个多语言的亚马逊商品评论数据集。
包含6种语言：英语(en)、德语(de)、西班牙语(es)、法语(fr)、日语(ja)、中文(zh)。评分范围是0-4，2是neutral。