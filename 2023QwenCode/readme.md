# 主页 
## 天池 - https://tianchi.aliyun.com/competition/entrance/532169

# Overview
    1. 初赛阶段主要聚焦在如何通过 SFT 提升基础模型的代码能力。需要选手基于最新开源的 Qwen 1.8 模型作为基础模型，在我们提供的训练框架上自行进行数据收集与微调，训练完成后将进行自动评估，返回最终结果进行排名；
# 方案
## 初赛
    主要是查找高质量的数据集，然后进行微调

### 数据集
    1. bigcode/starcoderdata： （太大了，跑不动，没有使用）
    2. Safurai/Code-Instruct-700k(过滤掉无用语言，最终大概150多k)
    3. HuggingFaceH4/CodeAlpaca_20K
    4. codefuse-ai/CodeExercise-Python-27k
    5. codefuse-ai/Evol-Instruct-66k

    最终使用2 + 3 大概总共 175K（跑完要2-3天，哎）

### finetune
#### gpu服务
    1. aliyun 8c30G，1* A10 22G
#### 主要配置参数
   * Num examples = 175225
   * Num Epochs = 10
   * batch size per device = 2
   * Gradient Accumulation steps = 4
   * qlora：
        - 4bit
        - "lora_rank": 64,
        - "lora_alpha": 32,
        - "lora_dropout": 0.05

### issue
    1. batch size 太大容易gpu oom，=4 就跑一会oom
    2. 数据集太多的话，cpu mem 容易oom

## 结果
    1. 没跑完，大概跑了1/10，得分 

# 总结
## 竞赛内容的一些总结
    1. 数据：
    2. 资源：
    3. 模型：
    4. 任务：
    5. 环境

## 不足
    1. 应该把evaluate的代码也跑通，这样可以更好的评估fine-tune后的模型的好坏，快速进行迭代

## 优点

## 收获

## 遗憾

## todo
    1. evaluate常用数据集跑通，方便以后自己微调评估效果