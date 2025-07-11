# StudyGPT
用来让初学者学习llm的项目。
# TinyDialogue 对话模型项目

这是一个轻量级中文对话模型项目，基于Transformer架构实现，支持训练和交互式对话功能。

## 项目特点

🚀 **轻量高效**：精简Transformer架构，模型大小仅~100MB
💬 **中文优化**：针对中文对话场景定制分词器和训练数据
🎓 **简单易用**：提供完整训练和推理代码，无需复杂配置
🔧 **模块化设计**：各组件解耦，易于扩展和定制

## 模型信息

| 特性 | 描述 |
|------|------|
| 模型名称 | StudyTinyModel |
| 架构 | 自定义Transformer (4层, 8头注意力) |
| 嵌入维度 | 512 |
| 词表大小 | 10,000 |
| 最大序列长度 | 256 |
| 训练数据 | 小黄鸡50万对话语料 |
### 模型下载
[ModelScope模型主页](https://www.modelscope.cn/models/fenuolian/StudyTinyModel/summary)
**由于是在小黄鸡的数据集上训练，并未清晰数据，因此会出现对话短、话不雅、无前因后果的情况**
## 快速开始

### 环境准备

```bash
# 创建Python环境（3.12）
conda create -n tinydialogue python=3.12
conda activate tinydialogue

# 安装依赖
pip install torch tqdm
```

### 交互式对话

```python
from My import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 安全加载模型（PyTorch 2.6+兼容）
model, optimizer, scheduler, start_epoch, config = load_checkpoint(
"data/model_epoch_2.pt", device
)

# 加载词表
tokenizer = Tokenizer().load_vocab("vocab.txt")

# 进入对话模式
while True:
  user_input = input("\n您: ")
  if user_input.lower() in ["exit", "quit", "bye"]:
  break

  # 生成响应
  response = generate_response(
  model=model,
  tokenizer=tokenizer,
  input_text=user_input,
  device=device,
  max_length=100,
  temperature=0.7
  )

  print(f"AI: {response}")
```

### 启动对话
```bash
python test.py
```

## 训练自定义模型

### 数据准备
1. 将对话数据整理为`.conv`格式
2. 文件格式示例：
```
M 用户对话内容1
M 助手回复内容1
M 用户对话内容2
M 助手回复内容2
E
```

### 训练步骤
```python
# 初始化数据集
conv_dataset = ConvDataset("data/your_data.conv")
conv_dataset.process_file()

# 构建词表
tokenizer = conv_dataset.build_vocab(max_vocab_size=10000, vocab_file="vocab.txt")

# 创建训练数据集
dialogue_dataset = DialogueDataset(conv_dataset, tokenizer, max_length=256)

# 配置训练参数
config = {
"lr": 5e-3,
"weight_decay": 0.01,
"epochs": 10,
"grad_accum_steps": 4,
"use_amp": True,
"max_grad_norm": 1.0,
"save_every": 2,
"assistant_token_id": tokenizer.vocab["<|assistant|>"]
}

# 初始化模型
model = DialogueModel(tokenizer.vocab_size, d_model=512, n_head=8, n_layers=4)

# 创建训练器并开始训练
trainer = TransformerTrainer(model, train_loader, device, config)
trainer.train()
```

### 启动训练
```bash
python train.py
```

## 文件说明

| 文件 | 描述 |
|------|------|
| `My.py` | 核心模型实现（模型架构、分词器、数据集） |
| `train.py` | 模型训练脚本 |
| `dialogue.py` | 交互式对话脚本 |
| `data/` | 训练数据和模型检查点目录 |
| `vocab.txt` | 分词器词表文件 |

## 高级选项

### 生成参数调优
```python
response = generate_response(
model=model,
tokenizer=tokenizer,
input_text=user_input,
device=device,
max_length=150,# 增加生成长度
temperature=0.9,# 提高创造性
top_k=50,# 使用top-k采样
repetition_penalty=1.2 # 减少重复
)
```

### 性能优化
```python
# 训练配置优化
config = {
"use_amp": True,# 启用混合精度训练
"grad_accum_steps": 8,# 小显存设备增大梯度累积步数
"batch_size": 16# 根据GPU调整批次大小
}
```

## 常见问题

### 模型加载失败（PyTorch 2.6+）
```python
# 添加安全声明
from My import DialogueModel
import torch.serialization

torch.serialization.add_safe_globals([DialogueModel])
```

### 显存不足
1. 减小`batch_size`
2. 增加`grad_accum_steps`
3. 启用`use_amp`混合精度

## 贡献指南

欢迎提交Issue和Pull Request：
- 报告问题
- 添加新功能
- 优化文档
- 改进模型架构
