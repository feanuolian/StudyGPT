import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import re
from collections import Counter
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from tqdm import tqdm
class DialogueModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, n_layers=3):
        super().__init__()
        # 角色嵌入（user/assistant）
        self.role_embed = nn.Embedding(2, d_model)  
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model,max_len=256, device="cuda")  # 位置编码

        # 精简Transformer层
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, n_head) for _ in range(n_layers)
        ])
        
        # 输出层
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, role_ids):
        # 嵌入融合：token嵌入 + 角色嵌入
        tok_emb = self.token_embed(input_ids)
        role_emb = self.role_embed(role_ids)
        x = tok_emb + role_emb  # 
        x = self.pos_embed(x)
        
        # 通过Transformer层
        for layer in self.encoder_layers:
            x = layer(x)
            
        return self.lm_head(x)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self,x):
        # 自注意力
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, device=None):
        """
        compute sinusoid encoding.
        
        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # 存储编码以备后用
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(1)
        _2i = torch.arange(0, d_model, step=2).float()
        
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.encoding = self.encoding.to(device)  # 移动到设备

    def forward(self, x):
        """
        x: 输入张量，形状为 (batch_size, seq_len, d_model)
        """
        # 获取序列长度和批次大小
        batch_size, seq_len, _ = x.shape
        
        # 从预计算的位置编码中取出前seq_len个
        pos_encoding = self.encoding[:seq_len, :].clone().detach()
        pos_encoding = pos_encoding.unsqueeze(0)  # 增加批次维度 -> (1, seq_len, d_model)
        pos_encoding = pos_encoding.to(x.device)  # 确保在相同设备上
        
        return x + pos_encoding  # 广播到整个批次

class Tokenizer:
    def __init__(self, vocab=None, special_tokens=None):
        # 定义特殊标记
        self.special_tokens = special_tokens or [
            "<|user|>", "<|assistant|>", "<|end|>", 
            "<pad>", "<unk>", "<bos>", "<eos>"
        ]
        
        self.vocab = vocab or {}
        self.inverse_vocab = {}
        if self.vocab:
            self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
    def tokenize(self, text):
        """分词函数，处理特殊标记和普通文本"""
        # 先处理特殊标记
        tokens = []
        while text:
            found = False
            for token in self.special_tokens:
                if text.startswith(token):
                    tokens.append(token)
                    text = text[len(token):]
                    found = True
                    break
            
            if not found:
                # 普通字符级分词（中文）
                tokens.append(text[0])
                text = text[1:]
        
        return tokens
    
    def build_vocab(self, texts, max_vocab_size=10000):
        """从文本构建词表"""
        # 添加特殊标记
        vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        # 统计普通字符频率
        char_counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                # 跳过空token和特殊标记
                if token and token not in self.special_tokens:
                    char_counter[token] += 1
        
        # 添加最常见的字符（过滤空字符）
        valid_chars = [(char, count) for char, count in char_counter.most_common() 
                      if char and char.strip()]  # 过滤空字符和空白字符
        
        # 确保不超过最大词表大小
        for idx, (char, count) in enumerate(valid_chars[:max_vocab_size - len(self.special_tokens)], 
                                          start=len(self.special_tokens)):
            vocab[char] = idx
        
        self.vocab = vocab
        self.inverse_vocab = {idx: token for token, idx in vocab.items()}
        return self
    
    def encode(self, text):
        """文本转token ID"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
    
    def decode(self, token_ids):
        """token ID转文本"""
        return ''.join(self.inverse_vocab.get(idx, "<unk>") for idx in token_ids)
    
    def save_vocab(self, file_path):
        """保存词表到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                # 跳过空token
                if not token:
                    continue
                    
                # 清理token中的特殊字符
                clean_token = token.replace('\t', '\\t').replace('\n', '\\n').replace('\r', '\\r')
                f.write(f"{clean_token}\t{idx}\n")
    
    @classmethod
    def load_vocab(cls, file_path):
        """从文件加载词表（修复格式问题）"""
        vocab = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                parts = line.split('\t', 1)  # 最多分割成两部分
                
                if len(parts) < 2:
                    print(f"警告: 第 {line_num} 行格式错误 - '{line}'，跳过")
                    continue
                
                token, idx_str = parts
                
                try:
                    # 还原特殊字符
                    token = token.replace('\\t', '\t').replace('\\n', '\n').replace('\\r', '\r')
                    idx = int(idx_str)
                    vocab[token] = idx
                except ValueError:
                    print(f"警告: 第 {line_num} 行索引无效 - '{idx_str}'，跳过")
        
        return cls(vocab)
    
    @property
    def vocab_size(self):
        return len(self.vocab)


class ConvDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dialogues = []
        self.tokenizer = Tokenizer()
        
    def process_file(self):
        """处理对话文件（增强错误处理）"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        dialogues = []
        current_dialogue = []
        
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if line == 'E':
                        if current_dialogue:
                            dialogues.append(current_dialogue)
                            current_dialogue = []
                    elif line.startswith('M '):
                        # 移除开头的'M '并处理转义字符
                        content = line[2:]
                        
                        # 清理特殊字符
                        content = re.sub(r'\\', '', content)  # 移除转义字符
                        content = content.replace('\t', ' ')  # 替换制表符
                        content = content.replace('\r', ' ')  # 替换回车符
                        content = content.replace('\n', ' ')  # 替换换行符
                        
                        # 确保内容非空
                        if content.strip():
                            current_dialogue.append(content)
                except Exception as e:
                    print(f"处理第 {line_num} 行时出错: {e}")
                    continue
        
        # 处理最后一个对话
        if current_dialogue:
            dialogues.append(current_dialogue)
        
        self.dialogues = dialogues
        print(f"成功加载 {len(dialogues)} 个对话")
        return self
    
    def format_dialogues(self):
        """将对话格式化为训练文本（增强鲁棒性）"""
        formatted_texts = []
        skipped_dialogues = 0
        
        for dialogue in self.dialogues:
            # 确保对话有偶数个轮次（用户-助手交替）
            if len(dialogue) % 2 != 0:
                dialogue = dialogue[:-1]  # 移除最后一个不完整的轮次
                
            # 跳过无效对话
            if len(dialogue) == 0:
                skipped_dialogues += 1
                continue
                
            formatted_dialogue = []
            for i, text in enumerate(dialogue):
                # 清理文本
                clean_text = text.strip()
                if not clean_text:
                    continue
                    
                # 偶数索引为用户，奇数为助手
                role = "<|user|>" if i % 2 == 0 else "<|assistant|>"
                formatted_dialogue.append(f"{role}{clean_text}<|end|>")
            
            # 确保有内容
            if formatted_dialogue:
                formatted_texts.append(''.join(formatted_dialogue))
        
        if skipped_dialogues:
            print(f"跳过 {skipped_dialogues} 个无效对话")
            
        return formatted_texts
    
    def build_vocab(self, max_vocab_size=10000, vocab_file=None):
        """构建词表并保存（增加错误处理）"""
        formatted_texts = self.format_dialogues()
        
        if vocab_file and os.path.exists(vocab_file):
            print(f"尝试从 {vocab_file} 加载现有词表...")
            try:
                self.tokenizer = Tokenizer.load_vocab(vocab_file)
                print(f"加载成功，词表大小: {self.tokenizer.vocab_size}")
                return self.tokenizer
            except Exception as e:
                print(f"加载现有词表失败: {e}，重建词表")
        
        print(f"构建新词表，使用 {len(formatted_texts)} 个格式化对话...")
        self.tokenizer.build_vocab(formatted_texts, max_vocab_size)
        
        if vocab_file:
            try:
                self.tokenizer.save_vocab(vocab_file)
                print(f"词表保存到 {vocab_file}，词表大小: {self.tokenizer.vocab_size}")
            except Exception as e:
                print(f"保存词表失败: {e}")
        
        return self.tokenizer
    
    def __len__(self):
        return len(self.dialogues)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size if self.tokenizer else 0
    
    def get_dialogues(self):
        return self.dialogues
from torch.utils.data import default_collate
class DialogueDataset(Dataset):
    def __init__(self, conv_dataset, tokenizer, max_length=512):
        """
        Args:
            conv_dataset: 预处理好的ConvDataset实例
            tokenizer: 自定义的Tokenizer实例
            max_length: 最大序列长度 (默认512)
        """
        self.tokenizer = tokenizer
        self.formatted_texts = conv_dataset.format_dialogues()
        self.max_length = max_length
        
        # 预先计算特殊token ID
        self.user_token_id = tokenizer.vocab["<|user|>"]
        self.assistant_token_id = tokenizer.vocab["<|assistant|>"]
        self.end_token_id = tokenizer.vocab["<|end|>"]
        self.pad_token_id = tokenizer.vocab["<pad>"]
    
    def __len__(self):
        return len(self.formatted_texts)
    
    def __getitem__(self, idx):
        """处理单个对话样本"""
        text = self.formatted_texts[idx]
        input_ids = self.tokenizer.encode(text)
        
        # 智能截断 (保留完整对话轮次)
        if len(input_ids) > self.max_length:
            # 查找最后一个完整的轮次结束位置
            last_end_index = -1
            for i in range(len(input_ids)-1, -1, -1):
                if input_ids[i] == self.end_token_id and i < self.max_length:
                    last_end_index = i
                    break
            input_ids = input_ids[:last_end_index+1] if last_end_index > 0 else input_ids[:self.max_length]
        
        # 创建标签 (仅助理回复部分需要学习)
        labels = self._create_labels(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "pad_token_id": self.pad_token_id
        }
    
    def _create_labels(self, input_ids):
        """
        创建标签序列 (基于多轮对话的高效标注方法)
        - 用户部分：-100 (忽略loss)
        - 助理回复部分：实际token (模型学习目标)
        """
        labels = [-100] * len(input_ids)  # 初始化为忽略标签
        in_assistant = False  # 当前是否在助理回复中
        
        for i, token_id in enumerate(input_ids):
            if token_id == self.assistant_token_id:
                in_assistant = True
                continue
                
            if token_id == self.user_token_id:
                in_assistant = False
                continue
                
            if in_assistant and token_id != self.end_token_id:
                labels[i] = token_id
                
            # 结束标记需要学习生成
            if token_id == self.end_token_id:
                labels[i] = token_id
                in_assistant = False
        
        return labels

    def dynamic_padding_collate(self,batch_list):
        """处理batch数据并动态填充"""
        # 分离不同元素（如果数据集返回元组）
        input_ids = [item["input_ids"] for item in batch_list]
        labels = [item["labels"] for item in batch_list]
        attention_masks = [item.get("attention_mask", None) for item in batch_list]
    
        # 动态填充逻辑
        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []
    
        for i in range(len(input_ids)):
            pad_len = max_len - len(input_ids[i])
            padded_input_ids.append(
                torch.cat([input_ids[i], torch.full((pad_len,), self.pad_token_id)])
            )
            padded_labels.append(
                torch.cat([labels[i], torch.full((pad_len,), -100)])  # -100用于忽略loss
            )
            if attention_masks[i] is not None:
                padded_attention_masks.append(
                    torch.cat([attention_masks[i], torch.zeros(pad_len)])
                )
    
        # 返回标准字典格式
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_attention_masks) if attention_masks[0] is not None else None
        }

def batch_padding_collate(batch_list):
    """独立于Dataset类的填充函数"""
    # 检查输入类型（调试用）
    if not batch_list or not isinstance(batch_list, list):
        raise ValueError(f"Invalid batch type: {type(batch_list)}")
    
    # 获取批次中所有元素的键
    keys = batch_list[0].keys()
    
    # 初始化填充后的批次字典
    padded_batch = {}
    
    # 获取填充ID（假设所有批次元素结构相同）
    pad_token_id = batch_list[0].get("pad_token_id", 0)
    
    # 对每个键进行填充
    for key in keys:
        # 跳过特殊键
        if key == "pad_token_id":
            continue
            
        # 获取当前键的所有值
        values = [item[key] for item in batch_list]
        
        # 如果是张量列表，进行填充
        if isinstance(values[0], torch.Tensor):
            # 找到最大长度
            max_len = max(v.size(0) for v in values)
            
            # 填充张量
            padded_values = []
            for v in values:
                padding_length = max_len - v.size(0)
                if key == "labels":
                    # 对于标签，使用-100填充（忽略loss）
                    padding = torch.full((padding_length,), -100, dtype=v.dtype)
                else:
                    # 对于输入和attention_mask，使用pad_token_id
                    padding = torch.full((padding_length,), pad_token_id, dtype=v.dtype)
                
                padded_values.append(torch.cat([v, padding], dim=0))
            
            padded_batch[key] = torch.stack(padded_values)
        else:
            # 非张量数据直接堆叠
            padded_batch[key] = torch.tensor(values)
    
    return padded_batch
class TransformerTrainer:
    def __init__(self, model, train_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.config = config
        
        # 优化器选择 (AdamW适合Transformer)
        self.optimizer = AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        # 混合精度训练
        self.scaler = GradScaler(enabled=config["use_amp"])
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config["lr"],
            epochs=config["epochs"],
            steps_per_epoch=len(train_loader)
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, batch):
        """计算损失并处理角色ID"""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # 创建角色ID (0:用户, 1:助手)
        role_ids = torch.zeros_like(input_ids)
        for i, seq in enumerate(input_ids):
            for j, token_id in enumerate(seq):
                if token_id == self.config["assistant_token_id"]:
                    role_ids[i, j:] = 1  # 助理标记后的所有token设为1
        
        # 混合精度前向传播
        with autocast(device_type=f'{self.device}',enabled=self.config["use_amp"]):
            outputs = self.model(input_ids, role_ids)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return loss

    def train_epoch(self):
        """单epoch训练循环"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # 梯度累积
            with torch.set_grad_enabled(True):
                loss = self.compute_loss(batch) / self.config["grad_accum_steps"]
                
                # 混合精度反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪防止爆炸
                if (step + 1) % self.config["grad_accum_steps"] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["max_grad_norm"]
                    )
                    
                    # 参数更新
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item()*self.config['grad_accum_steps']:.4f}"})
        
        return total_loss / len(self.train_loader)

    def train(self):
        """完整训练流程"""
        for epoch in range(self.config["epochs"]):
            avg_loss = self.train_epoch()
            print(f"Epoch {epoch+1}/{self.config['epochs']} | Loss: {avg_loss:.4f}")
            
            # 保存中间模型
            if (epoch + 1) % self.config["save_every"] == 0:
                self.save_checkpoint(epoch)
        
        print("Training completed!")

    def save_checkpoint(self, epoch):
        """保存模型检查点"""
        checkpoint = {
            "epoch": epoch,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "config": self.config
        }
        torch.save(checkpoint, f"model_epoch_{epoch+1}.pt")
def load_checkpoint(checkpoint_path, device):
    """加载检查点并恢复训练状态"""
    # 加载检查点文件到指定设备
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    
    # 恢复模型
    model = checkpoint["model"]  # 直接加载模型对象
    model.to(device)
    
    # 恢复优化器和调度器
    optimizer = checkpoint["optimizer"]
    scheduler = checkpoint["scheduler"]
    
    # 恢复训练状态
    epoch = checkpoint["epoch"]
    config = checkpoint["config"]
    
    # 设置模型模式
    model.train()  # 或 model.eval() 用于推理
    
    return model, optimizer, scheduler, epoch, config