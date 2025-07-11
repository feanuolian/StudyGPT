from torch import device
from My import *
from torch.utils.data import DataLoader

def main():
    # 初始化数据集
    conv_dataset = ConvDataset("data/xiaohuangji50w_nofenci.conv")
    conv_dataset.process_file()
    tokenizer = conv_dataset.build_vocab(max_vocab_size=10000, vocab_file="vocab.txt")

    # 创建PyTorch Dataset
    dialogue_dataset = DialogueDataset(conv_dataset, tokenizer, max_length=256)

    # 创建DataLoader
    train_loader = DataLoader(
        dialogue_dataset,
        batch_size=32,  # 根据GPU内存调整
        shuffle=True,
        collate_fn=batch_padding_collate,  # 使用动态填充函数
        num_workers=4,  # 多进程加载
        pin_memory=True  # 加速GPU传输
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
    "lr": 5e-3,  # 初始学习率
    "weight_decay": 0.01,  # 权重衰减
    "epochs": 2,  # 训练轮次
    "grad_accum_steps": 4,  # 梯度累积步数
    "use_amp": True,  # 启用混合精度训练
    "max_grad_norm": 1.0,  # 梯度裁剪阈值
    "save_every": 2,  # 每2轮保存一次
    "assistant_token_id": tokenizer.vocab["<|assistant|>"]  # 助理标记ID
    }
    # 初始化模型
    vocab_size = tokenizer.vocab_size
    model = DialogueModel(vocab_size, d_model=512, n_head=8, n_layers=4)

    # 创建训练器
    trainer = TransformerTrainer(model, train_loader, device, config)

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()