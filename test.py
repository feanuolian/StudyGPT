
from My import *
# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载检查点
model, optimizer, scheduler, start_epoch, config = load_checkpoint(
    "data/model_epoch_2.pt", device 
)

# 初始化分词器（需要与训练时相同）
tokenizer = Tokenizer().load_vocab("vocab.txt")  # 假设您的词表文件路径

# 设置模型为评估模式
model.eval()

def generate_response(model, tokenizer, input_text, device, max_length=50, temperature=0.8):
    """
    使用模型生成对话响应
    :param model: 训练好的对话模型
    :param tokenizer: 分词器实例
    :param input_text: 用户输入文本
    :param device: 计算设备
    :param max_length: 生成的最大长度
    :param temperature: 控制生成随机性的温度参数
    :return: 生成的响应文本
    """
    # 构建输入序列
    input_seq = f"<|user|>{input_text}<|end|><|assistant|>"  # 遵循训练格式
    input_ids = tokenizer.encode(input_seq)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # 创建角色ID (0=用户, 1=助理)
    role_ids = torch.zeros_like(input_tensor)
    # 助理标记之后都为助理角色
    role_ids[:, len(tokenizer.encode("<|user|>")) + len(tokenizer.encode(input_text)) + len(tokenizer.encode("<|end|>")):] = 1
    
    model.eval()  # 设置为评估模式
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            outputs = model(input_tensor, role_ids)
            
            # 获取最后一个token的logits
            next_token_logits = outputs[0, -1, :] / temperature
            
            # 应用softmax获取概率
            probs = F.softmax(next_token_logits, dim=-1)
            
            # 从概率分布中采样下一个token
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # 如果遇到结束标记则停止
            if next_token_id == tokenizer.vocab["<|end|>"]:
                break
                
            # 添加生成的token到序列
            generated_ids.append(next_token_id)
            
            # 更新输入序列
            new_input = torch.tensor([[next_token_id]], device=device)
            input_tensor = torch.cat((input_tensor, new_input), dim=1)
            
            # 更新角色ID（后续所有token都是助理生成）
            new_role = torch.ones_like(new_input)
            role_ids = torch.cat((role_ids, new_role), dim=1)
            
            # 动态截断过长的序列
            if input_tensor.size(1) > model.pos_embed.encoding.size(0):
                input_tensor = input_tensor[:, -model.pos_embed.encoding.size(0):]
                role_ids = role_ids[:, -model.pos_embed.encoding.size(0):]
    
    # 解码生成的序列
    response_text = tokenizer.decode(generated_ids)
    
    # 清理特殊字符和空格
    response_text = response_text.replace("<|end|>", "").replace("<|user|>", "").replace("<|assistant|>", "")
    response_text = response_text.replace("  ", " ").strip()
    
    return response_text

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
        max_length=100,  # 最大生成长度
        temperature=0.7  # 控制创造性：0.0-1.0，值越大越随机
    )
    
    print(f"AI: {response}")