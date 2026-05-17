import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from embedding import read_text, gen_vocab, get_text_ids

MODEL_PATH = "torch_model.pth"

N_SAMPLES     = 4000 # 样本数量
MIN_LENGTH    = 5
MAX_LENGTH    = 5   # 一句话的最大长度
TARGET_CHAR   = "你" # 分类依据的字符
EMBEDDING_DIM = 64 # embedding 向量维度
HIDDEN_DIM    = 64 # 隐藏层向量维度
LR            = 1e-3 # 学习率
BATCH_SIZE    = 64 # 批量大小
EPOCHS        = 20 # 训练轮数
DROP_OUT      = 0.3
TRAIN_RATIO   = 0.8

vocab = gen_vocab(read_text())

def build_dataset(n=N_SAMPLES, minlength=MIN_LENGTH, maxlength=MAX_LENGTH):
    """
    构建一个数据集，包含 n 个样本，每个样本是一个长度为 MAX_LENGTH 的文本序列
    且该文本序列中的任意一个字符必然是 TARGET_CHAR
    """
    data = []
    vocab_keys = [k for k in vocab.keys() if k != TARGET_CHAR][4:]

    for _ in range(n):
        text_list = np.random.choice(vocab_keys, maxlength).tolist()
        target_idx = np.random.randint(0, maxlength)
        text_list.insert(target_idx, TARGET_CHAR)

        data.append(("".join(text_list), target_idx))
    return data

class TextDataset(Dataset):
    def __init__(self, text_list, text_ids_list):
        self.x = text_ids_list
        self.y = [target_idx for _, target_idx in text_list]
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.x[idx]),
            # torch.LongTensor(self.y[idx]),
            self.y[idx],
        )

# 定义模型
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout=DROP_OUT, num_classes=MAX_LENGTH):
        super(Net, self).__init__()
        self.loss = nn.CrossEntropyLoss() # loss 函数
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.max(dim=1)[0]
        x = self.ln(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

# 训练评估
def evaluate(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            outputs = model(x) # 形状：[batch_size, num_classes]
            pred = outputs.argmax(dim=1) # 形状：[batch_size]

            y_tensor = torch.as_tensor(y, dtype=torch.long,device=pred.device)

            correct += (pred == y_tensor).sum().item()
            total += len(y)

    accuracy = correct / total if total != 0 else 0
    print("正确预测: %d, 正确率: %.4f" % (correct, accuracy))
    return accuracy

def train():
    print("生成数据集...")
    split = int(N_SAMPLES * TRAIN_RATIO)
    text_data = build_dataset(N_SAMPLES, maxlength=MAX_LENGTH)
    text_ids = get_text_ids(text_data, vocab, maxlength=MAX_LENGTH)
    train_data = text_data[:split], text_ids[:split]
    valid_data = text_data[split:], text_ids[split:]
    # print(text_ids)
    tran_loader = DataLoader(TextDataset(train_data[0], train_data[1]), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(TextDataset(valid_data[0], valid_data[1]), batch_size=BATCH_SIZE)
    vocab_size = len(vocab)
    model = Net(vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout=DROP_OUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in tran_loader:
            pred = model(x)
            loss = model.loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(tran_loader)
        valid_acc = evaluate(model, valid_loader)
        print(f"Epoch {epoch+1:02d}: loss: {avg_loss:.4f}, acc: {valid_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), MODEL_PATH)


def test (model_path=MODEL_PATH):
    """
    加载训练好的模型并进行测试

    参数:
        model_path: 模型文件路径
    """
    # 1. 加载模型
    model = Net(len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout=DROP_OUT)

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 不存在!")
        return

    try:
        # 加载模型参数
        model.load_state_dict(torch.load(model_path))
        print(f"✓ 成功加载模型: {model_path}")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return

    # 2. 设置模型为评估模式
    model.eval()

    # 3. 生成测试数据集
    print("\n生成测试数据集...")
    test_samples = 1000  # 测试样本数量
    test_data = build_dataset(test_samples, maxlength=MAX_LENGTH)
    test_ids = get_text_ids(test_data, vocab, maxlength=MAX_LENGTH)
    test_loader = DataLoader(TextDataset(test_data, test_ids), batch_size=BATCH_SIZE)

    # 4. 评估模型性能
    print("\n=== 模型测试结果 ===")
    test_accuracy = evaluate(model, test_loader)
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 5. 详细测试：显示一些具体的预测例子
    print("\n=== 详细预测示例 ===")
    num_examples = 10  # 显示的例子数量

    with torch.no_grad():
        for i in range(min(num_examples, len(test_data))):
            # 获取测试样本
            text, true_label = test_data[i]
            text_tensor = torch.LongTensor(test_ids[i]).unsqueeze(0)  # 添加batch维度

            # 模型预测
            output = model(text_tensor)
            predicted_label = output.argmax(dim=1).item()

            # 准备可视化文本
            text_chars = list(text)
            # 标记目标字符的位置
            marked_text = ""
            for j, char in enumerate(text_chars):
                if j == true_label:
                    marked_text += f"[{char}]"  # 真实位置
                elif j == predicted_label:
                    marked_text += f"{{{char}}}"  # 预测位置
                else:
                    marked_text += char

            # 判断是否正确
            is_correct = "✓" if predicted_label == true_label else "✗"

            print(f"示例 {i + 1}:")
            print(f"  文本: {marked_text}")
            print(f"  真实位置: {true_label}, 预测位置: {predicted_label} {is_correct}")
            # print(f"  预测置信度: {torch.softmax(output, dim=1)[predicted_label]:.4f}")

            # 显示所有位置的置信度（可选）
            if predicted_label != true_label:
                probs = torch.softmax(output, dim=1)
                print(f"  各位置概率: {[f'{p:.3f}' for p in probs]}")
            print()

    # 模型参数统计
    print("\n=== 模型参数统计 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print(f"模型文件 '{MODEL_PATH}' 已存在，跳过训练过程。")
    else:
        train()

    test()
