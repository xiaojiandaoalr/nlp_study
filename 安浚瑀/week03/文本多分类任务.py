"""
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 参数
SEED = 12
EMBED_DIM = 64
HIDDEN_DIM = 64
N_SAMPLES = 5000
CLASS_NUM = 5
MAX_LEN = 5
LR = 1e-3
BATCH_SIZE = 64 
EPOCHS = 5
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# 数据集
char_pool = ["我", "他", "她", "它", "你", "那", "又", "是", "不", "了",
             "在", "人", "和", "有", "大", "小", "多", "少", "上", "下",
             "来", "去", "好", "坏", "美", "丑", "长", "短", "高", "低",
             "天", "地", "山", "水", "日", "月", "火", "风", "雨", "雪",
             "金", "木", "土", "石", "田", "云", "鸟", "鱼", "马", "牛",
             "羊", "花", "草", "树", "叶", "红", "黄", "蓝", "绿", "白",
             "黑", "光", "明", "暗", "声", "音", "笑", "哭", "走", "跑",
             "跳", "吃", "喝", "睡", "醒", "开", "关", "进", "出", "起",
             "落", "生", "死", "爱", "恨", "喜", "怒", "哀", "乐", "善",
             "恶", "真", "假", "对", "错", "正", "反", "左", "右", "中"]
def build_dataset(n = N_SAMPLES):
    data = []
    # 随机生成n个样本，格式为：["五个汉字", 你字在第几位]
    for i in range(n):
        chars = random.sample(char_pool, CLASS_NUM)
        if "你" not in chars:
            chars[random.randint(0, CLASS_NUM-1)] = "你"
        random.shuffle(chars)
        s = "".join(chars)
        data.append([s, s.index("你")])
    return data


# 词典
def build_vocab(data):
    vocab = {}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    unique_chars = sorted(set(data))
    vocab |= {c:i for i, c in enumerate(unique_chars,2)}
    return vocab
def encode(sent, vocab, max_len=MAX_LEN):
    ids = []
    for c in sent:
        ids.append(vocab.get(c, 1))
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# 模型
class RNNModel(nn.Module):
    def __init__(self, vocab_len, embed_dim = EMBED_DIM, hidden_dim = HIDDEN_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = x.max(dim=1).values
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x 

# DataSet
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(i[0], vocab) for i in data]
        self.Y = [i[1] for i in data]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.Y[idx], dtype=torch.long)

# 评估
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            y_pred = model(x)
            pred = y_pred.argmax(dim=1)
            correct += (pred == y).sum().item() 
            total += len(y)
    return correct / total
# 训练
def train():
    print("生成数据...")
    data = build_dataset()
    vocab = build_vocab(char_pool)
    print(f"样本数：{len(data)} 词库大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    rnn_model = RNNModel(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn_model.parameters(), lr=LR)

    totoal_params = sum(p.numel() for p in rnn_model.parameters())
    print(f"模型参数总量：{totoal_params}")

    for epoch in range(1, EPOCHS+1):
        rnn_model.train()
        total_loss = 0.0
        for x, y in train_loader:
            y_pred = rnn_model(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(rnn_model, val_loader)
        print(f'第{epoch:2d}/{EPOCHS}轮训练结束，损失值为{avg_loss:.4f}，验证集准确率为{val_acc:.4f}')

    print(f"\n最终验证准确率：{evaluate(rnn_model, val_loader):.4f}")

    # 推理
    test_sents = [
        '你生死对错',   # 你-0
        '生你死对错',   # 你-1
        '生死你对错',   # 你-2
        '生死对你错',   # 你-3
        '生死对错你',   # 你-4
    ]
    print("\n--- 推理示例 ---")
    rnn_model.eval()
    with torch.no_grad():
        for sent in test_sents:
            x = encode(sent, vocab)
            input_tensor = torch.tensor([x], dtype=torch.long)
            y = rnn_model(input_tensor)
            pred_idx = y.argmax(dim=1).item()
            print(f"句子: {sent} | 预测'你'的位置: {pred_idx} (真实位置: {sent.index('你') if '你' in sent else -1})")


if __name__ == "__main__":
    train()
