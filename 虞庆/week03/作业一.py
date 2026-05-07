"""
train_rnn_multi_class.py
RNN 多分类任务：五字文本中「你」字的位置分类（5分类）
任务：输入5个汉字，判断「你」在第0/1/2/3/4位 → 对应5个类别
模型：Embedding -> RNN -> 最后时刻隐藏状态 -> Linear -> 多分类
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss
无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 4000
MAXLEN = 5          # 固定5字文本
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用汉字（用来随机拼接句子）
COMMON_CHARS = [
    '我','他','她','它','们','今','天','很','开','心','难','过','生','气','高','兴',
    '吃','饭','睡','觉','学','习','工','作','跑','跳','坐','躺','趴','蹲','着','路',
    '走','看','听','说','写','读','玩','笑','哭','美','丑','真','的','眼','手','嘴'
]

def generate_5_sentence():
    """
    生成：固定5字句子 + 标签（你在第几位）
    规则：
    - 句子里必须有且只有一个「你」
    - 位置随机：0/1/2/3/4 → 对应标签 0/1/2/3/4
    """
    # 随机选「你」的位置（0~4）
    pos = random.randint(0, 4)
    # 生成5个字
    sent = []
    for i in range(5):
        if i == pos:
            sent.append('你')
        else:
            sent.append(random.choice(COMMON_CHARS))
    return ''.join(sent), pos

# 构建数据集
def build_dataset():
    data = []
    for _ in range(N_SAMPLES):
        sent, label = generate_5_sentence()
        data.append((sent, label))
    random.shuffle(data)
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>':0, '<UNK>':1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode_sentence(sentence, vocab, maxlen=MAXLEN):
    """把句子转成数字id"""
    ids = [vocab.get(c, 1) for c in sentence]
    # 固定长度5
    ids = ids[:maxlen] + [0]*(maxlen-len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextMultiClassDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent, label = self.data[idx]
        x = encode_sentence(sent, self.vocab)
        y = torch.tensor(label, dtype=torch.long)  # 多分类必须用long
        return x, y

# ─── 4. 模型定义，使用RNN ────────────────────────────────────────────
class RNNMultiClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)  # 输出5类

    def forward(self, x):
        """
        x: (batch, seq_len=5)
        输出: (batch, num_classes=5)
        """
        # embedding: (B,5,embed)
        x_emb = self.embedding(x)
        # RNN: out=(B,5,hidden), hn=(1,B,hidden)
        out_rnn, _ = self.rnn(x_emb)
        # 取最后一个时刻的隐藏状态
        last_hidden = out_rnn[:, -1, :]  # (B, hidden)
        # 分类
        logits = self.fc(last_hidden)    # (B, 5)
        return logits

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def train():
    # 数据
    data = build_dataset()
    vocab = build_vocab(data)
    vocab_size = len(vocab)

    # 划分训练/验证
    split = int(len(data)*TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    # 加载器
    train_loader = DataLoader(TextMultiClassDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextMultiClassDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # 模型、损失、优化器
    model = RNNMultiClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()  # 多分类用交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  

    print(f"词表大小: {vocab_size}")
    print(f"训练样本: {len(train_data)} 验证样本: {len(val_data)}\n")

    # 开始训练
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}")

    print("\n--- 推理示例 ---")
    test_sents = [
        "你真的开心",     # 0
        "爱你一辈子",     # 1
        "今天你好看",     # 2
        "认真的你呀",     # 3
        "我很喜欢你"      # 4
    ]

    model.eval()
    with torch.no_grad():
        for sent in test_sents:
            x = encode_sentence(sent, vocab).unsqueeze(0)  # (1,5)
            logits = model(x)
            pred_pos = torch.argmax(logits, dim=1).item()
            print(f"句子：{sent} -> 预测「你」在第 {pred_pos} 位")

if __name__ == "__main__":
    train()
