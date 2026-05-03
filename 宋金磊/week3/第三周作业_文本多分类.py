"""
train_chinese_multicls_rnn_lstm.py
中文句子多分类（3类：正面/中性/负面）
支持模型：RNN / LSTM
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4500
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 多分类数据生成（3类：0=负面 1=中性 2=正面）────────────
POS_KEYS = ['好', '棒', '赞', '喜欢', '满意']
NEG_KEYS = ['差', '坏', '烂', '坑', '不满']

# 正面模板
TEMPLATES_POS = [
    '这家{}真的很{}，下次还来',
    '这款{}设计让我{}',
    '{}的服务态度让我感到{}',
    '{}体验非常{}',
    '这次购物感觉{}极了',
]

# 中性模板
TEMPLATES_NEU = [
    '今天天气正常，出门散步',
    '这部电影内容一般',
    '下午开了会议',
    '路上正常通勤',
    '这道题正在思考',
    '最近工作正常进行',
    '超市人不多',
    '换季注意身体',
    '今天作业正常',
    '公交车准点',
    '吃饭睡觉上班',
    '日常学习记录',
]

# 负面模板
TEMPLATES_NEG = [
    '这家{}真的很{}，再也不来',
    '这款{}设计让我很{}',
    '{}的服务态度让人感到{}',
    '{}体验非常{}',
    '这次购物感觉{}极了',
]

OBJ_WORDS = ['店铺', '餐厅', '产品', '服务', '环境', '系统', '设计', '课程']
ADJ_WORDS = ['方便', '简洁', '独特', '舒适', '高效']

# 正面
def make_positive():
    kw   = random.choice(POS_KEYS)
    tmpl = random.choice(TEMPLATES_POS)
    obj  = random.choice(OBJ_WORDS)
    try:
        sent = tmpl.format(obj, kw)
    except:
        sent = obj + kw
    return sent, 2

# 中性
def make_neutral():
    sent = random.choice(TEMPLATES_NEU)
    return sent, 1

# 负面
def make_negative():
    kw   = random.choice(NEG_KEYS)
    tmpl = random.choice(TEMPLATES_NEG)
    obj  = random.choice(OBJ_WORDS)
    try:
        sent = tmpl.format(obj, kw)
    except:
        sent = obj + kw
    return sent, 0

# 构建3分类数据集
def build_dataset(n=N_SAMPLES):
    data = []
    per_class = n // 3
    for _ in range(per_class):
        data.append(make_positive())
        data.append(make_neutral())
        data.append(make_negative())
    random.shuffle(data)
    return data

# ─── 2. 词表与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# ─── 3. Dataset ────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

# ─── 4. 多分类模型（支持 RNN / LSTM）────────────────────
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes=3, model_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        # RNN 或 LSTM
        if model_type == "RNN":
            self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)

        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        e = self.embedding(x)  # (B, L, E)
        output, _ = self.rnn(e)  # (B, L, H)
        feat = output.max(dim=1)[0]  # 池化
        feat = self.dropout(self.bn(feat))
        logits = self.fc(feat)  # (B, 3)
        return logits

# ─── 5. 评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

# ─── 6. 训练 ──────────────────────────────────────────
def train(model_type="RNN"):
    print("生成 3 分类数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = TextClassifier(vocab_size=len(vocab), model_type=model_type)
    criterion = nn.CrossEntropyLoss()  # 多分类
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"模型：{model_type} | 3分类\n")

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d} | loss={avg_loss:.4f} | val_acc={acc:.4f}")

    # 测试
    print("\n--- 测试示例 ---")
    model.eval()
    label_map = {0:"负面", 1:"中性", 2:"正面"}
    test_sents = [
        '这款产品真的很棒，非常满意',
        '今天天气正常，出门散步',
        '服务太差了，再也不来',
        '公交车准点到达',
        '这个设计烂透了',
    ]
    with torch.no_grad():
        for s in test_sents:
            ids = torch.tensor([encode(s, vocab)])
            logits = model(ids)
            pred_idx = torch.argmax(logits).item()
            print(f"[{label_map[pred_idx]}] {s}")

if __name__ == '__main__':
    # 训练RNN模型
    #train(model_type="RNN")
    # 训练LSTM模型
    train(model_type="LSTM")
