import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pandas as pd

# ====================== 全局超参数 ======================
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

VOCAB = ["我", "你", "他", "她", "它", "好", "坏", "爱", "吃", "睡"]
VOCAB_SIZE = len(VOCAB)
NUM_CLASS = 5
SEQ_LEN = 5
EMBED_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

word2idx = {w:i for i,w in enumerate(VOCAB)}

# ====================== 数据集生成 ======================
def generate_sample():
    pos = random.randint(0,4)
    sent = ["你" if i==pos else random.choice([w for w in VOCAB if w!="你"]) for i in range(5)]
    label = pos
    return [word2idx[w] for w in sent], label

def build_dataset(n):
    data = [generate_sample() for _ in range(n)]
    x = torch.tensor([d[0] for d in data], dtype=torch.long)
    y = torch.tensor([d[1] for d in data], dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x,y)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

train_loader = build_dataset(2000)
test_loader  = build_dataset(500)

# ====================== 模型1：RNN ======================
class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASS)
    def forward(self,x):
        x = self.emb(x)
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

# ====================== 模型2：LSTM ======================
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASS)
    def forward(self,x):
        x = self.emb(x)
        _, (h,_) = self.lstm(x)
        return self.fc(h.squeeze(0))

# ====================== 模型3：GRU ======================
class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, NUM_CLASS)
    def forward(self,x):
        x = self.emb(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# ====================== 模型4：单层Transformer ======================
class TransformerSingle(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(SEQ_LEN, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=2, dim_feedforward=HIDDEN_DIM,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 1)
        self.fc = nn.Linear(EMBED_DIM, NUM_CLASS)
    def forward(self,x):
        pos = torch.arange(SEQ_LEN, device=DEVICE).unsqueeze(0)
        x = self.emb(x) + self.pos_emb(pos)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))

# ====================== 模型5：多层Transformer ======================
class TransformerMulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(SEQ_LEN, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=2, dim_feedforward=HIDDEN_DIM,
            batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, 3)
        self.fc = nn.Linear(EMBED_DIM, NUM_CLASS)
    def forward(self,x):
        pos = torch.arange(SEQ_LEN, device=DEVICE).unsqueeze(0)
        x = self.emb(x) + self.pos_emb(pos)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))

# ====================== 训练 & 测试 ======================
def run(model):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        model.train()
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()

    model.eval()
    correct, total = 0,0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += len(x)
    return round(correct/total, 4)

# ====================== 一键对比 ======================
if __name__ == "__main__":
    print("正在训练对比...")
    result = {
        "RNN": run(RNN()),
        "LSTM": run(LSTM()),
        "GRU": run(GRU()),
        "Transformer(单层)": run(TransformerSingle()),
        "Transformer(多层)": run(TransformerMulti()),
    }
    df = pd.DataFrame(list(result.items()), columns=["模型", "测试准确率"])
    print("\n===== 文本分类效果对比 =====")
    print(df)