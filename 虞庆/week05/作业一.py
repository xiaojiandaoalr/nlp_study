import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===================== 超参数 =====================
SEQ_LEN = 32
BATCH_SIZE = 32
EMBED_DIM = 256
HIDDEN_DIM = 512
NHEAD = 4
NUM_LAYERS = 1
DROPOUT = 0.1
DEVICE = torch.device("cpu")  #

# ===================== 1. 读取语料 =====================
with open("corpus.txt", "r", encoding="utf-8") as f:
    text = f.read().replace("\n", " ").strip()

vocab = sorted(list(set(text)))
VOCAB_SIZE = len(vocab)
char2idx = {c:i for i,c in enumerate(vocab)}
idx2char = {i:c for i,c in enumerate(vocab)}
print(f"词表大小：{VOCAB_SIZE}")

# ===================== 2. 数据集 =====================
class TextDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.ids = [char2idx[c] for c in text]
        self.seq_len = seq_len
    def __len__(self):
        return len(self.ids)-self.seq_len
    def __getitem__(self, idx):
        x = self.ids[idx:idx+self.seq_len]
        y = self.ids[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

dataset = TextDataset(text, char2idx, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===================== 3. 位置编码 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000)/embed_dim))
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

# ===================== 4. 单向Transformer（带因果mask） =====================
class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos = PositionalEncoding(EMBED_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NHEAD, dim_feedforward=HIDDEN_DIM,
            batch_first=True, dropout=DROPOUT
        )
        self.encoder = nn.TransformerEncoder(layer, NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=DEVICE), diagonal=1)
        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.emb(x) * math.sqrt(EMBED_DIM)
        x = self.pos(x)
        mask = self.causal_mask(seq_len)
        x = self.encoder(x, mask=mask)
        return self.fc(x)

# ===================== 5. 训练 =====================
model = TransformerLM().to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 30  # 训练轮数
print("开始训练...")

for epoch in range(EPOCHS):
    total_loss = 0
    for x,y in dataloader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1,VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        opt.step()
        total_loss += loss.item()
    avg_loss = total_loss/len(dataloader)
    print(f"Epoch {epoch+1:2d} | Loss: {avg_loss:.3f} | PPL: {math.exp(avg_loss):.2f}")

# ===================== 6. top-p 采样 =====================
def generate(start_str, max_len=300, temperature=0.5, top_p=0.95):
    model.eval()
    ids = [char2idx[c] for c in start_str if c in char2idx]
    with torch.no_grad():
        for _ in range(max_len):
            x = torch.tensor([ids[-SEQ_LEN:]], device=DEVICE)
            logits = model(x)[0,-1] / temperature

            # top-p 采样
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumulative_probs > top_p
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            logits[sorted_idx[remove_mask]] = -float('inf')

            prob = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(prob, 1).item()
            ids.append(next_id)
    return ''.join([idx2char[i] for i in ids])

# ===================== 生成 =====================
print("\n生成文本：\n", generate("运输", max_len=10))
