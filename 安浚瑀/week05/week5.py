"""
训练基于transformer的单向语言模型，并完成文本生成。
"""

import os
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = pathlib.Path(__file__).parent

def load_corpus(corpus_path=None):
    if corpus_path is None:
        corpus_path = SCRIPT_DIR / "corpus.txt"
    else:
        corpus_path = pathlib.Path(corpus_path)
    with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text

# 根据语料文件，生成词典，每个词对应一行，并且有序，保存到vocab.txt
def build_vocab(text, vocab_path):
    vocab = sorted(set(text))
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("<PAD>\n")
        f.write("<UNK>\n")
        f.write("<SOS>\n")
        f.write("<EOS>\n")
        for word in vocab:
            if word.strip():  # 只写入非空字符
                f.write(word + "\n")
    print(f"  生成词典：{len(vocab)}个词")

# 加载词表, 每个词对应一行
def load_vocab(vocab_path="vocab.txt"):
    word_to_id = {}
    id_to_word = {}
    unk_id = None
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line_idx, word in enumerate(f):
            word = word.strip()
            if not word:
                continue
            word_to_id[word] = line_idx
            id_to_word[line_idx] = word
            if word == '<UNK>':
                unk_id = line_idx
    if unk_id is None:
        unk_id = word_to_id.get('<PAD>', 0)
    return word_to_id, id_to_word, unk_id
        
# 自定义数据集类，继承数据集类Dataset
class CharDataset(Dataset):
    def __init__(self, text, word_to_id, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor([word_to_id.get(word, word_to_id.get('<UNK>', 0)) for word in text])

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_len]
        Y = self.data[idx+1:idx+self.seq_len+1]
        return X, Y


# 自定义模型，基于transformer架构
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(1024, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        max_pos = self.pos_embedding.num_embeddings
        if T > max_pos:
            new_pos_emb = nn.Embedding(T + 10, self.pos_embedding.embedding_dim).to(x.device)
            with torch.no_grad():
                new_pos_emb.weight[:max_pos] = self.pos_embedding.weight
                new_pos_emb.weight[max_pos:] = self.pos_embedding.weight[-1:]
            self.pos_embedding = new_pos_emb
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.transformer(x)
        x = self.linear(x)
        return x

# 模型隐藏层的维度，也是词嵌入的维度
HIDDEN_SIZE = 512
# Transformer模型层数
NUM_LAYERS = 6
# 多头注意力机制中注意力头的数量
NUM_HEADS = 8
# Dropout概率
DROPOUT = 0.1
# 轮次
EPOCHS = 100
# 批次
BATCH_SIZE = 64
# 序列长度
SEQ_LEN = 10
# 学习率
LR = 0.001


# 模型训练
def train(save_model_name):

    # 加载语料
    text = load_corpus()

    # 加载词表
    word_to_id, id_to_word, unk_id = load_vocab(str(VOCAB_PATH))

    split_idx = int(0.8 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = CharDataset(train_text, word_to_id, seq_len=SEQ_LEN)
    val_dataset = CharDataset(val_text, word_to_id, seq_len=SEQ_LEN)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 初始化模型
    model = TransformerModel(len(word_to_id), HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 按轮次训练
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X, Y in train_loader:
            logits = model(X)
            loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                logits = model(X)
                loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        ppl = torch.exp(torch.tensor(val_loss)).item()
        print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={train_loss:.4f}  ppl={ppl:.4f}")

    # 保存模型
    torch.save(model.state_dict(), save_model_name)

# 生成文本
def generate_text(model, word_to_id, id_to_word, prompt, max_len, temperature=1.0, top_p=0.9, device="cpu"):
    """
    使用训练好的模型生成文本
    参数:
        model: 训练好的模型
        word_to_id: 词到索引的映射
        id_to_word: 索引到词的映射
        prompt: 初始提示文本
        max_len: 最大生成长度
        temperature: 温度参数，控制随机性
        top_p: nucleus采样概率阈值
        device: 设备
    返回:
        生成的文本
    """
    model.eval()
    import torch.nn.functional as F
    
    # 将prompt转换为tensor
    if isinstance(prompt, str):
        unk_id = word_to_id.get('<UNK>', 0)
        prompt_ids = [word_to_id.get(c, unk_id) for c in prompt]
        prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    elif prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    
    generated_text = ""
    
    with torch.no_grad():
        for i in range(max_len):
            # 限制prompt长度（滑动窗口）
            if prompt.size(1) > 64:
                prompt = prompt[:, -64:]
            
            # 模型预测
            logits = model(prompt)[:, -1, :] / temperature
            
            # Top-p (nucleus) sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 移除概率超过top_p的token
            sorted_probs[cumulative_probs > top_p] = 0
            
            if sorted_probs.sum() == 0:
                sorted_probs = torch.ones_like(sorted_probs) / sorted_probs.numel()
            else:
                sorted_probs /= sorted_probs.sum()
            
            # 采样下一个token
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_char_id = sorted_indices.gather(1, next_token).item()
            
            # 转换为字符
            next_char = id_to_word.get(next_char_id, '<UNK>')
            generated_text += next_char
            
            # 添加到prompt
            next_tensor = torch.tensor([[next_char_id]], device=device)
            prompt = torch.cat([prompt, next_tensor], dim=1)
            
            # 如果生成结束符，停止
            if next_char == '<EOS>':
                break
    
    return generated_text

if __name__ == "__main__":
    VOCAB_PATH = SCRIPT_DIR / "vocab.txt"
    MODEL_PATH = SCRIPT_DIR / "transformer_model.pth"

    if VOCAB_PATH.exists():
        VOCAB_PATH.unlink()
    text = load_corpus()
    build_vocab(text, str(VOCAB_PATH))

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()
    train(str(MODEL_PATH))

    word_to_id, id_to_word, unk_id = load_vocab(str(VOCAB_PATH))

    model = TransformerModel(len(word_to_id), HIDDEN_SIZE, NUM_LAYERS, NUM_HEADS, DROPOUT)
    model.load_state_dict(torch.load(str(MODEL_PATH)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    prompt = "注意力"
    generated_text = generate_text(model, word_to_id, id_to_word, prompt, 100, device=device)
    print(generated_text)