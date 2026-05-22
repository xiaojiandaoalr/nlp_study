"""
字符级 Transformer 单向语言模型（Causal LM），使用 Causal Mask 实现自回归预测。
用法:
    python transformer_lm.py --epochs 30
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 语料 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── Causal Mask ───────────────────────────

def generate_causal_mask(seq_len, device):
    """
    生成因果掩码（Causal Mask），形状为 (1, 1, seq_len, seq_len)。
    上三角部分（不含对角线）填充 -inf，使 softmax 后未来位置的注意力权重为 0。
    """
    # torch.triu 返回上三角矩阵，diagonal=1 排除对角线
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T) 便于多头广播


# ─────────────────────────── 模型组件 ───────────────────────────

class CausalSelfAttention(nn.Module):
    """带 Causal Mask 的多头自注意力"""

    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B, H, T, T)

        # 施加 Causal Mask：将上三角（未来位置）的分数设为 -inf
        if causal_mask is not None:
            scores = scores + causal_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer 解码器块"""

    def __init__(self, hidden_size, num_heads, intermediate_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """基于 Transformer 的单向语言模型"""

    def __init__(self, vocab_size, hidden_size, num_heads, num_layers,
                 intermediate_size, max_seq_len, dropout):
        """
        vocab_size:        词表大小（字符总数），为每个字符分配一个嵌入向量
        hidden_size:       模型隐藏维度 (d_model)，贯穿嵌入、注意力、FFN 的统一维度
        num_heads:         多头注意力的头数，将 hidden_size 拆成 num_heads 个子空间并行计算
                           必须满足 hidden_size % num_heads == 0，每头维度 = hidden_size // num_heads
        num_layers:        Transformer 堆叠的层数（TransformerBlock 的个数）
        intermediate_size: FFN 中间层维度 (d_ff)，先升维再降回 hidden_size，形成窄→宽→窄瓶颈结构
                           通常为 2~4 × hidden_size，让 FFN 有更大表达空间
        max_seq_len:       最大输入序列长度，位置编码的上限；生成时截断到此长度
        dropout:           Dropout 比率，在注意力权重、FFN 中间层、嵌入输出等处随机丢弃以防过拟合
        """
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        # 权重共享：输出层与词嵌入共用权重
        self.head.weight = self.tok_embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        causal_mask = generate_causal_mask(T, x.device)             # (1, 1, T, T)

        x = self.drop(self.tok_embed(x) + self.pos_embed(positions))
        for block in self.blocks:
            x = block(x, causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 生成文本 ───────────────────────────

@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_new_tokens=100, device="cpu", temperature=0.8, top_p=0.9):
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # 截断到最大序列长度
        x_cond = x[:, -model.pos_embed.num_embeddings:]
        logits = model(x_cond)[:, -1, :]  # 只取最后一个位置
        logits = logits / temperature

        # Top-P (Nucleus) 采样：从概率累积和达到 p 的最小词集中采样
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 将累积概率超过 top_p 的位置移除（置为 -inf）
        sorted_mask = cumulative_probs - sorted_probs > top_p
        sorted_logits[sorted_mask] = float("-inf")

        # 恢复原始顺序
        logits.scatter_(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    generated = "".join(idx2char[i.item()] for i in x[0])
    return generated


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",           type=int,   default=30)
    parser.add_argument("--seq_len",          type=int,   default=128)
    parser.add_argument("--batch_size",       type=int,   default=64)
    parser.add_argument("--hidden_size",      type=int,   default=256)
    parser.add_argument("--num_heads",        type=int,   default=4)
    parser.add_argument("--num_layers",       type=int,   default=4)
    parser.add_argument("--intermediate_size", type=int,  default=512)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--val_ratio",        type=float, default=0.05)
    parser.add_argument("--corpus",           default="*.txt")
    parser.add_argument("--save",             default="best_transformer_lm.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  Transformer Causal LM")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.seed(42)
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # 加载最佳模型并生成示例文本
    ckpt = torch.load(args.save, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    idx2char = ckpt["idx2char"]
    char2idx = ckpt["char2idx"]

    prompts = ["黄金", "原油", "沪铜"]
    print("\n" + "=" * 50)
    print("生成示例:")
    print("=" * 50)
    for p in prompts:
        result = generate(model, char2idx, idx2char, p, max_new_tokens=80, device=device)
        print(f"\nPrompt: {p}")
        print(f"生成:   {result}")


if __name__ == "__main__":
    main()
