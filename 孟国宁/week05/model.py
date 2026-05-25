"""
    训练基于transformer的单向语言模型，并完成文本生成。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

# 先实现transformer部分
# 多头注意力实现
class MaskAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert self.d_model % self.num_heads == 0

        self.d_k = self.d_model // self.num_heads

        self.W_Q = nn.Linear(self.d_model, self.d_model)
        self.W_K = nn.Linear(self.d_model, self.d_model)
        self.W_V = nn.Linear(self.d_model, self.d_model)
        self.W_O = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        B, T, H = x.shape
        # 1. 计算QKV    [B, T, H(d_model)] -> [B, T, H], 权重矩阵[H, H]
        '''
            [B,T,H]
            →
            对每个token的H维向量
            做一次 H→H 的线性变换
            →
            [B,T,H]
        '''
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # 2. 按头数进行拆分
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B, T, H] -> [B, T, num_heads, d_k] -> [B, num_heads, T, d_k]
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # 3. 计算注意力分数 scores = Q @ K^T
        scores = Q @ K.transpose(-1, -2)    # [B, num_heads, T, d_k] @ [B, num_heads, d_k, T] = [B, num_heads, T, T]
        # 4. 进行缩放
        scores = scores / math.sqrt(self.d_k)   # [B, num_heads, T, T]

        # 添加掩码
        casual_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()   # torch.triu(..., diagonal=1),取上三角（不包括对角线）
        scores = scores.masked_fill(casual_mask, float("-inf"))     # 如果 mask=True → 把 attention score 变成 -inf

        # 5. 进行softmax归一化
        weights = torch.softmax(scores, dim = -1)  # [B, num_heads, T, T]
        # 6. 加权 output = weights @ V
        output = weights @ V    # [B, num_heads, T, T] @ [B, num_heads, T, d_k] = # [B, num_heads, T, d_k]
        # 7. 进行拼接
        output = output.transpose(1, 2).contiguous().view(B, T, H)
        # 8. 信息融合
        output = self.W_O(output)

        return output

# Add & Norm    y = LayerNorm(x + dropout(sublayer_output))
class AddNorm(nn.Module):
    def __init__(self, d_model, dropout = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):

        return self.norm(x + self.dropout(sublayer_output))


# FNN   y = W2(GELU(W1 x  + b1)) + b2
class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x

# 完整transformer
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout):
        super().__init__()

        self.attention = MaskAttention(d_model, num_heads)
        self.norm1 = AddNorm(d_model, dropout)
        self.fnn = FeedForward(d_model, hidden_dim)
        self.norm2 = AddNorm(d_model, dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x, attn_output)
        fnn_output = self.fnn(x)
        x = self.norm2(x, fnn_output)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(1024, d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embedding(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)

        x = token_emb + pos_emb

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.fc(x)

        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -1024:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

