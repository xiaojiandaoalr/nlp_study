import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        self.q_linear             = nn.Linear(embedding_dim, embedding_dim)
        self.k_linear             = nn.Linear(embedding_dim, embedding_dim)
        self.v_linear             = nn.Linear(embedding_dim, embedding_dim)
        self.dropout       = nn.Dropout(dropout)
        self.out           = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 线性变换并分头
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention(Q,K,V) = softmax( QKᵀ/√dₖ ) · V
        # 计算注意力分数
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_output = torch.matmul(self.dropout(attn_weights), v)

        # 将多头输出拼接回原始维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        return self.out(attn_output)

class FeedForward(nn.Module):
    def __init__(self, embedding_dim=768, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        return self.dropout(self.linear2(x))

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attention(x)
        out = self.norm1(x + self.dropout(attn_out))

        ff_out = self.feed_forward(x)
        return self.norm2(out + self.dropout(ff_out))

# TEST
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4
    ff_dim = 64

    x = torch.rand(batch_size, seq_len, embed_dim)  # 模拟输入

    encoder_layer = TransformerLayer(embed_dim, num_heads, ff_dim)
    out = encoder_layer(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
