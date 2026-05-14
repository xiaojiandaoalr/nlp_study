"""
用pytorch实现一个transformer层
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """多头注意力机制：允许模型同时关注不同位置的信息

    核心公式：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
    将输入分为多个头，每个头独立计算注意力，最后拼接输出
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_O(attn_output)
        return output, attn_weights


class FeedForward(nn.Module):
    """前馈神经网络：两层线性变换+GELU激活函数

    Transformer中每个位置的独立非线性变换
    FFN(x) = W2 * GELU(W1 * x)
    通常隐藏层维度是输入维度的4倍
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """位置编码：为序列中的每个位置添加位置信息

    使用正弦和余弦函数编码位置信息
    PE(pos,2i) = sin(pos / 10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
    这样可以让模型学习到相对位置关系
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层：包含自注意力和前馈网络

    每个编码器层有两个子层：
    1. Multi-Head Self-Attention + Add & Norm
    2. Feed-Forward Network + Add & Norm
    使用残差连接和层归一化稳定训练
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class DiyTransformerEncoder(nn.Module):
    """完整的Transformer编码器

    由以下部分组成：
    1. Token Embedding层：将词表索引映射为向量
    2. 位置编码：添加序列位置信息
    3. N个编码器层：堆叠多层TransformerEncoderLayer
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层：包含三个子层

    每个解码器层有三个子层：
    1. Masked Multi-Head Self-Attention + Add & Norm（因果掩码，防止看到未来信息）
    2. Multi-Head Cross-Attention + Add & Norm（Q来自解码器，K/V来自编码器）
    3. Feed-Forward Network + Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class DiyTransformerDecoder(nn.Module):
    """完整的Transformer解码器

    由以下部分组成：
    1. Token Embedding层：将词表索引映射为向量
    2. 位置编码：添加序列位置信息
    3. N个解码器层：堆叠多层TransformerDecoderLayer
    4. 输出层：将隐藏状态映射回词表大小
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.fc(x)


class DiyTransformer(nn.Module):
    """完整的Transformer模型：编码器-解码器架构

    工作流程：
    1. 源序列通过编码器提取上下文表示
    2. 目标序列通过解码器，结合编码器输出进行交叉注意力计算
    3. 解码器输出通过线性层预测下一个词的概率分布

    编码器：捕获源序列的上下文信息
    解码器：在给定源序列和已生成部分的情况下，预测下一个词
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff,
                 enc_layers, dec_layers, max_len=5000, dropout=0.1):
        super().__init__()
        self.encoder = DiyTransformerEncoder(src_vocab_size, d_model, num_heads, d_ff,
                                             enc_layers, max_len, dropout)
        self.decoder = DiyTransformerDecoder(tgt_vocab_size, d_model, num_heads, d_ff,
                                             dec_layers, max_len, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        return decoder_output


def generate_square_subsequent_mask(sz, device):
    """生成因果掩码（上三角为True）：防止解码器看到未来信息

    在自回归生成时，每个位置只能看到该位置及之前的内容
    """
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask


if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    d_model = 64
    num_heads = 4
    d_ff = 256
    num_layers = 2

    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len))

    torch.manual_seed(42)
    diy_transformer = DiyTransformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        enc_layers=num_layers,
        dec_layers=num_layers
    )
    diy_transformer.eval()

    tgt_mask = generate_square_subsequent_mask(seq_len, src.device)
    output = diy_transformer(src, tgt, tgt_mask=tgt_mask)

    print("=" * 60)
    print("DIY Transformer 测试结果")
    print("=" * 60)
    print(f"输入src形状: {src.shape}")
    print(f"输入tgt形状: {tgt.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出均值: {output.mean().item():.6f}")
    print(f"输出标准差: {output.std().item():.6f}")