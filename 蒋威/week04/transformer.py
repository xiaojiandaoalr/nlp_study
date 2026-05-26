# bert的transformer层
import torch
import torch.nn as nn
import torch.nn.functional as F  # 纯函数，无内部参数，适合操作区别于nn.Module
import math


class Attention(nn.Module):
    def __init__(self, input_dim=768, hidden_num=12):
        super().__init__()

        # 必须整除确保维度匹配
        assert input_dim % hidden_num == 0

        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.d_k = input_dim // hidden_num

        # 一次计算 Q、K、V 矩阵,数据只被计算一次,计算密度高更好利用显卡内存
        self.qkv = nn.Linear(input_dim, input_dim * 3)

        # 输出投影混淆矩阵,否则不同头的信息只是简单拼接,头之间没有交互
        self.out = nn.Linear(input_dim, input_dim)

    def attention(self, _Q, _K, _V, mask=None):
        Q = _Q.transpose(1, 2)
        K = _K.transpose(1, 2)
        V = _V.transpose(1, 2)
        
        K = K.transpose(-1, -2)
        scores = Q @ K / math.sqrt(self.d_k)

        if mask is not None:
            # float('-inf') 表示负无穷,softmax 后这些位置的权重就是e(-inf) = 0
            scores = scores.masked_fill(mask, float("-inf"))

        _attention = F.softmax(scores, dim=-1)

        # shape: (batch_size, hidden_num, seq_len, d_k)
        output = _attention @ V
        return output

    def forward(self, data, mask=None):
        B, T, _ = data.size()
        
        QKV = self.qkv(data).chunk(3, dim=-1)
        # 为了保留对多头和token长度的信息,必须先transpose再contiguous再view
        Q = QKV[0].view(B, T, self.hidden_num, self.d_k)
        K = QKV[1].view(B, T, self.hidden_num, self.d_k)
        V = QKV[2].view(B, T, self.hidden_num, self.d_k)

        attention_output = self.attention(Q, K, V, mask)

        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(B, -1, self.input_dim)
        )

        # 最后通过输出投影，让不同头的信息混合
        output = self.out(attention_output)
        return output


class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.fnn(x)


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num, dropout=0.1):
        super().__init__()
        self.attention = Attention(input_dim, hidden_num)
        self.feed_forward = FNN(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm_attention = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

    def forward(self, qkv, mask=None):
        # 标准四步子层
        attention_output = self.attention(qkv, mask=mask)
        qkv = self.norm_attention(qkv + self.dropout(attention_output))
        ffn_output = self.feed_forward(qkv)
        qkv = self.norm_ffn(qkv + self.dropout(ffn_output))
        return qkv

class BERT(nn.Module):
    def __init__(self, vocab_size = 30522, input_dim=768, hidden_num=12, num_layers=12, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embedding = nn.Embedding(vocab_size, input_dim)
        fnn_dim = 4 * input_dim
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(input_dim, fnn_dim, hidden_num, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(input_dim)

    def forward(self, data, mask=None):
        x = self.embedding(data)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        return x  # (batch, seq_len, 768) 上下文表示


if __name__ == "__main__":
    vocab_size = 128  # 词表大小（BERT 原始词表）
    batch_size = 2      # 批次大小
    seq_len = 128       # 序列长度

    model = BERT(vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    output = model(x, mask)

    print("Output shape:", output.shape)