""" 手推Bert实现

    input
      |
    Embedding层 ———— token Embedding + position Embedding + segment Embedding
      |
    Transformer层 ———— Muti-self-Attention -- Add & Norm -- FNN -- Add & Norm
      |
    Output

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

# ======================================================================
# 1. 输入处理（文本——>token——>token_id
# ======================================================================

# 词表
vocab = {
    "我" : 0,
    "爱" : 1,
    "你" : 2
}

sentence = ["我", "爱", "你"]
# token——>token_id
input_ids = torch.tensor([
    vocab[word] for word in sentence
])

print(f"\n输入id：{input_ids}")

# ======================================================================
# 2. Embedding层
# ======================================================================

# 参数定义
vocab_size = 3  # 词表大小
max_len = 10    # 最大序列长度，句子最大有多长
d_model = 4     # 向量维度，每个词用多少维的向量表示


# --------------token Embedding----------------------------------------
token_embedding = nn.Embedding(
    num_embeddings = vocab_size,  # 词表大小
    embedding_dim = d_model   # 向量维度，每个单词用多少维的向量来进行表示
)

# ---------------position Embedding-------------------------------------
position_embedding = nn.Embedding(
    num_embeddings = max_len,
    embedding_dim = d_model
)
# 位置id
position_ids = torch.arange(len(sentence))
print(f"\n位置id：\n{position_ids}")

# --------------segment Embedding-----------------------------------------
segment_embedding = nn.Embedding(
    num_embeddings = 2,
    embedding_dim = d_model
)
# 句子id
segment_ids = torch.zeros(len(sentence), dtype = torch.long)
print(f"\n句子id：\n{segment_ids}")

token_embed = token_embedding(input_ids)
position_embed = position_embedding(position_ids)
segment_embed = segment_embedding(segment_ids)

bert_input = (token_embed + position_embed + segment_embed)
print(f"\nbert经过Embedding层后最终输入：\n{bert_input}")


# ======================================================================
# 3. Transformer层
# ======================================================================

# self-attention，单头自注意力
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 1. 先计算Q、K、V
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # 2. 计算 注意力分数 scores = QK^T
        scores = Q @ K.transpose(-2, -1)    # transpose()交换任意两个维度，为什么不直接转置是因为经常前面可能会有batch_size
        # 3. 进行缩放 scores / √d_k
        scores = scores / math.sqrt(x.size(-1))
        # 4. softmax归一化
        attention_weights = torch.softmax(scores, dim = -1)
        # 5. 计算Output，output = attention_weights @ V
        output = attention_weights @ V

        return output


# transformer的encoder层
# 多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0     # 断言语句，如果执行结果为False，程序会报错停止

        self.d_k = d_model // num_heads     # 每个注意力头的维度

        self.W_Q = nn.Linear(d_model, d_model)  # 用线性层实现矩阵相乘
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        # 1. 计算QKV
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # 2. 将QKV 按头数切分
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        # 3. 计算注意力分数 scores = Q @ K^T
        scores = Q @ K.transpose(-2, -1)
        # 4. 特征缩放
        scores = scores / math.sqrt(self.d_k)
        # 5. 进行softmax归一化
        attn = torch.softmax(scores, dim = -1)
        # 6. 计算输出
        output = attn @ V
        # 7. 将多个注意力头进行拼接
        output = output.transpose(1, 2).contiguous().view(B, T, C)  # view()操作要求张量的内存要连续，转置之后内存不连续，需要使用contiguous()使内存连续

        return self.W_O(output)  # W_O 将多个头计算后的信息进行融合

# 残差和归一化
class AddNorm(nn.Module):
    def __init__(self, hidden_size, dropout = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)   # 层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.norm(x + self.dropout(sublayer_output))

# 前向传播层
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

# 完整的transformer层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        # 多头注意力
        self.attention = MultiHeadAttention(d_model, num_heads)
        # 第一个 Add & Norm
        self.norm1 = AddNorm(d_model, dropout)
        # FNN层
        self.fnn = FeedForward(d_model, hidden_dim)
        # 第二个Add & Norm
        self.norm2 = AddNorm(d_model, dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x, attn_output)
        fnn_output = self.fnn(x)
        x = self.norm2(x, fnn_output)

        return x

# 测试
bert_input = bert_input.unsqueeze(0)
encoder = EncoderLayer(
    d_model=4,
    num_heads=2,
    hidden_dim=16
)
output = encoder(bert_input)
print(output.shape)
print(output)
