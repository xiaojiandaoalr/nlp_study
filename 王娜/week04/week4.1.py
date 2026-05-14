"""
手动实现多头自注意力机制的 PyTorch Transformer 层。

该模块实现一个标准的 Transformer 编码器层，包括：
- 手动实现的多头自注意力机制（Multi-head Self-Attention）
- 前向传播全连接网络（Feed-Forward Network）
- 残差连接与层归一化（Residual Connection + LayerNorm）

输入张量 `src` 的形状为 (batch_size, seq_len, d_model)，更符合直观理解。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """手动实现的多头自注意力机制。

    参数:
        d_model (int): 模型隐藏维度。
        nhead (int): 注意力头数。
        dropout (float): Dropout 比例。
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead  # 每个头的维度

        # 线性变换：将输入映射到 Q、K、V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性变换
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """计算缩放点积注意力。

        参数:
            Q (Tensor): 查询张量，形状为 (batch_size, nhead, seq_len, d_k)
            K (Tensor): 键张量，形状为 (batch_size, nhead, seq_len, d_k)
            V (Tensor): 值张量，形状为 (batch_size, nhead, seq_len, d_k)
            mask (Tensor, optional): 掩码张量，形状为 (batch_size, 1, seq_len, seq_len)

        返回:
            Tensor: 注意力输出，形状为 (batch_size, nhead, seq_len, d_k)
            Tensor: 注意力权重，形状为 (batch_size, nhead, seq_len, seq_len)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores 形状: (batch_size, nhead, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        # output 形状: (batch_size, nhead, seq_len, d_k)

        return output, attn_weights

    def forward(self, src, mask=None):
        """执行前向计算。

        参数:
            src (Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
            mask (Tensor, optional): 掩码张量

        返回:
            Tensor: 输出张量，形状为 (batch_size, seq_len, d_model)
            Tensor: 注意力权重
        """
        batch_size, seq_len, _ = src.size()

        # 1. 线性变换得到 Q、K、V
        Q = self.W_q(src)  # (batch_size, seq_len, d_model)
        K = self.W_k(src)
        V = self.W_v(src)

        # 2. 分割成多个头
        # 从 (batch_size, seq_len, d_model) 变为 (batch_size, seq_len, nhead, d_k)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k)

        # 3. 转置为 (batch_size, nhead, seq_len, d_k) 以便并行计算
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 4. 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 5. 合并多头结果
        # 从 (batch_size, nhead, seq_len, d_k) 变为 (batch_size, seq_len, nhead, d_k)
        attn_output = attn_output.transpose(1, 2)
        # 合并为 (batch_size, seq_len, d_model)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)

        # 6. 最终线性变换
        output = self.W_o(attn_output)

        return output, attn_weights


class TransformerLayerManualAttention(nn.Module):
    """单层 Transformer 编码器（手动实现注意力）。

    参数:
        d_model (int): 模型隐藏维度。
        nhead (int): 注意力头数。
        dim_feedforward (int): 前向传播网络隐藏层维度。
        dropout (float): Dropout 比例。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayerManualAttention, self).__init__()

        # 手动实现的多头自注意力
        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout=dropout)

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 额外的 dropout 用于残差连接前
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        """执行前向计算。

        参数:
            src (Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
            mask (Tensor, optional): 掩码张量

        返回:
            Tensor: 输出张量，形状与输入相同。
        """
        # 自注意力子层
        attn_output, _ = self.self_attn(src, mask)
        attn_output = self.dropout1(attn_output)
        src = self.norm1(src + attn_output)

        # 前馈子层
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        ff_output = self.dropout2(ff_output)
        src = self.norm2(src + ff_output)

        return src


# 测试代码
if __name__ == "__main__":
    # 设置随机种子保证可重复
    torch.manual_seed(42)

    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    nhead = 8

    # 创建模型
    model = TransformerLayerManualAttention(d_model, nhead)

    # 创建随机输入 (batch_size, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    output = model(x)

    print(f"输入形状:  {x.shape}")
    print(f"输出形状:  {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 验证输出形状是否正确
    assert output.shape == x.shape, "输出形状应与输入形状相同"
    print("\n✓ 测试通过！")
