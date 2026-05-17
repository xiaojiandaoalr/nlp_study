"""
简单的 PyTorch Transformer 层实现。

该模块实现一个标准的 Transformer 编码器层，包括：
- 多头自注意力机制（Multi-head Self-Attention）
- 前向传播全连接网络（Feed-Forward Network）
- 残差连接与层归一化（Residual Connection + LayerNorm）

输入张量 `src` 的形状为 (seq_len, batch_size, d_model)，符合 PyTorch `nn.MultiheadAttention` 的默认输入格式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerLayer(nn.Module):
    """单层 Transformer 编码器。

    参数:
        d_model (int): 模型隐藏维度。
        nhead (int): 注意力头数。
        dim_feedforward (int): 前向传播网络隐藏层维度。
        dropout (float): Dropout 比例。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayer, self).__init__()

        # 多头自注意力模块：输入、输出维度均为 d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈网络的两层线性变换，中间使用 ReLU 激活
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 两个层归一化，分别用于自注意力子层和前馈子层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        """执行前向计算。

        参数:
            src (Tensor): 输入张量，形状为 (seq_len, batch_size, d_model)

        返回:
            Tensor: 输出张量，形状与输入相同。
        """

        # 自注意力：查询、键、值都来自同一个输入 src
        attn_output, _ = self.self_attn(src, src, src)

        # 第一条残差连接 + 层归一化
        # src + attn_output 保留原始输入信息，并缓解梯度消失
        src = self.norm1(src + attn_output)

        # 前馈网络：线性变换 -> ReLU -> Dropout -> 线性变换
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))

        # 第二条残差连接 + 层归一化
        src = self.norm2(src + ff_output)

        return src
    
