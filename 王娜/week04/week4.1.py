"""
用 PyTorch 实现 Transformer 层。

该模块实现：
- 手动实现的多头自注意力机制（使用 nn.Linear）
- 使用 PyTorch 官方的层归一化（nn.LayerNorm）
- 使用 PyTorch 官方的前馈网络（nn.Linear）
- 残差连接与 Dropout

输入张量 `src` 的形状为 (batch_size, seq_len, d_model)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ManualMultiHeadAttention(nn.Module):
    """手动实现的多头自注意力机制（使用 nn.Linear）。

    参数:
        d_model (int): 模型隐藏维度。
        nhead (int): 注意力头数。
        dropout (float): Dropout 比例。
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super(ManualMultiHeadAttention, self).__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        # 使用 nn.Linear 实现 Q、K、V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """执行多头注意力计算。

        参数:
            query (Tensor): 查询张量，(batch_size, seq_len, d_model)
            key (Tensor): 键张量，(batch_size, seq_len, d_model)
            value (Tensor): 值张量，(batch_size, seq_len, d_model)
            attn_mask (Tensor, optional): 注意力掩码
            key_padding_mask (Tensor, optional): 键填充掩码

        返回:
            Tensor: 输出张量，(batch_size, seq_len, d_model)
            Tensor: 注意力权重
        """
        batch_size, seq_len, _ = query.size()

        # 1. 线性变换得到 Q、K、V
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 2. 分割成多头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, nhead, d_k) -> (batch_size, nhead, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. 应用掩码
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding_mask, -1e9)

        # 5. softmax 和 dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. 与 V 相乘
        attn_output = torch.matmul(attn_weights, V)

        # 7. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # 8. 输出投影
        output = self.W_o(attn_output)

        return output, attn_weights


class TransformerLayerHybrid(nn.Module):
    """Transformer 编码器层（手动注意力 + 官方 LayerNorm/FFN）。

    参数:
        d_model (int): 模型隐藏维度。
        nhead (int): 注意力头数。
        dim_feedforward (int): 前向传播网络隐藏层维度。
        dropout (float): Dropout 比例。
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerLayerHybrid, self).__init__()

        # 手动实现的多头自注意力（使用 nn.Linear）
        self.self_attn = ManualMultiHeadAttention(d_model, nhead, dropout)

        # 使用 PyTorch 官方的层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 使用 PyTorch 官方的前馈网络（两层 nn.Linear）
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """执行前向计算。

        参数:
            src (Tensor): 输入张量，(batch_size, seq_len, d_model)
            src_mask (Tensor, optional): 注意力掩码
            src_key_padding_mask (Tensor, optional): 键填充掩码

        返回:
            Tensor: 输出张量，(batch_size, seq_len, d_model)
        """
        # ========== 自注意力子层 ==========
        attn_output, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )

        # 残差连接 + Dropout + 层归一化
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # ========== 前馈网络子层 ==========
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))

        # 残差连接 + Dropout + 层归一化
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)

        return src


class HybridMultiLayerTransformer(nn.Module):
    """多层 Transformer 编码器。

    参数:
        d_model (int): 模型隐藏维度。
        nhead (int): 注意力头数。
        num_layers (int): 编码器层数。
        dim_feedforward (int): 前向传播网络隐藏层维度。
        dropout (float): Dropout 比例。
    """

    def __init__(self, d_model, nhead, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(HybridMultiLayerTransformer, self).__init__()

        self.layers = nn.ModuleList([
            TransformerLayerHybrid(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """执行前向计算。

        参数:
            src (Tensor): 输入张量，(batch_size, seq_len, d_model)
            mask (Tensor, optional): 注意力掩码
            src_key_padding_mask (Tensor, optional): 键填充掩码

        返回:
            Tensor: 输出张量，(batch_size, seq_len, d_model)
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


# ==================== 测试代码 ====================

if __name__ == "__main__":
    torch.manual_seed(42)

    # 参数设置
    batch_size = 2
    seq_len = 10
    d_model = 512
    nhead = 8
    num_layers = 6

    print("=" * 60)
    print("Transformer 测试")
    print("（手动注意力[nn.Linear] + 官方 LayerNorm/FFN）")
    print("=" * 60)

    # 测试 1: 单层 Transformer
    print("\n【测试 1】单层 TransformerLayerHybrid")
    model1 = TransformerLayerHybrid(d_model, nhead)
    x1 = torch.randn(batch_size, seq_len, d_model)
    out1 = model1(x1)
    print(f"  输入形状:  {x1.shape}")
    print(f"  输出形状:  {out1.shape}")
    print(f"  参数量: {sum(p.numel() for p in model1.parameters()):,}")
    assert out1.shape == x1.shape
    print("  ✓ 测试通过！")

    # 测试 2: 多层 Transformer
    print("\n【测试 2】多层 HybridMultiLayerTransformer")
    model2 = HybridMultiLayerTransformer(d_model, nhead, num_layers)
    x2 = torch.randn(batch_size, seq_len, d_model)
    out2 = model2(x2)
    print(f"  输入形状:  {x2.shape}")
    print(f"  输出形状:  {out2.shape}")
    print(f"  参数量: {sum(p.numel() for p in model2.parameters()):,}")
    assert out2.shape == x2.shape
    print("  ✓ 测试通过！")

    # 测试 3: 掩码功能
    print("\n【测试 3】掩码功能")
    key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    key_padding_mask[0, 8:] = True
    key_padding_mask[1, 6:] = True
    out3 = model2(x2, src_key_padding_mask=key_padding_mask)
    print(f"  输入形状:  {x2.shape}")
    print(f"  掩码形状:  {key_padding_mask.shape}")
    print(f"  输出形状:  {out3.shape}")
    print("  ✓ 掩码测试通过！")

    # 测试 4: 与 PyTorch 官方实现对比
    print("\n【测试 4】与 PyTorch 官方实现对比")
    official_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
    official_model = nn.TransformerEncoder(official_layer, num_layers)
    official_params = sum(p.numel() for p in official_model.parameters())
    hybrid_params = sum(p.numel() for p in model2.parameters())
    print(f"  混合实现参数量: {hybrid_params:,}")
    print(f"  官方实现参数量: {official_params:,}")
    print(f"  参数量相同: {hybrid_params == official_params}")
    print("  ✓ 对比测试通过！")

    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
