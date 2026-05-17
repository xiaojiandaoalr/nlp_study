"""
单块 Transformer 编码器层：手写多头缩放点积自注意力 + FFN + 残差 + LayerNorm。
不依赖 nn.MultiheadAttention，便于对照公式实现与调试。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力：Q/K/V 投影、缩放点积 softmax、对 V 加权求和、多头拼接与输出投影。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, batch_first: bool = False):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除")

        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.batch_first = batch_first
        self.scale = self.d_head ** -0.5

        self.W_q = nn.Linear(d_model, d_model, bias=True)
        self.W_k = nn.Linear(d_model, d_model, bias=True)
        self.W_v = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def _to_bhld(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """统一为 (batch, seq, d_model)，返回 (张量, 是否从 (L,B,D) 转来)。"""
        if self.batch_first:
            return x, False
        return x.transpose(0, 1), True

    def _from_bhld(self, x: torch.Tensor, was_lb: bool) -> torch.Tensor:
        if was_lb:
            return x.transpose(0, 1)
        return x

    def _apply_attn_mask(
        self,
        scores: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        batch: int,
        tgt_len: int,
        src_len: int,
    ) -> torch.Tensor:
        """将 attn_mask 合并到 scores (B, H, tgt, src)。"""
        if attn_mask is None:
            return scores
        if attn_mask.dim() == 2 and attn_mask.shape == (tgt_len, src_len):
            m = attn_mask.unsqueeze(0).unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                return scores.masked_fill(m, float("-inf"))
            return scores + m
        if attn_mask.dim() == 3 and attn_mask.shape == (batch * self.nhead, tgt_len, src_len):
            m = attn_mask.view(batch, self.nhead, tgt_len, src_len)
            if attn_mask.dtype == torch.bool:
                return scores.masked_fill(m, float("-inf"))
            return scores + m
        if attn_mask.dim() == 4 and attn_mask.shape == (batch, self.nhead, tgt_len, src_len):
            if attn_mask.dtype == torch.bool:
                return scores.masked_fill(attn_mask, float("-inf"))
            return scores + attn_mask
        raise ValueError(
            "attn_mask 支持形状 (L, L)、(B*nhead, L, L) 或 (B, nhead, L, L)"
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, was_lb = self._to_bhld(x)
        b, t, d = x.shape

        q = self.W_q(x).view(b, t, self.nhead, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(b, t, self.nhead, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(b, t, self.nhead, self.d_head).transpose(1, 2)
        # (B, H, T, Dh)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = self._apply_attn_mask(scores, attn_mask, b, t, t)

        if key_padding_mask is not None:
            # True 表示该 key 位置为 padding，整列置 -inf
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)
        out = self.out_proj(out)
        out = self._from_bhld(out, was_lb)

        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            # 与各 head 平均后的 (B, T, T) 对齐，便于可视化
            attn_weights = attn.mean(dim=1)

        return out, attn_weights


class TransformerLayer(nn.Module):
    """一层 Transformer Encoder：Pre-LayerNorm + 手写自注意力 + 前馈，带残差连接。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除")

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.batch_first = batch_first

        self.self_attn = MultiHeadSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError("activation 应为 'relu' 或 'gelu'")

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """
        Args:
            src: 若 batch_first=False，形状 (seq_len, batch, d_model)；
                 若 batch_first=True，形状 (batch, seq_len, d_model)。
            src_mask: 加在注意力 logits 上；支持 (L,L)、(B*nhead,L,L)、(B,nhead,L,L)。
            src_key_padding_mask: (batch, seq_len)，True 表示该 key 为 padding。
            need_weights: True 时额外返回各 head 平均后的注意力 (batch, L, L)。

        Returns:
            默认返回输出张量；need_weights=True 时返回 (out, attn_weights)。
        """
        x = src
        residual = x
        x = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights,
        )
        x = residual + self.dropout1(attn_out)

        residual = x
        x = self.norm2(x)
        x = residual + self._ffn(x)

        if need_weights:
            return x, attn_weights
        return x


def _demo() -> None:
    batch, seq, d_model, nhead = 2, 10, 256, 8
    layer = TransformerLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=1024,
        dropout=0.1,
        batch_first=True,
    )
    x = torch.randn(batch, seq, d_model)
    padding = torch.zeros(batch, seq, dtype=torch.bool)
    padding[:, 8:] = True
    out = layer(x, src_key_padding_mask=padding)
    assert out.shape == x.shape
    print("TransformerLayer 输出形状:", tuple(out.shape))
    out2, w = layer(x, src_key_padding_mask=padding, need_weights=True)
    assert w is not None and w.shape == (batch, seq, seq)
    print("注意力权重形状 (对 head 平均):", tuple(w.shape))

    # 默认 Post-LN，与手写 Pre-LN 数值不同，此处只核对形状与能否跑通
    ref = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=1024,
        dropout=0.1,
        batch_first=True,
        norm_first=False,
    )
    with torch.no_grad():
        y = ref(x, src_key_padding_mask=padding)
    assert y.shape == x.shape
    print("nn.TransformerEncoderLayer 参考输出形状:", tuple(y.shape))


if __name__ == "__main__":
    _demo()
