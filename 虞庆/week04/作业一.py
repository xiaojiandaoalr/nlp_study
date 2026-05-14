import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiyTransformerLayerTorch(nn.Module):
    def __init__(self, weights, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 解包权重
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attn_out_w, attn_out_b, \
        attn_norm_w, attn_norm_b, \
        inter_w, inter_b, \
        out_w, out_b, \
        ff_norm_w, ff_norm_b = weights

        # ===================== 注意力层 =====================
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attn_out = nn.Linear(hidden_size, hidden_size)

        # 赋值预训练权重
        self.query.weight = nn.Parameter(torch.from_numpy(q_w))
        self.query.bias = nn.Parameter(torch.from_numpy(q_b))
        self.key.weight = nn.Parameter(torch.from_numpy(k_w))
        self.key.bias = nn.Parameter(torch.from_numpy(k_b))
        self.value.weight = nn.Parameter(torch.from_numpy(v_w))
        self.value.bias = nn.Parameter(torch.from_numpy(v_b))
        self.attn_out.weight = nn.Parameter(torch.from_numpy(attn_out_w))
        self.attn_out.bias = nn.Parameter(torch.from_numpy(attn_out_b))

        # ===================== LayerNorm =====================
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.attn_norm.weight = nn.Parameter(torch.from_numpy(attn_norm_w))
        self.attn_norm.bias = nn.Parameter(torch.from_numpy(attn_norm_b))
        self.ff_norm.weight = nn.Parameter(torch.from_numpy(ff_norm_w))
        self.ff_norm.bias = nn.Parameter(torch.from_numpy(ff_norm_b))

        # ===================== FFN =====================
        self.intermediate = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)
        self.intermediate.weight = nn.Parameter(torch.from_numpy(inter_w))
        self.intermediate.bias = nn.Parameter(torch.from_numpy(inter_b))
        self.output.weight = nn.Parameter(torch.from_numpy(out_w))
        self.output.bias = nn.Parameter(torch.from_numpy(out_b))

    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def transpose_for_scores(self, x):
        # x: [seq_len, hidden] -> [num_heads, seq_len, head_dim]
        seq_len, hidden = x.shape
        x = x.view(seq_len, self.num_heads, self.head_dim)
        x = x.transpose(0, 1)  # [heads, seq_len, head_dim]
        return x

    def self_attention(self, x):
        # x: [seq_len, hidden]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)

        # 加权
        context = torch.matmul(attn_probs, v)
        context = context.transpose(0, 1).contiguous().view(seq_len, self.hidden_size)

        attn_out = self.attn_out(context)
        return attn_out

    def feed_forward(self, x):
        x = self.intermediate(x)
        x = self.gelu(x)
        x = self.output(x)
        return x

    def forward(self, x):
        # 注意力 + 残差 + norm
        attn_out = self.self_attention(x)
        x = self.attn_norm(x + attn_out)

        # FFN + 残差 + norm
        ff_out = self.feed_forward(x)
        x = self.ff_norm(x + ff_out)
        return x


# ================= 多层 Transformer 编码器 =================
class DiyTransformerEncoderTorch(nn.Module):
    def __init__(self, layers_weights):
        super().__init__()
        self.layers = nn.ModuleList([
            DiyTransformerLayerTorch(w) for w in layers_weights
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
