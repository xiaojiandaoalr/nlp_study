import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分头: [batch, seq_len, d_model] -> [batch, n_head, seq_len, d_k]
        Q = self.w_q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, V)  # [batch, n_head, seq_len, d_k]
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(out)


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """位置编码"""
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


class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, d_model=512, n_head=8, n_layers=6, 
                 d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, src, mask=None):
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        
        x = src_emb
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, d_model=512, n_head=8, n_layers=6,
                 d_ff=2048, max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        x = tgt_emb
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return self.fc_out(x)


class Transformer(nn.Module):
    """完整Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8,
                 n_encoder_layers=6, n_decoder_layers=6, d_ff=2048,
                 max_len=5000, dropout=0.1, pad_idx=0):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_head, n_encoder_layers,
            d_ff, max_len, dropout, pad_idx
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_head, n_decoder_layers,
            d_ff, max_len, dropout, pad_idx
        )
        
    def make_src_mask(self, src, pad_idx=0):
        """创建源序列的padding mask"""
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt, pad_idx=0):
        """创建目标序列的mask（padding + 因果）"""
        batch_size, tgt_len = tgt.shape
        device = tgt.device
        
        # padding mask
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        
        # 因果mask
        causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        return tgt_pad_mask & (~causal_mask)
    
    def forward(self, src, tgt, pad_idx=0):
        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, pad_idx)
        
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型配置
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    n_head = 8
    n_encoder_layers = 6
    n_decoder_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_len = 5000
    pad_idx = 0
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        pad_idx=pad_idx
    ).to(device)
    
    # 测试数据
    batch_size = 32
    src_len = 20
    tgt_len = 25
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len)).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(src, tgt, pad_idx)
    
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    print(f"输出形状: {output.shape}")  # [batch, tgt_len, tgt_vocab_size]
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
