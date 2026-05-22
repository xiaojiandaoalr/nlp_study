import torch
import torch.nn as nn
import math

# ========================
# 🔥 位置编码（必须和训练一样）
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=23):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ========================
# 🔥 模型必须和训练完全一致！
# ========================
class MyNlpModel(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.vocab_len = vocab_len
        dim = 128
        self.emb = nn.Embedding(self.vocab_len, dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(dim)
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=dim*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(transformer_decoder_layer, num_layers=3)
        self.linear = nn.Linear(dim, self.vocab_len)

    def triu_mask(self, t):
        ones = torch.ones(t, t)
        triu = torch.triu(ones, diagonal=1)
        mask = triu.masked_fill(triu == 1, float('-inf'))
        return mask.to(self.emb.weight.device)

    def forward(self, input):
        emb = self.emb(input)
        emb = self.position_encoding(emb)
        
        mask = self.triu_mask(emb.size(1))
        out = self.transformer(
            tgt=emb,
            memory=emb,
            tgt_mask=mask
        )
        return self.linear(out)

# ======================
# 加载模型
# ======================
def load_model():
    vocab = torch.load("vocab_dict.pth")
    vocab_len = len(vocab)

    model = MyNlpModel(vocab_len)
    model.load_state_dict(torch.load("my_nlp_model.pth"))
    model.eval()
    print("✅ Transformer 模型加载成功！")
    return model, vocab

# ======================
# 生成句子
# ======================
def generate(model, vocab, start_text, max_len=20):
    idx2word = {v: k for k, v in vocab.items()}
    input_ids = [vocab.get(c, 1) for c in start_text]

    with torch.no_grad():
        for _ in range(max_len):
            x = torch.LongTensor([input_ids])
            out = model(x)
            
            # 🔥 改成随机采样，不要死抓最大概率！
            logits = out[:, -1, :]
            # 过滤掉低概率词，防止乱输出
            topk = torch.topk(logits, k=3)
            idx = torch.multinomial(torch.softmax(topk.values, dim=-1), 1)
            next_id = topk.indices[0][idx].item()
            
            input_ids.append(next_id)

    return "".join([idx2word[i] for i in input_ids])

# ======================
# 测试
# ======================
if __name__ == "__main__":
    model, vocab = load_model()

    while True:
        s = input("\n输入开头（q退出）：")
        if s == 'q':
            break
        res = generate(model, vocab, s)
        print("生成：", res)