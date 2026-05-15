
import torch
import torch.nn as nn
import torch.nn.functional as F

#手动实现自注意力机制 
class MySelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim #词向量维度
        self.num_heads = num_heads #注意力头数
        self.head_dim = embed_dim // num_heads #每个头的维度
        
        #手动定义Q，K，V 三个线性层
        self.wq = nn.Linear(embed_dim,embed_dim)
        self.wk = nn.Linear(embed_dim,embed_dim)
        self.wv = nn.Linear(embed_dim,embed_dim)
        
        #输出线性层
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        #计算Q, k, V
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        
        #拆分成多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        #计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        #加权求和
        out = torch.matmul(attn_weights, V)
        
        #拼接多头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        #线性层
        out = self.out_proj(out)
        return out
    
#手动实现前馈网络
class MyFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()   
        #2层线性层 + 激活函数
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)   
        return x
    
#手动实现完整Transform层    
class MyTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        #自注意力
        self.self_attn = MySelfAttention(embed_dim, num_heads)
        #前馈网络
        self.ffn = MyFeedForward(embed_dim, hidden_dim)
        
        #层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        #Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #自注意力+残差+归一化
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out)) #残差连接
        
        #前馈网络+残差+归一化
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out)) #残差连接
        
        return x
    

#测试代码
if __name__ == "__main__":
    batch_size = 2
    seq_len = 3
    embed_dim = 16
    num_heads = 2
    hidden_dim = 32
    
    #创建随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)
    print(f"输入张量形状：{x.shape}")  
    
    my_transformer = MyTransformerLayer(embed_dim, num_heads, hidden_dim)
    
    out_put = my_transformer(x)
    print(f"Transformer层输出形状：,{out_put.shape}")
    print("Transformer层运行成功！")
    
