import torch
import torch.nn as nn

#构造字符表
vocab = {
    "[pad]" : 0,
    "你" : 1,
    "你好" : 2,
    "中国" : 3,
    "好" : 4,
    "[cls]" : 5,
    "[sep]" : 6,
    "[unk]":7
}

vocab_size = len(vocab)
embedding_dim = 256

token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab["[pad]"])
segment_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab["[sep]"])
position_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab["[sep]"])

#构造输入
#       [cls] 你 您好 中国 [sep] 中国 好 [sep]
token = [5,   1,  2,   3,   6,  3,  4,  6]
seg =   [0,   0,  0,  0,   0,  1,  1,  1]
pos =   [0,   1,  2,   3,   4,  5,  6,  7]

tensor_token_embedding = token_embedding(torch.LongTensor(token))
tensor_position_embedding = position_embedding(torch.LongTensor(pos))
tensor_segment_embedding = segment_embedding(torch.LongTensor(seg))

output = tensor_token_embedding + tensor_position_embedding + tensor_segment_embedding

print(tensor_token_embedding.shape)
print(tensor_position_embedding.shape)
print(tensor_segment_embedding.shape)
print(output.shape)