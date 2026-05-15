import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

# bert词典
tokenizer = BertTokenizer.from_pretrained(r"D:\八斗AI\workspace\models\bert-base-chinese")
encoded = tokenizer("世界你好！")
x = encoded["input_ids"]

# bert模型
bert = BertModel.from_pretrained(r"D:\八斗AI\workspace\models\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()
torch_x = torch.LongTensor([x])          #pytorch形式输入
#seqence_output, pooler_output = bert(torch_x)
#print(seqence_output.shape, pooler_output.shape)
# print(seqence_output, pooler_output)

print(bert.state_dict().keys())  #查看所有的权值矩阵名称


class DiyTransformer(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = bert.config.num_hidden_layers
        self.load_weights(state_dict)

    def load_bert_weithgs(self, state_dict):
        # 词向量矩阵
        self.word_embedding = state_dict["embeddings.word_embeddings.weight"].numpy()
        # 位置向量矩阵
        self.position_embedding = state_dict["embeddings.position_embeddings.weight"].numpy()
        # 句子向量矩阵
        self.token_type_embedding = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()

    def forward(self, x):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        pooler_output = self.pooler_output_layer(sequence_output[0])
        return sequence_output, pooler_output

    def pooler_output_layer(self, x):
        x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
        x = np.tanh(x)
        return x

    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.transformer_layer_forward(x, self.transformer_weights[i])
        return x

    def transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        attention_output = self.self_attention(x,
                                q_w, q_b,
                                k_w, k_b,
                                v_w, v_b,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias, num_attention_heads, hidden_size):
        q = np.dot(x, q_w) + q_b
        k = np.dot(x, k_w) + k_b
        v = np.dot(x, v_w) + v_b
        attention_head_size = int(hidden_size / num_attention_heads)
        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = np.matmul(q, k.swapaxes(1, 2))  
        qk /= np.sqrt(attention_head_size)
        qk = self.softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = np.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = np.dot(qkv, attention_output_weight.T) + attention_output_bias
        return attention

    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 2)
        return x

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def feed_forward(self,
                    x,
                    intermediate_weight,  # intermediate_size, hidden_size
                    intermediate_bias,  # intermediate_size
                    output_weight,  # hidden_size, intermediate_size
                    output_bias,  # hidden_size
                    ):
        # output shpae: [max_len, intermediate_size]
        x = np.dot(x, intermediate_weight.T) + intermediate_bias
        x = self.gelu(x)
        # output shpae: [max_len, hidden_size]
        x = np.dot(x, output_weight.T) + output_bias
        return x

    def gelu(self, x):
        return x * (1 + np.tanh(0.5 * (x / np.sqrt(2))))

    def embedding_forward(self, x):
        we = self.get_embedding(self.word_embedding, x)
        pe = self.get_embedding(self.position_embedding, [range(len(x))])
        te = self.get_embedding(self.token_type_embedding, [0] * len(x))
        em = we + pe + te
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[id] for id in x])

    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x


if __name__ == "__main__":
    db = DiyTransformer(state_dict)
    diy_sequence_output, diy_pooler_output = db.forward(x)
    #torch
    torch_sequence_output, torch_pooler_output = bert(torch_x)
    print(diy_sequence_output)
    print(torch_sequence_output)