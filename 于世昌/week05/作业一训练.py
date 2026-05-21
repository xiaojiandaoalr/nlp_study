"""
基于网上下的语料训练一个nlp模型
"""

import pandas as pd
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch

def getSentencesList():
    """解析语料csv，把第一列，也就是中文句子都拿回来
    """
    result = pd.read_csv("/Users/yushichang/python/hub-kzSW/于世昌/week04/cmn_sen_db_2.tsv",
                         sep="\t", header=None, encoding="utf-8")
    print(result[1])
    return result[1]

def getVocabularyDict():
    """
    构建词表
    基于中文语料， 按字分词， 然后获取词和id的映射关系
    """
    # 先从语料中获取所有中文句子
    chinese_sentences = getSentencesList()
    word_list = []
    for sentence_item in chinese_sentences:
        for word_item in sentence_item:
            word_list.append(word_item)
    
    vocabulary_dict = {"padding": 0, "unknown": 1}
    
    for index, word_item in enumerate(set(word_list)):
        vocabulary_dict.update({word_item: index + 2})
        
    return vocabulary_dict

def count_sentence_length_distribution():
    corpus = getSentencesList()
    """
    统计句子长度分布
    :param corpus: 你的语料 list，每个元素是句子（字符串 或 token 列表）
    """
    # 定义区间
    bins = [
        (1, 10, "1~10"),
        (11, 20, "11~20"),
        (21, 30, "21~30"),
        (31, 50, "31~50"),
        (51, 100, "51~100"),
        (101, 256, "101~256"),
        (257, float('inf'), ">256"),
    ]

    # 初始化计数器
    counts = {name: 0 for _, _, name in bins}
    total = len(corpus)

    # 统计
    for sent in corpus:
        length = len(sent)
        for min_len, max_len, name in bins:
            if min_len <= length <= max_len:
                counts[name] += 1
                break

    # 打印结果
    print("=" * 50)
    print("📊 语料句子长度分布统计")
    print("=" * 50)
    for _, _, name in bins:
        cnt = counts[name]
        pct = cnt / total * 100
        print(f"{name:>8}: {cnt:>6,} 条  ({pct:>5.1f}%)")
    print("=" * 50)
    print(f"✅ 总句子数：{total:,}")
    print(f"✅ 中位数长度：10（你提供的）")
    print(f"✅ 最大长度：266（你提供的）")
    print("=" * 50)

def get_max_length():
    corpus = getSentencesList()
    """
    计算语料 list 中最长句子的长度
    :param corpus: 你的语料 list，里面每个元素是一个句子（字符串 or token 列表）
    :return: 最大长度
    """
    return max(len(sentence) for sentence in corpus)
    
def get_sentences_lenth_median():
    """
    基于中文句子list，转句子长度list
    """
    chinese_sentences_list = getSentencesList()
    chinese_sentences_len_list = [len(chinese_sentence) for chinese_sentence in chinese_sentences_list]
    
    return get_median(chinese_sentences_len_list)
    
def get_median(num_list):
    """
    计算整数列表的中位数
    :param num_list: 输入的整数列表 list[int]
    :return: 中位数（int 或 float）
    """
    # 1. 排序（必须先排序）
    sorted_list = sorted(num_list)
    # 2. 获取列表长度
    n = len(sorted_list)
    
    # 3. 计算中位数
    if n % 2 == 1:
        # 奇数长度：取中间元素
        median = sorted_list[n // 2]
    else:
        # 偶数长度：取中间两个数的平均值
        median = (sorted_list[n//2 - 1] + sorted_list[n//2]) / 2
    
    return median

import torch
import torch.nn as nn
import math

# ========================
# 🔥 位置编码 - 终极黑盒
# 你不用懂内部，直接用！
# ========================
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=23):
        super().__init__()
        # 算好位置编码（你句子最长24，我给你写死）
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # ----------------------
        # 🔥 黑盒核心用法！
        # 输入 x：字向量
        # 输出 x + 位置信息
        # ----------------------
        return x + self.pe[:, :x.size(1)]

class MyNlpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_len = len(getVocabularyDict())
        # 3880词表长度， 128维度
        dim = 128
        self.emb = nn.Embedding(self.vocab_len, dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(dim)
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, # embedding维度
            nhead=8, # 注意力层头数,需要被d_model整除
            dim_feedforward=dim*2, # ffn前馈网络层的放大倍数
            dropout=0.1, # 防止过拟合，drop系数
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(transformer_decoder_layer, num_layers=3)
        self.linear = nn.Linear(dim, self.vocab_len)
    def forward(self, input):
        # 语料的句子长度中位数是 data_length，所以input都会约定成len = data_length - 1
        # 所以输入是 (batch, data_length - 1)
        emb_output = self.emb(input) # emb_output = (batch, data_length - 1, dim)
        position_encoding_out = self.position_encoding(emb_output)
        mask = self.triu_mask(position_encoding_out.shape[1])
        transformer_out = self.transformer(
            tgt=position_encoding_out, # 入参矩阵
            memory=position_encoding_out,
            tgt_mask=mask,
            memory_mask=mask,    # 👈 就加这一行！！！
            tgt_is_causal=True
        ) # transformer_out = (batch, data_length, dim)
        
        return self.linear(transformer_out) # linear_output = (batch, data_length - 1, self.vocab_len)
    def triu_mask(self, t):
        """定义一个上三角mask矩阵，用于transformer的causal mask
        """
        # 定义一个 t,t 的全1矩阵
        ones_tensors = torch.ones(t, t)
        # 将对角线上面的上三角保留1，其余位置成0
        triu_tensors = torch.triu(ones_tensors, diagonal=1)
        # 第一个参数的含义是， 将triu_tensors转成一个 布尔的矩阵，然后 mask会把True的位置转成 float负无穷
        return triu_tensors.masked_fill(triu_tensors == 1, float('-inf'))
        
def dealDataSet():
    """处理数据集
    1、拿到中文句子的列表
    2、中文句子基于词表换词id， 找不到的则id为1=unk
    3、思考了一件事情， 因为我们训练的时候要固定数据集的长度，
    小于这个长度需要补padding ，大于这个长度需要阶段
    问了ai，补0 和 截断 哪个对 训练的结果影响更大， ai说的是截断
    但是补0也会有影响， 所以 我统计了一下 语料的长度分布
    ==================================================
    📊 语料句子长度分布统计
    ==================================================
        1~10: 34,441 条  ( 54.4%)
    11~20: 25,014 条  ( 39.5%)
    21~30:  2,827 条  (  4.5%)
    31~50:    838 条  (  1.3%)
    51~100:    201 条  (  0.3%)
    101~256:     30 条  (  0.0%)
        >256:      1 条  (  0.0%)
    ==================================================
    ✅ 总句子数：63,352
    ✅ 中位数长度：10（你提供的）
    ✅ 最大长度：266（你提供的）
    ==================================================
    基于这个 ai告诉我 固定长度定为24，这个数怎么来的我也不清楚， 先基于这个24来做，看看最终训练效果如何
    """
    # 如果我们规定句子长度是24， 那么数据就是24 - 1， 因为要考虑到训练的时候 数据集是 0 - n-1, 标签集为 1 - n
    data_lenth = 24
    # 获取句子列表
    sentenceList = getSentencesList()
    # 获取词和id的字典
    vocab_dict = getVocabularyDict()
    # [[1, 2, 3, 4], [2, 3, 4, 5]]
    sentenceList = [[vocab_dict.get(word, 1) for word in sentence[:data_lenth]] for sentence in sentenceList]
    # 句子不足data_lenth个长度， 补0 也就是 padding
    for sentenceItem in sentenceList:
        for i in range(data_lenth - len(sentenceItem)):
            sentenceItem.append(0)
    
    # 然后准备 data和label, data 0 - data_lenth , label 1 - data_lenth+1
    dataList = [sentenceItem[:data_lenth - 1] for sentenceItem in sentenceList]
    labelList = [sentenceItem[1:] for sentenceItem in sentenceList]
    tensorDataset = TensorDataset(torch.LongTensor(dataList), torch.LongTensor(labelList))
    return DataLoader(dataset=tensorDataset, batch_size=128, shuffle=True)
    
    
def main():
    print("=" * 20)
    print(count_sentence_length_distribution())
    # print(getSentencesList())
    # vocabulary_dict = getVocabularyDict();
    # print(vocabulary_dict)
    # print(len(vocabulary_dict))
    # 语料中的句子平均长度是10
    # print(get_sentences_lenth_median())
    # 总共训练500轮
    max_epoch = 3
    # 学习率0.001
    learning_rate = 0.001
    myModel = MyNlpModel()
    # 定义优化器
    adam = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
    # 定义交叉熵loss
    loss_func = nn.CrossEntropyLoss()
    
    dataLoader = dealDataSet()
    myModel.train()
    for i in range(max_epoch):
        total_loss = 0
        for data_batch, label_batch in dataLoader:
            # 梯度零化
            adam.zero_grad()
            # 模型预测
            pred = myModel(data_batch) # batch, data_length - 1, vocab_size
            # 计算梯度
            pred_flat = pred.view(pred.shape[0] * pred.shape[1], pred.shape[2])
            label_flat = label_batch.view(label_batch.shape[0] * label_batch.shape[1])
            loss = loss_func(pred_flat, label_flat)
            loss.backward()
            adam.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataLoader)
        print(f"Epoch {i+1}/{max_epoch}  |  Loss: {avg_loss:.4f}")
    # 👇 就加在这里
    # 保存模型和词表
    torch.save(myModel.state_dict(), "my_nlp_model.pth")
    torch.save(getVocabularyDict(), "vocab_dict.pth")
    print("✅ 模型已保存到本地！")

    return myModel, getVocabularyDict()


if __name__ == "__main__":
    main()
    