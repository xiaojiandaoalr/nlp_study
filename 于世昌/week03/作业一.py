"""
三周作业：
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。如果不知道怎么设计，可以选择如下任务:
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

思路：
因为涉及中文文本，所以需要引入embedding层
分类任务， loss用交叉熵
分类任务， 最终需要生成一个向量，所以最后需要用linear层来降维到1维向量
本次简单一些不进行分词， 直接把每个字一个id组成词表
1、先通过ai生成一段包含 "你" 字 的 文章，200字左右
2、然后把文章每个字一行，准备词表， 0位置padding  ， 1位置unk
2.5、准备数据集，比如句子长度不超过5，每个句子只能出现一次 "你"， 你出现在哪， 对应的向量哪个位置为1， 独热
先准备100个这种数据，
然后通过词表 对数据集进行预处理，长度约定为5，小于5的补padding
3、创建embedding， 词表大小 = 2的词表长度， 维度先用64
4、创建rnn 维度128， hidden_size 64 ， num_layers
5、rnn走完走池化
6、池化成1维向量后，走linear输出一个 预测的分类结果
"""

import torch
from torch import nn
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader
import json


class MyChineseClassificationModule(torch.nn.Module):
    """对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
    """
    def __init__(self, vocabulary_size):
        super(MyChineseClassificationModule, self).__init__()
        dim = 64
        self.emb = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=dim)
        self.rnn = nn.RNN(input_size=dim, hidden_size=64, num_layers=1, batch_first=True)
        # self.pool = nn.AdaptiveAvgPool2d((1, 64))
        self.linear = nn.Linear(64, 5)

    def forward(self, input):
         emb_output = self.emb(input) # batch, 5, 64
         rnn_output, _ = self.rnn(emb_output) # batch, 5, 64
        #  pool_output = self.pool(rnn_output)
        #  flat = pool_output.squeeze(1)
         rnn_last_step_output = rnn_output[:, -1, :] # batch, 64
         return self.linear(rnn_last_step_output)

def prepare_vocabulary():
    """准备词表
    """
    # 句子中包含 "你"
    chinese_sentence = "我一人口手足耳目头发心肝脾肺肾胃肠皮肉血骨筋天地日月星云风雨雷电山水火木金土石田河海江湖波浪泉溪雾霜雪冰春夏秋冬昼夜早晚晨昏东西南北中前后左右上下内外高低长短大小多少远近深浅宽窄厚薄黑白红黄蓝绿青紫灰粉一二三四五六七八九十百千万亿零两半整我你他她它们咱吾尔彼这那此谁何啥每各某的地得之于在向往至到从自由沿顺逆随伴同共和与及或且并但却虽然因为所以故则就才又再很更也已必刚正便即都只就别没去来过回出进开开关闭起落坐立行走跑跳吃喝穿住用看听说读写想记念思考感觉爱恨喜怒哀惧好恶欲望心意神情态度言行举止家国城乡街巷楼房屋院门窗桌椅床柜灯镜纸笔书本字画歌舞影戏琴棋书画车船飞机枪炮钱票券卡软硬轻重快慢冷热干湿亮暗动静强弱好坏优劣美丑真假虚实有无是非对错是"
    vocabulary_list = ["padding", "unknown"] + list(set(chinese_sentence))
    vocabulary_dict = {word:index for index, word in enumerate(vocabulary_list)}
    return vocabulary_dict

def prepare_dataset(file_name):
    """
        读取当前目录下的dataset.csv
        sentence,label,vector
        你笑,1,"[1,0,0,0,0]"
        获取数据集
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接csv文件路径
    csv_path = os.path.join(current_dir, file_name)
    df = pd.read_csv(csv_path, sep=",")
    return list(df["sentence"]), [json.loads(item) for item in df["vector"]]

def deal_dataset(data_list, vocabulary_dict):
    """处理数据集
    """
    data_list_deal = list()
    for data_item in data_list:
        data_vector = list()
        for data_item_word in data_item:
            if data_item_word in vocabulary_dict:
                data_vector.append(vocabulary_dict[data_item_word])
            else:
                # 如果没找到的话补unk
                data_vector.append(1)
        for i in range(5 - len(data_item)):
            # 不足5，补padding
            data_vector.append(0)
        # 将处理后的数据集放到最终交给模型训练的data_list_deal中
        data_list_deal.append(data_vector[:5])
    return data_list_deal

def test_model(model):
    # 测试集的也需要用词表这一套
    # 词表
    vocabulary_dict = prepare_vocabulary()
    print(f"词表:{vocabulary_dict}")
    # 数据集 和 数据集标注
    data_list, data_label_list = prepare_dataset("testset.csv")
    # 通过词表，处理数据集，以5维向量为基本单元，少的补padding
    data_list_deal = deal_dataset(data_list, vocabulary_dict)
    # 将数据集 和 标注 转批量向量
    tensor_data_set = TensorDataset(torch.LongTensor(data_list_deal), torch.LongTensor(data_label_list))
    test_dataLoader = DataLoader(tensor_data_set, batch_size=5, shuffle=True)
    # 开启评估模式
    model.eval()
    # 2. 关闭梯度计算（验证必须加！）
    with torch.no_grad():
        total_correct = 0  # 预测对的数量
        total_num = 0      # 总数量

        # 循环测试集
        for batch_x, batch_y in test_dataLoader:
            # 前向预测
            outputs = model(batch_x)

            # 3. 批量取预测类别（取最大概率的下标）[4,4,4,4,4,4,4,4,4,4]
            predict_labels = torch.argmax(outputs, dim=1)

            # 4. 统计正确个数
            #
            total_correct += (predict_labels == torch.argmax(batch_y, dim=1)).sum().item()
            total_num += len(batch_y)

        # 5. 计算准确率
        acc = total_correct / total_num
        print(f"测试集准确率：{acc:.4f}")
        return acc

def main():
    # print(f"词表长度：{len(prepare_vocabulary())}")
    # 词表
    vocabulary_dict = prepare_vocabulary()
    print(f"词表:{vocabulary_dict}")
    # 数据集 和 数据集标注
    data_list, data_label_list = prepare_dataset("dataset.csv")
    # 通过词表，处理数据集，以5维向量为基本单元，少的补padding
    data_list_deal = deal_dataset(data_list, vocabulary_dict)
    # print(f"处理后的数据集为:{data_list_deal}")

    # 将数据集 和 标注 转批量向量
    tensor_data_set = TensorDataset(torch.LongTensor(data_list_deal), torch.LongTensor(data_label_list))
    dataLoader = DataLoader(tensor_data_set, batch_size=5, shuffle=True)

    # 创建模型
    myChineseClassificationModule = MyChineseClassificationModule(len(vocabulary_dict))
    # 分类任务损失函数用交叉熵
    loss_function = nn.CrossEntropyLoss()
    # 优化器用adam
    optimizer = torch.optim.Adam(myChineseClassificationModule.parameters(), lr=0.001)
    max_epoch = 500
    for epoch in range(max_epoch):
        for data_batch, label_batch in dataLoader:
            # 清一下步长
            optimizer.zero_grad()
            y_pred_batch = myChineseClassificationModule(data_batch)
            loss = loss_function(y_pred_batch, torch.argmax(label_batch, dim=1))
            loss.backward()
            optimizer.step()

    # 测试模型
    result = myChineseClassificationModule(torch.LongTensor([[51, 0, 0, 0, 0]]))
    print(result)
    test_model(myChineseClassificationModule)



if __name__ == "__main__":
    main()