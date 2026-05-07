import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


#生成单个样本数据
def create_single_sample():
    sample = np.random.rand(5) #生成一个包含5个随机浮点数的一维数组作为样本数据
    label = np.argmax(sample) #找出数组中最大值的索引作为样本的类别标签
    return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)

#生成数据集
def generate_dataset(sample_count):
    data_x = [] #存储样本数据
    data_y = [] #存储对应的标签
    #循环生成样本和标签
    for _ in range(sample_count):
        x,y = create_single_sample()
        data_x.append(x)
        data_y.append(y)
    return torch.stack(data_x), torch.stack(data_y)

#定义多分类模型
class ClassificationModel(nn.module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ClassificationModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forword(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return nn.functional.log_softmax(x, dim=1) #应用 log_softmax 函数将输出转换为对数概率分布作为模型的预测结果
    
#负责训练模型，接受模型、训练数据、训练标签、训练轮数 epochs、批次大小 batch_size 和学习率 learning_rate 作为参数
def train_model(model, train_x, train_y, epochs, batch_size, learning_rate):
    criterion = nn.NLLLoss() #定义损失函数
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate) #选择RMSprop作为优化器
    #存储每一轮训练的平均损失和准确率
    train_losses = []
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0 #累计每个 epoch 内的损失
        for i in range (0, len(train_x), batch_size):
            #从训练数据和标签中取出一个批次的数据
            batch_x = train_x[i:i + batch_size]
            batch_y = train_y[i:i + batch_size]

            optimizer.zero_grad() #梯度清0
            outputs = model(batch_x) #进行前向传播，得到预测结果
            loss = criterion(outputs, batch_y) #计算预测结果和真实标签之间的损失
            loss.backward() #进行反向传播，计算损失关于模型参数的梯度
            optimizer.step() #根据梯度更新模型参数
            running_loss += loss.item() #累加当前批次损失

        avg_loss = running_loss / (len(train_x) // batch_size)
        train_losses.append(avg_loss)
        accuracy = evaluate_model(model, train_x, train_y)
        accuracies.append(accuracy)

        print(f'轮数 {epoch + 1}/{epochs}, 平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}')

    return train_losses, accuracies

#评估模型在给定数据上的准确率
def evaluate_model(model, test_x, test_y):
    model.eval()
    #记录正确预测的样本数量和总样本数量
    correct = 0
    total = 0
    with torch.no_grad(): #不计算梯度，无需更新模型参数
        outputs = model(test_x)
        _, predicted = torch.max(outputs.data, 1) #到预测结果中概率最大的类别索引
        total = test_y.size(0)
        correct = (predicted == test_y).sum().item()

    return correct / total

#绘制训练过程中的损失曲线和准确率曲线        
def plot_training_curves(train_losses, accuracies):
    epochs = range(1, len(train_losses) + 1) #创建一个表示训练轮数的范围对象，从 1 到训练轮数的总数

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss') #第一个子图中绘制训练损失随训练轮数的变化曲线，并为曲线添加标签 Training Loss
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Training Accuracy') #第二个子图中绘制训练准确率随训练轮数的变化曲线，并添加标签 Training Accuracy
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()    

#使用训练好的模型对新的输入数据进行预测    
def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        outputs = model(input_data)
        _, predicted = torch.max(outputs.data, 1)
        #循环遍历输入数据，打印每个输入数据及其预测类别
        for i in range(len(input_data)):
            print(f'Input: {input_data[i]}, Predicted Class: {predicted[i]}')        
        

def main():
    input_dim = 5
    hidden_dim = 15
    output_dim = 5
    epochs = 30
    batch_size = 32
    learning_rate = 0.001

    train_x, train_y = generate_dataset(1000)
    model = ClassificationModel(input_dim, hidden_dim, output_dim)

    train_losses, accuracies = train_model(model, train_x, train_y, epochs, batch_size, learning_rate)

    plot_training_curves(train_losses, accuracies)

    #测试向量 test_vec，并调用 predict 函数使用训练好的模型对其进行预测
    test_vec = torch.FloatTensor([
        [0.235, 0.4345, 0.13334, 0.34667, 0.03],
        [0.05345, 0.153, 0.66, 0.1345, 0.1],
        [0.1, 0.25435, 0.3889, 0.422, 0.0]
    ])
    predict(model, test_vec)        
    
    
if __name__ == "__main__":
    main()
