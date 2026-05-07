import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader




# 生成数据集
def build_dataset():
    texts = []
    labels = []
    for i in range(5):
        text = ['a'] * 5
        text[i] = '你'
        text = ''.join(text)
        texts.append(text)
        labels.append(i)

    char_to_num = {'a': 0, '你': 1}
    encoded_texts = []
    for text in texts:
        encoded = [char_to_num[char] for char in text]
        # 增加一个维度，将数据变为三维 (batch_size, sequence_length, input_size)
        encoded = [[num] for num in encoded]
        encoded_texts.append(encoded)

    texts_tensor = torch.tensor(encoded_texts, dtype = torch.long)
    labels_tensor = torch.tensor(labels, dtype = torch.long)
    return TensorDataset(texts_tensor, labels_tensor)


# 定义简单的RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        # 修改隐藏状态的初始化，使其符合要求
        hidden = torch.zeros(1, batch_size, self.hidden_size).squeeze(0)
        out, _ = self.rnn(x, hidden.unsqueeze(0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 定义简单的LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = (torch.zeros(1, batch_size, self.hidden_size).squeeze(0),
                  torch.zeros(1, batch_size, self.hidden_size).squeeze(0))
        out, _ = self.lstm(x, (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0)))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # 将输入数据转换为float32类型
            inputs = inputs.to(torch.float32)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


# 评估模型
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # 将输入数据转换为float32类型
            inputs = inputs.to(torch.float32)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network: {100 * correct / total}%')


def main():
    input_size = 1
    hidden_size = 4
    output_size = 5
    batch_size = 5
    epochs = 200

    dataset = build_dataset()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    rnn_model = SimpleRNN(input_size, hidden_size, output_size)
    lstm_model = SimpleLSTM(input_size, hidden_size, output_size)

    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.SGD(rnn_model.parameters(), lr=0.01)
    lstm_optimizer = optim.SGD(lstm_model.parameters(), lr=0.01)

    print("Training RNN model...")
    train_model(rnn_model, train_loader, criterion, rnn_optimizer, epochs)
    print("Evaluating RNN model...")
    evaluate_model(rnn_model, train_loader)

    print("Training LSTM model...")
    train_model(lstm_model, train_loader, criterion, lstm_optimizer, epochs)
    print("Evaluating LSTM model...")
    evaluate_model(lstm_model, train_loader)


if __name__ == "__main__":
    main()
