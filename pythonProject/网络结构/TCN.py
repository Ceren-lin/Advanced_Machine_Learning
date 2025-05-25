import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CustomDataset
from sklearn.metrics import accuracy_score

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return self.relu(out + residual)

class TCN(nn.Module):
    def __init__(self, input_size, num_classes, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        y = self.tcn(x)
        o = self.linear(y[:, :, -1])
        return F.log_softmax(o, dim=1)

def dynamic_target_adjustment(outputs, labels):
    batch_size = outputs.size(0)
    target = labels[:batch_size]
    return target

# 数据加载和模型训练的代码保持不变
if __name__ == "__main__":
    root_dir = "E:\\毕设\\电路仿真\\仿真结果\\four_op_amp\\小波包分解1"  # 更新为你的路径
    dataset = CustomDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.5)  # 使用50%的数据作为训练集
    test_size = dataset_size - train_size  # 剩下的50%作为测试集

    # 随机拆分数据集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建训练和测试的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)

    input_size = 1  # Adjust according to your dataset

    # Define the architecture of your TCN
    num_channels = [16, 32, 64]  # Example architecture

    model = TCN(input_size, num_classes, num_channels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    model.train()  # 确保模型处于训练模式
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)

            adjusted_labels = dynamic_target_adjustment(outputs, labels)

            if outputs.size(0) != adjusted_labels.size(0):
                continue
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 在所有训练完成后进行测试
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test set: {100 * correct / total}%')