import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from dataset import CustomDataset
from torch.utils.data import DataLoader

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        # 假设输入特征的数量是63，这里保持卷积层和池化层的定义不变
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 根据卷积层输出的实际平铺特征数量更新fc1层的输入尺寸
        self.fc1 = nn.Linear(32 * 15, 120)  # 假设经过两次池化后的特征长度为15
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加通道维度
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 15)  # 动态计算平铺尺寸
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ComplexCNN1D(nn.Module):
    def __init__(self, num_classes):
        super(ComplexCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.attention = SelfAttention(in_channels=128)  # 添加自注意力层

        self.fc1 = nn.Linear(128 * 7, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = self.attention(x)  # 应用自注意力机制

        x = x.view(-1, 128 * 7)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # Query 的线性转换
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        # Key 的线性转换
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        # Value 的线性转换
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        # 注意力得分的缩放因子
        self.scale = (in_channels // 8) ** -0.5
        # 输出的线性转换
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, length = x.size()
        query = self.query_conv(x).view(batch_size, -1, length).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, length)
        value = self.value_conv(x).view(batch_size, -1, length)

        # 计算注意力得分
        attention = torch.bmm(query, key) * self.scale
        attention = F.softmax(attention, dim=-1)

        # 应用注意力权重到 Value 上
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, length)

        # 残差连接和缩放
        out = self.gamma * out + x
        return out


def dynamic_target_adjustment(outputs, labels):
    batch_size = outputs.size(0)
    target = labels[:batch_size]
    return target




if __name__ == "__main__":
    root_dir = ".\仿真结果\\four_op_amp\\小波包分解1"  # 更新为你的路径
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
    model = ComplexCNN1D(num_classes).to(device)
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

