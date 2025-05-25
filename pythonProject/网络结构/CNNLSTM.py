import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import NormalDataset  # 确保你有这个文件和CustomDataset类
import matplotlib.pyplot as plt
import time
import pywt
import numpy as np
from torch.nn.utils.rnn import pad_sequence


def custom_collate_fn(batch):
    """
    batch中的每个元素形如(data, label)，其中data是张量，label是整数。
    因为数据可能具有不同的长度，我们将对数据进行填充。
    """
    # 分离数据和标签
    data, labels = zip(*batch)

    # 填充数据
    data = pad_sequence(data, batch_first=True, padding_value=0)

    # 将标签转换为张量
    labels = torch.tensor(labels, dtype=torch.long)

    return data, labels

class CrossWaveletTransform(nn.Module):
    def __init__(self):
        super(CrossWaveletTransform, self).__init__()

    def forward(self, x):
        # 假设 x 是一个形状为 (batch_size, channels, signal_length) 的PyTorch张量
        # 将PyTorch张量转换为NumPy数组以进行小波变换
        x_np = x.numpy()

        # 应用交叉小波变换，这里只是一个示例，具体实现应根据实际需要进行调整
        # 注意：这里需要自己实现交叉小波变换或者找到适合的函数来处理
        cwt_output = np.array([pywt.cwt(signal, scales=np.arange(1, 129), wavelet='cmor')[0] for signal in x_np])

        # 将结果转换回PyTorch张量
        cwt_output_torch = torch.from_numpy(cwt_output).float()  # 确保数据类型匹配
        return cwt_output_torch


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet1d18(num_classes):
    return ResNet1D(ResBlock, [2, 2, 2, 2], num_classes)


class ResNet1DLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden_size=128, lstm_layers=1):
        super(ResNet1DLSTM, self).__init__()
        # 使用resnet1d18作为特征提取器
        self.resnet1d = resnet1d18(num_classes=num_classes)
        # 移除resnet的全连接层
        self.resnet1d.fc = nn.Identity()

        # LSTM配置
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        # LSTM层，假设ResNet的输出特征大小为512
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_size,
                            num_layers=lstm_layers, batch_first=True)
        # 全连接层，用于分类
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        # 假设x的形状为(batch_size, sequence_length, C, L)，其中
        # sequence_length是序列长度，C是通道数，L是信号长度
        batch_size, sequence_length, C, L = x.size()
        # 将x调整为(resnet需要的形状(batch_size*sequence_length, C, L)
        x = x.view(batch_size * sequence_length, C, L)
        # 通过ResNet1D提取特征
        x = self.resnet1d(x)
        # 将特征调整回(batch_size, sequence_length, feature_size)
        x = x.view(batch_size, sequence_length, -1)
        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 取LSTM最后一个时间步的输出
        x = lstm_out[:, -1, :]
        # 全连接层
        x = self.fc(x)
        return x

# 接下来的部分 (如数据加载、模型训练和测试) 保持不变，
# 只需确保替换模型实例化部分为新的CNNLSTM类。
# 注意根据你的数据调整LSTM和全连接层的尺寸。
def dynamic_target_adjustment(outputs, labels):
    batch_size = outputs.size(0)
    target = labels[:batch_size]
    return target

if __name__ == "__main__":
    root_dir = ".\仿真结果\\sallen_key\\原始数据文件"  # Update to your path
    dataset = NormalDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.5)  # Use 50% of data for training
    test_size = dataset_size - train_size  # Remaining 50% for testing

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(dataset.classes)
    model = ResNet1DLSTM(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    # Lists to store loss for plotting
    train_losses = []
    test_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            adjusted_labels = dynamic_target_adjustment(outputs, labels)

            loss = criterion(outputs, adjusted_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate training loss and add to list
        train_losses.append(loss.item())

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                adjusted_labels = dynamic_target_adjustment(outputs, labels)
                test_loss += criterion(outputs, adjusted_labels).item()

        # Calculate average test loss and add to list
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f}s')

    # Plotting loss curve
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    start_time = time.time()
    # Evaluate accuracy
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total test time: {total_time:.2f}s')
    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')

