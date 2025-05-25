import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import NormalDataset
import matplotlib.pyplot as plt
import time

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

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 15, 120)  # Update this based on actual output size after convolution and pooling
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        print(f"conv1 output shape: {x.size()}")
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        print(f"conv2 output shape: {x.size()}")
        x = self.pool(x)
        print(f"conv3 output shape: {x.size()}")
        x = x.view(x.size(0), -1)  # Update this based on actual output size after convolution and pooling
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

        # 根据计算调整大小：128个通道，每个通道长度为7
        #self.fc1 = nn.Linear(128 * 7, 256)
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

        # 动态计算全连接层的输入尺寸
        num_features_before_fc = torch.numel(x) // x.shape[0]  # 计算每个样本的特征数量
        if self.fc1.in_features != num_features_before_fc:
            # 如果实际尺寸与预设不符，动态调整全连接层的输入尺寸
            self.fc1 = nn.Linear(num_features_before_fc, 256).to(x.device)

        x = x.view(-1, int(num_features_before_fc))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def dynamic_target_adjustment(outputs, labels):
    batch_size = outputs.size(0)
    target = labels[:batch_size]
    return target

if __name__ == "__main__":
    root_dir = "E:\\毕设\\电路仿真\\仿真结果\\sallen_key\\原始数据文件"  # Update to your path
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
    model = ComplexCNN1D(num_classes).to(device)
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
