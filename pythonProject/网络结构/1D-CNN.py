import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import CustomDataset,NormalDataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import copy

class CNN1D(nn.Module):
    def __init__(self, num_classes):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 15, 120)  # Update this based on actual output size after convolution and pooling
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        print(x.size())
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        #print(f"conv1 output shape: {x.size()}")
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        #print(f"conv2 output shape: {x.size()}")
        x = self.pool(x)
        #print(f"conv3 output shape: {x.size()}")
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
        self.fc1 = nn.Linear(128 * 15, 256) #nol-four
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # print(x.size())
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        #print(f"conv1 output shape: {x.size()}")
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(f"conv2 output shape: {x.size()}")
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        #print(f"conv3 output shape: {x.size()}")
        x = self.pool(x)
        #print(f"pool output shape: {x.size()}")
        # 更新视图大小为：128个通道，每个通道长度为7
        x = x.view(-1, 128 * 15)
        #print(f"x output shape: {x.size()}")
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def dynamic_target_adjustment(outputs, labels):
    batch_size = outputs.size(0)
    target = labels[:batch_size]
    return target

def evaluate_model(model, data_loader, device):
    model.eval()  # 将模型设置为评估模式
    true_labels = []
    pred_labels = []

    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predictions.cpu().numpy())

    # 计算评价指标
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    confusion = confusion_matrix(true_labels, pred_labels)

    return precision, recall, f1, confusion


if __name__ == "__main__":
    root_dir = "E:\\毕设\\电路仿真\\仿真结果\\sallen_key\\小波包分解1"  # 更新为你的路径
    dataset = CustomDataset(root_dir=root_dir)

    # 拆分数据集为训练集、验证集和测试集
    train_size = int(len(dataset) * 0.6)
    valid_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexCNN1D(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 20
    best_acc = 0.0
    early_stopping_patience = 10
    no_improve_epochs = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    start_time = time.time()

    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # 验证集性能评估
        model.eval()
        total_loss = 0
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_valid_loss = total_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        epoch_acc = correct / total

        print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {epoch_acc:.4f}')

        # 检查是否需要更新最佳模型权重
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            print(f'New best model found at epoch {epoch + 1}, Accuracy: {best_acc:.4f}')
        else:
            no_improve_epochs += 1

        # 早停机制
        # if no_improve_epochs >= early_stopping_patience:
        #     print('Early stopping triggered.')
        #     break

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f}s')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    # 保存最佳模型
    torch.save(model.state_dict(), "CNNGRU_1_best_model_batchsize1.pth")

    # 测试最佳模型
    model.eval()
    correct = total = 0
    precision, recall, f1, confusion = evaluate_model(model, test_loader, device)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:\n', confusion)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Loss Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

