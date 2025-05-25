import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from dataset import CustomDataset,NormalDataset
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_channels, in_channels // 8)
        self.key = nn.Linear(in_channels, in_channels // 8)
        self.value = nn.Linear(in_channels, in_channels)
        self.scale = (in_channels // 8) ** -0.5

    def forward(self, x):
        batch_size, length, channels = x.size()
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention = torch.bmm(query, key.transpose(1, 2)) * self.scale
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(attention, value)

        return out


class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x).transpose(-2, -1)
        v = self.value(x)

        attn_weights = self.softmax(torch.bmm(q, k) / (x.size(-1) ** 0.5))
        attn_output = torch.bmm(attn_weights, v)

        return attn_output


class CNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # 确保x的形状为(batch_size, channels, seq_length)
        #print('x',x.size())
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class GRUCNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUCNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 假设input_size是每个时间步的特征数
        self.cnn = CNNLayer(1, 32, kernel_size=3, stride=1, padding=1)  # in_channels设置为1
        self.gru = nn.GRU(32, hidden_size, num_layers, batch_first=True)
        self.self_attention = SelfAttention(hidden_size)
        self.cross_attention = CrossAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h0=None):
        # 如果x的形状是(batch_size, seq_length)，我们需要在seq_length维度之后添加一个特征维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 将形状变为(batch_size, seq_length, 1)

        #x = x.permute(0, 2, 1)  # 调整为(batch_size, 1, seq_length)以适配Conv1d的期望输入
        #print('x:',x.size())

        x = self.cnn(x)
        # 如果你在CNN后做了池化或其他维度变换，请相应调整x的形状以匹配GRU的期望输入
        x = x.permute(0, 2, 1)  # 可能需要根据CNN输出调整维度

        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.gru(x, h0)

        out = self.self_attention(out)  # 先应用自注意力
        out = self.cross_attention(out)  # 然后应用交叉注意力

        out = self.fc(out[:, -1, :])  # 选择最后一个时间步的输出进行分类
        return out

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
    root_dir = ".\仿真结果\\sallen_key\\原始数据文件"  # 更新为你的路径
    dataset = NormalDataset(root_dir=root_dir)

    # 拆分数据集为训练集、验证集和测试集
    train_size = int(len(dataset) * 0.6)
    valid_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUCNNNet(input_size=1, hidden_size=16, num_layers=2, num_classes=len(dataset.classes)).to(device)
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