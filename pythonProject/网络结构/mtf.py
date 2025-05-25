import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import NormalDataset

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout_rate=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, src):
        # src shape: (seq_length, batch_size, input_dim)
        src = self.embedding(src)  # (seq_length, batch_size, hidden_dim)
        src = src.permute(1, 0, 2)  # Transformer expects (batch_size, seq_length, hidden_dim)
        transformer_output = self.transformer_encoder(src)
        # 只选择序列的最后一个元素进行分类
        output = self.output_layer(transformer_output[:, -1, :])
        return output


# 继续上面的代码

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')


def evaluate_model(model, valid_loader, criterion):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(valid_loader.dataset)
    total_acc = running_corrects.double() / len(valid_loader.dataset)

    print(f'Validation - Loss: {total_loss:.4f}, Acc: {total_acc:.4f}')


def test_model(model, test_loader):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    total_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {total_acc:.4f}')




if __name__ == "__main__":
    # 定义一些超参数
    input_dim = 1  # 每个时间步的特征维度
    hidden_dim = 64  # Transformer内部的特征维度
    num_layers = 2  # Transformer编码器层数
    num_heads = 4  # 多头注意力头数
    num_classes = 10  # 分类任务的类别数
    dropout_rate = 0.1  # Dropout率
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001

    # 数据加载和预处理
    root_dir = "E:\\毕设\\电路仿真\\仿真结果\\sallen_key\\原始数据文件"  # Update to your path
    dataset = NormalDataset(root_dir=root_dir)
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerClassifier(input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 省略了训练循环和评估代码，与之前类似，只是模型的调用方式稍有不同。
    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 在验证集上评估模型
    evaluate_model(model, valid_loader, criterion)

    # 在测试集上评估模型
    test_model(model, test_loader)