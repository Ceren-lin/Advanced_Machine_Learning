import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset import CustomDataset
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import copy


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, value, key, query):
        # Split the embedding into `self.heads` pieces
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        # Split the embedding into `self.heads` parts
        values = value.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Add skip connection, followed by normalization
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        # Add skip connection, followed by normalization
        out = self.dropout(self.norm2(forward + x))
        return out


class ClassifierWithSelfAttention(nn.Module):
    def __init__(self, sequence_length, embed_size, num_classes, heads, forward_expansion, dropout):
        super(ClassifierWithSelfAttention, self).__init__()
        self.linear = nn.Linear(sequence_length, sequence_length * embed_size)  # 新增线性层
        self.encoder = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(embed_size, num_classes)
        self.embed_size=embed_size

    def forward(self, x):
        # x: [batch_size, sequence_length]
        x = self.linear(x)  # [batch_size, sequence_length * embed_size]
        x = x.view(x.size(0), -1, self.embed_size)  # 重塑为[batch_size, sequence_length, embed_size]
        x = self.encoder(x, x, x)
        x = self.pool(x.permute(0, 2, 1)).squeeze(2)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device):
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

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / len(data_loader.dataset)
    return accuracy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = "E:\\毕设\\电路仿真\\仿真结果\\four_op_amp\\小波包分解1"  # 更新为你的路径
    dataset = CustomDataset(root_dir=root_dir)
    num_classes = len(dataset.classes)
    # 假定embed_size与数据集中序列的维度相匹配
    model = ClassifierWithSelfAttention(sequence_length=63,embed_size=128, num_classes=num_classes, heads=8, forward_expansion=4, dropout=0.1).to(
        device)

    # 拆分数据集为训练集、验证集和测试集
    train_size = int(len(dataset) * 0.6)
    valid_size = int(len(dataset) * 0.2)
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        valid_accuracy = evaluate_model(model, valid_loader, device)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}')

    test_accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
