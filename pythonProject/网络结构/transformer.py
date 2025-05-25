import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import NormalDataset,CustomDataset
import copy
from torch.nn.utils.rnn import pad_sequence
import math

# 定义一个自定义的collate_fn来处理不同长度的序列
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    # 保证每个序列都有一个特征维度
    x_pad = pad_sequence([x.view(-1, 1).float() for x in xx], batch_first=True, padding_value=0)
    y_tensor = torch.tensor(yy, dtype=torch.long)
    return x_pad, y_tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout_rate=0.1, max_seq_length=200):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout_rate, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.hidden_dim)
        src = src.permute(1, 0, 2)  # Adjust shape to (seq_length, batch_size, hidden_dim)
        src = self.pos_encoder(src)
        encoder_output = self.transformer_encoder(src)

        # In a typical translation task, the target would be the actual sequence to decode.
        # Here, we simulate a decoding step using encoder_output as both src and tgt for simplification.
        # This is not a typical use case for decoders in classification tasks.
        decoder_output = self.transformer_decoder(encoder_output, encoder_output)

        decoder_output = decoder_output.permute(1, 0, 2)  # Adjust shape back to (batch_size, seq_length, hidden_dim)
        output = self.output_layer(decoder_output[:, -1, :])  # Use the last time step's output for classification
        return output


def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
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

        # Evaluate the model on validation set
        val_acc = evaluate_model(model, valid_loader, criterion)
        # Update best model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f'New best model found at epoch {epoch + 1}, Validation Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def evaluate_model(model, valid_loader, criterion):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    total_acc = running_corrects.double() / len(valid_loader.dataset)
    print(f'Validation Accuracy: {total_acc:.4f}')
    return total_acc


def tt_model(model, test_loader):
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
    num_layers = 1  # Transformer编码器层数
    num_heads = 4  # 多头注意力头数
    #num_classes = 10  # 分类任务的类别数
    dropout_rate = 0.1  # Dropout率
    batch_size = 8
    num_epochs = 1000
    learning_rate = 0.5

    # 数据加载和预处理
    root_dir = "E:\\毕设\\电路仿真\\仿真结果\\sallen_key\\原始数据文件"  # Update to your path
    dataset = NormalDataset(root_dir=root_dir)
    #num_classes = len(dataset.classes)
    num_classes=dataset.get_num_classes()
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
    # 在DataLoader中使用自定义的collate_fn
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerClassifier(input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 省略了训练循环和评估代码，与之前类似，只是模型的调用方式稍有不同。
    # 下面的代码继续执行训练和评估流程
    best_model = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs)

    # 在测试集上评估最佳模型
    tt_model(best_model, test_loader)

    # 可以选择将最佳模型保存到文件
    torch.save(best_model.state_dict(), 'best_model.pth')