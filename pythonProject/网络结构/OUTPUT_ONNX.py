import torch
import torch.onnx
from CNNGRU import GRUCNNNet

# 根据实际情况设置
seq_length = 63
batch_size = 1
num_classes = 13  # 根据您的datasetName变量选择

model = GRUCNNNet(input_size=1, hidden_size=32, num_layers=2, num_classes=num_classes)
model.load_state_dict(torch.load("CNNGRU_best_model.pth"))
model.eval()

# 调整虚拟输入的形状以匹配模型的期望输入形状
# 注意，此处的形状为(batch_size, 1, seq_length)，匹配Conv1d层的期望输入
dummy_input = torch.randn(batch_size, 1, seq_length,requires_grad=True)

# 如果模型的GRU层需要初始状态h0，则创建一个对应的虚拟h0
dummy_h0 = torch.zeros(model.num_layers, batch_size, model.hidden_size)

# 使用torch.onnx.export导出模型，包括h0作为模型的一个输入
torch.onnx.export(model,
                  dummy_input,  # 现在只需要传入dummy_input
                  "model.onnx",
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],  # 'h0'已从这里移除，因为我们不再将其作为必需的输入
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})