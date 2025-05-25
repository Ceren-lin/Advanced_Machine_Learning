import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
import scipy.signal
from scipy.signal.windows import gaussian
# TODO: 错误的数据加载
class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []
        self.classes = os.listdir(root_dir)
        max_length = 0

        # 首先遍历一次文件以确定最大的特征长度
        for _class in self.classes:
            class_dir = os.path.join(root_dir, _class)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                df = pd.read_csv(file_path, header=None)
                if len(df.columns) - 1 > max_length:  # 减去1是因为我们跳过第一列（特征名）
                    max_length = len(df.columns) - 1

        # 再次遍历文件，读取特征并进行处理
        for label, _class in enumerate(self.classes):
            class_dir = os.path.join(root_dir, _class)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                df = pd.read_csv(file_path, header=None)
                if df.shape[0] > 1:  # 确保文件至少有两行
                    features = df.iloc[1, 1:].values.astype(np.float32)  # 跳过第一列（特征名）
                    # 如果特征长度小于max_length，则进行填充
                    if len(features) < max_length:
                        features = np.pad(features, (0, max_length - len(features)), 'constant', constant_values=0)
                    self.samples.append(features)
                    self.labels.append(label)

        # 转换为Tensor
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]



import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch

# 用于读取原始数据
class NormalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the csv files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(root_dir)
        self.samples = self._make_dataset()
        self.num_classes = len(self.classes)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir() and not d.name.startswith('.')]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self):
        samples = []
        for target in self.classes:
            d = os.path.join(self.root_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_csv_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, self.class_to_idx[target])
                        samples.append(item)
        return samples

    def is_csv_file(self, filename):
        return filename.endswith('.csv')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self._load_csv_data(path)
        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def _load_csv_data(self, path):
        # 根据需要修改这个方法来适应数据的具体格式
        data_frame = pd.read_csv(path, header=None, skiprows=1)  # 跳过首行
        data = data_frame.values[:, 1].astype('float32')  # 取第二列的值
        # 如果数据长度大于95，截断它；如果小于95，用0填充它
        if len(data) > 95:
            data = data[:95]  # 截断多余的部分
        elif len(data) < 95:
            data = np.pad(data, (0, 95 - len(data)), 'constant', constant_values=0)  # 用0填充
        return torch.tensor(data)

    def get_num_classes(self):
        """返回数据集中的类别数"""
        return len(self.classes)



def moving_average(data, window_size):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def smooth_waveform(waveform, window_size=5):
    """简单的滑动平均平滑"""
    if window_size < 2:  # 如果窗口大小小于2，则不进行平滑
        return waveform
    # 构造一个滑动窗口的卷积核，使用均匀权重
    kernel = np.ones(window_size) / window_size
    waveform_smooth = np.convolve(waveform, kernel, mode='same')
    return waveform_smooth


def gaussian_smooth_waveform(waveform, sigma=2):
    """使用高斯滤波进行波形平滑"""
    window_size = int(sigma * 4) + 1  # 确保窗口大小是奇数
    kernel = gaussian(window_size, std=sigma)
    kernel = kernel / np.sum(kernel)
    waveform_smooth = np.convolve(waveform, kernel, mode='same')
    return waveform_smooth

# def plot_and_save_sample_waveforms(dataset, save_path, num_samples=1, smooth=False, window_size=5):
#     plt.figure(figsize=(14, 10))
#
#     colors = plt.cm.jet(np.linspace(0, 1, len(dataset.classes)))
#     line_styles = ['-', '--', '-.', ':']
#     styles = [(color, ls) for ls in line_styles for color in colors]
#
#     for i, cls in enumerate(dataset.classes):
#         indices = [idx for idx, (_, label) in enumerate(dataset.samples) if label == dataset.class_to_idx[cls]]
#         random_idx = np.random.choice(indices, size=min(num_samples, len(indices)), replace=False)
#
#         for idx in random_idx:
#             sample, _ = dataset[idx]
#             waveform = sample.numpy() if torch.is_tensor(sample) else sample
#             if smooth:
#                 waveform = gaussian_smooth_waveform(waveform)
#             plt.plot(waveform, label=f'Class {cls}', color=styles[i][0], linestyle=styles[i][1])
#
#     #plt.title('Sample Waveforms by Class')
#     #plt.xlabel('Sample Points')
#     #plt.ylabel('Amplitude')
#     plt.legend(loc='upper right')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
#
#     # 保存图像到指定路径
#     plt.savefig(save_path)
#     plt.show()




if __name__ == "__main__":
    # 示例使用
    # root_dir = "E:\\毕设\\电路仿真\\仿真结果\\chebylm358\\小波包分解1"  # 更新为你的路径
    # dataset = CustomDataset(root_dir=root_dir)
    #
    # # 简单验证
    # print(f"Total samples: {len(dataset)}")
    # sample, label = dataset[0]
    # print(f"Sample shape: {sample.shape}, Label: {label}")
    #
    # # 创建DataLoader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #
    # #通过迭代DataLoader来获取一批数据
    # for samples, labels in dataloader:
    #     print(f"Batch samples shape: {samples.shape}")  # 打印批样本的形状
    #     print(f"Batch labels: {labels}")  # 打印批标签
    #     break  # 只迭代一次，即只获取一批数据
    #
    # # 进一步验证，打印出第一个样本的特征值以确保数据被正确读取
    # first_sample_features, first_sample_label = next(iter(dataset))
    # print(f"First sample features: {first_sample_features}")
    # print(f"First sample label: {first_sample_label}")
    #
    # # 计算并打印每个类别的样本数量
    # label_counts = torch.zeros(len(dataset.classes))
    # for _, label in dataset:
    #     label_counts[label] += 1
    #
    # for i, count in enumerate(label_counts):
    #     print(f"Class {dataset.classes[i]} has {int(count)} samples.")
    #
    # import matplotlib.pyplot as plt

    # 假设你的CustomDataset类已经定义好了
    root_dir = "E:\\毕设\\电路仿真\\仿真结果\\four_op_amp\\原始数据文件"

    # 实例化你的数据集，这里假设你的数据集存储在"your_dataset_directory"目录下
    dataset = NormalDataset(root_dir=root_dir)

   # plot_and_save_sample_waveforms(dataset, 'E:\\毕设\\电路仿真\\仿真结果\\chebylm358\\pic.png', smooth=True, window_size=5)


    # 选择要展示的样本数量
    num_samples_to_display = 2400

    for i in range(num_samples_to_display):
        # 获取单个样本
        sample, label = dataset[i]

        #print(sample)
        if sample.size != [95]:
            print(sample.size())

        # 打印标签和样本的形状
        # print(f"Sample #{i} - Class Label: {label}")
        # print(f"Sample Shape: {sample.shape}")
        #
        # # 绘制样本数据
        # plt.figure(figsize=(10, 2))
        # plt.plot(sample.numpy())  # 如果你的样本是Tensor类型，需要转换为NumPy数组
        # plt.title(f"Sample #{i} - Class Label: {label}")
        # plt.xlabel('Time Steps')
        # plt.ylabel('(V(OUT))@1')
        # plt.grid(True)
        # plt.show()

    # 假设dataset是我们已经创建好的CustomDataset实例
