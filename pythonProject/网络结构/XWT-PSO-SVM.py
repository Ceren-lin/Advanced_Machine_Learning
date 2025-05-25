import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pyswarm import pso
from sklearn.model_selection import cross_val_score
import os

def cross_wavelet_features(signal1, signal2, wavelet_name='db1', max_level=1):
    """
    应用交叉小波变换并提取特征
    """
    coeffs1 = pywt.wavedec(signal1, wavelet_name, level=max_level)
    coeffs2 = pywt.wavedec(signal2, wavelet_name, level=max_level)
    features = []

    # 计算每一级小波变换系数的交叉小波变换
    for i in range(min(len(coeffs1), len(coeffs2))):
        cross_coeff = np.multiply(coeffs1[i], coeffs2[i])
        features.append(cross_coeff.flatten())

    return np.concatenate(features)

def svm_optimize(x, X_train, y_train):
    """
    SVM超参数优化目标函数
    """
    C, gamma = x
    model = SVC(C=C, gamma=gamma)
    scores = cross_val_score(model, X_train, y_train, cv=min(5, len(np.unique(y_train))))
    return -scores.mean()  # 返回负的平均交叉验证分数

def load_data_with_cross_wavelet(dataset_path, fixed_length=100):
    """
    加载数据并应用交叉小波变换提取特征
    """
    signals = []
    labels = []
    label_map = {}

    for label, subdir in enumerate(sorted(os.listdir(dataset_path))):
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        label_map[subdir] = label

        subdir_signals = []
        for file in sorted(os.listdir(subdir_path)):
            if file.endswith('.csv'):
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path)
                target_column = df.columns[1]  # 假设目标列总是第二列
                signal = df[target_column].values[:fixed_length]
                if len(signal) < fixed_length:
                    signal = np.pad(signal, (0, fixed_length - len(signal)), 'constant')
                subdir_signals.append(signal)

        # 交叉小波变换相邻样本
        for i in range(0, len(subdir_signals)-1, 2):
            features = cross_wavelet_features(subdir_signals[i], subdir_signals[i+1])
            signals.append(features)
            labels.append(label)

    signals = np.array(signals)
    labels = np.array(labels)

    scaler = StandardScaler()
    signals = scaler.fit_transform(signals)

    return signals, labels, label_map

# 其余部分（如svm_optimize函数和主函数）保持不变

if __name__ == "__main__":
    dataset_path = "E:\\毕设\\电路仿真\\仿真结果\\four_op_amp\\原始数据文件"  # 替换为你的数据集路径
    X, y, label_map = load_data_with_cross_wavelet(dataset_path)


    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 设定C和gamma参数的搜索范围
    lb = [0.001, 0.0001]  # 参数下界
    ub = [1000, 1]  # 参数上界
    print("Optimizing SVM parameters...")
    # 使用PSO找到最优参数
    xopt, fopt = pso(lambda x: svm_optimize(x, X_train, y_train), lb, ub, maxiter=100)

    print(f"Optimal parameters found: C={xopt[0]}, gamma={xopt[1]} with score={-fopt}")

    # 使用最优参数训练SVM模型
    optimal_model = SVC(C=xopt[0], gamma=xopt[1])
    optimal_model.fit(X_train, y_train)

    # 使用测试集评估模型
    y_pred = optimal_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
