#from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from pyswarm import pso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
#from tqdm import tqdm
import time
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def svm_optimize(x, X_train, y_train):
    C, gamma = x
    model = SVC(C=C, gamma=gamma)
    # 减少交叉验证的拆分数，适应样本数量较少的情况
    scores = cross_val_score(model, X_train, y_train, cv=min(5, len(np.unique(y_train))))
    return -scores.mean()

def load_data(dataset_path, fixed_length=100):
    X = []
    y = []
    label_map = {}

    for label, subdir in enumerate(sorted(os.listdir(dataset_path))):
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        label_map[subdir] = label  # 注意这里的修改，使标签映射更直观

        for file in sorted(os.listdir(subdir_path)):
            if file.endswith('.csv'):
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path)
                # 假设目标列名为 '(V(OUT))@1'
                target_column = df.columns[1]  # 假设目标列总是第二列
                features = df[target_column].values[:fixed_length]
                if len(features) < fixed_length:
                    features = np.pad(features, (0, fixed_length - len(features)), 'constant')
                X.append(features)
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, label_map


# 使用tqdm显示PSO优化进度的函数
def optimize_without_progress(X_train, y_train, lb, ub, maxiter=100):
    print("Starting optimization...")
    xopt, fopt = pso(svm_optimize, lb, ub, args=(X_train, y_train), maxiter=maxiter)
    print("Optimization finished.")
    return xopt, fopt

if __name__ == "__main__":
    # 加载数据
    dataset_path = "E:\\毕设\\电路仿真\\仿真结果\\four_op_amp\\原始数据文件"  # 替换为你的数据集路径
    X, y, label_map = load_data(dataset_path)

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 定义SVM模型的超参数优化目标函数

    # 设定C和gamma参数的搜索范围
    lb = [0.001, 0.0001]  # 参数下界
    ub = [500, 1]  # 参数上界


    start_time = time.time()
    print("Optimizing SVM parameters...")
    # 使用PSO找到最优参数
    # xopt, fopt = pso(svm_optimize, lb, ub, args=(X_train, y_train))
    xopt, fopt = optimize_without_progress(X_train, y_train, lb, ub, maxiter=20)
    print(f"Optimal parameters found: C={xopt[0]}, gamma={xopt[1]} with score={-fopt}")

    # 使用最优参数训练SVM模型
    optimal_model = SVC(C=xopt[0], gamma=xopt[1])
    optimal_model.fit(X_train, y_train)
    print("Model training completed.")

    end_time=time.time()
    time=end_time-start_time
    print("Training time:",time)

    # 使用测试集评估模型
    y_pred = optimal_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
    print(f"Recall on test set: {recall:.2f}")
    print(f"F1 Score on test set: {f1:.2f}")
    print("Confusion Matrix on test set:\n", confusion)


    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
