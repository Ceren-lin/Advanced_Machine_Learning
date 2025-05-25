from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

def load_data(dataset_path, fixed_length=100):
    X = []
    y = []
    label_map = {}

    for label, subdir in enumerate(sorted(os.listdir(dataset_path))):
        subdir_path = os.path.join(dataset_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        label_map[subdir] = label

        for file in sorted(os.listdir(subdir_path)):
            if file.endswith('.csv'):
                file_path = os.path.join(subdir_path, file)
                df = pd.read_csv(file_path)
                target_column = df.columns[1]  # Assuming the target column is always the second one
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

if __name__ == "__main__":
    # Load the data
    dataset_path = "E:\\毕设\\电路仿真\\仿真结果\\four_op_amp\\原始数据文件"  # Replace with your dataset path
    X, y, label_map = load_data(dataset_path)

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators is the number of trees in the forest

    # Train the model
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy * 100:.2f}%")
