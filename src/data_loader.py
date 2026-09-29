import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from dataset import DigitDataset
from torch.utils.data import DataLoader

# 1. 读取训练数据(DataFrame)
train_data = pd.read_csv("data/train.csv")

# 2. 查看数据形状(42000, 785)，其中第一列是标签，后面784列是像素值
print("训练集形状:", train_data.shape)
#.shape属性返回一个包含数据集行数和列数的元组，其中shape[0]表示数据集的行数(样本数量)，shape[1]表示数据集的列数(特征数量)。

# 3. 查看前五行
print(train_data.head())

# 4. 提取标签
labels = train_data["label"].values

# 5. 提取像素数据
images = train_data.drop("label", axis=1).values
#drop()函数用于删除指定的列或行，axis=1表示删除列，axis=0表示删除行

# 6. 数据归一化，将像素值缩放到0到1之间
images = images / 255.0

# 7. 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42,stratify=labels
)
# 7. 将图片转换成 CNN 输入格式
X_train = X_train.reshape(-1, 1, 28, 28)
X_val = X_val.reshape(-1, 1, 28, 28)

# 8. 将 NumPy 数组转换成 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# 创建训练集
train_dataset = DigitDataset(X_train, y_train)

# 创建验证集
val_dataset = DigitDataset(X_val, y_val)

# 创建训练集 DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# 创建验证集 DataLoader
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

# 查看数据集信息
print("训练集样本数:", len(train_dataset))
print("训练集 Batch 数量:", len(train_loader))

# 取出第一个 Batch
images_batch, labels_batch = next(iter(train_loader))

print("Batch 图片形状:", images_batch.shape)
print("Batch 标签形状:", labels_batch.shape)

# 9. 查看数据形状
print("训练集图片:", X_train.shape)
print("验证集图片:", X_val.shape)

print("训练集标签:", y_train.shape)
print("验证集标签:", y_val.shape)

# 10. 查看数据类型
print("图片数据类型:", X_train.dtype)
print("标签数据类型:", y_train.dtype)

# 11. 显示第一张图片
if __name__ == "__main__":
    plt.imshow(X_train[0, 0].numpy(), cmap="gray")
    plt.title(f"Label: {y_train[0].item()}")
    plt.axis("off")
    plt.show()