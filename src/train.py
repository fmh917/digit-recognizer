
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed) #固定python内置的随机数生成器的种子
    np.random.seed(seed) #固定numpy库的随机数生成器的种子
    torch.manual_seed(seed) #固定torch库的随机数生成器的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

from model import DigitCNN
from data_loader import train_loader, val_loader

# 1. 选择训练设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("使用设备:", device)

# 2. 创建模型
model = DigitCNN().to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 4. 设置训练轮数
epochs = 100
best_acc = 0.0
patience = 5
early_stop_counter = 0

# 5. 创建列表记录训练过程
train_losses = []   #每轮训练结束时的平均Loss
val_accuracies = [] #每轮验证结束时的准确率

# 6. 开始训练
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)
        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # 7. 验证模型
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # 找到每张图片分数最高的类别
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total

    # 8. 输出本轮结果

    # 保存当前 Epoch 的平均 Loss
    train_losses.append(avg_loss)
    # 保存当前 Epoch 的验证准确率
    val_accuracies.append(val_acc)
    # 打印训练日志
    print(
        f"Epoch [{epoch+1}/{epochs}], "
        f"Loss: {avg_loss:.4f}, "
        f"Val Acc: {val_acc:.4f}"
    )

    # 9. 保存验证准确率最高的模型
    if val_acc > best_acc:
        best_acc = val_acc
        early_stop_counter = 0
        torch.save(
            model.state_dict(),
            "best_model_dropout.pth"
        )
        print("已保存当前最佳模型！")
    else:
        early_stop_counter += 1
        print(
            f"验证准确率未提升，"
            f"连续 {early_stop_counter} 轮未改善"
        )
    # 判断是否触发早停
    if early_stop_counter >= patience:
        print("触发 Early Stopping，停止训练！")
        break

print(f"最佳验证准确率: {best_acc:.4f}")


plt.figure(figsize=(12, 5))
# 绘制训练 Loss
plt.subplot(1, 2, 1)
epoch_numbers = range(1, len(train_losses) + 1)
plt.plot(epoch_numbers,train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

# 绘制验证准确率
plt.subplot(1, 2, 2)
plt.plot(epoch_numbers,val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves.png")
plt.show()