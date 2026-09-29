
import torch
import torch.nn as nn
from model import DigitCNN
from data_loader import val_loader

# 1. 选择设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. 创建模型结构
model = DigitCNN().to(device)

# 3. 加载训练好的参数
model.load_state_dict(
    torch.load(
        "best_model_dropout.pth",
        map_location=device,
        weights_only=True
    )
)

# 4. 切换到评估模式
model.eval()

# 5. 计算验证集准确率
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"验证集准确率: {accuracy:.4f}")
print(f"正确预测数量: {correct}/{total}")

# 取出一个 Batch
images, labels = next(iter(val_loader))
images = images.to(device)
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 显示前 10 张图片的预测结果
for i in range(10):
    print(
        f"真实标签: {labels[i].item()}, "
        f"预测结果: {predicted[i].item()}"
    )