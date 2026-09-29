import torch
from model import DigitCNN

# 创建模型
model = DigitCNN()

# 模拟一个 Batch：64 张 28×28 的灰度图片
x = torch.randn(64, 1, 28, 28)

# 前向传播
output = model(x)

print("输入形状:", x.shape)
print("输出形状:", output.shape)
print("模型结构:")
print(model)