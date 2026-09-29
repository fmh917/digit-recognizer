import torch
import matplotlib.pyplot as plt
from model import DigitCNN
from data_loader import val_loader

# 1. 选择设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("使用设备:", device)

# 2. 创建模型
model = DigitCNN().to(device)

# 3. 加载训练好的模型
model.load_state_dict(
    torch.load(
        "best_model_dropout.pth",
        map_location=device,
        weights_only=True
    )
)

# 切换到评估模式
model.eval()
print("模型加载成功！")

# 4. 获取一批验证集图片
images, labels = next(iter(val_loader))

# 选择前 12 张图片
images = images[:12].to(device)
labels = labels[:12]

# 5. 使用模型进行预测
with torch.no_grad():
    outputs = model(images)
    predictions = torch.argmax(outputs, dim=1)
# 将预测结果移回 CPU
predictions = predictions.cpu()

# 6. 绘制图片
plt.figure(figsize=(12, 8))
for i in range(12):

    # 创建 3 行 4 列的子图
    plt.subplot(3, 4, i + 1)

    # 将图片转换为二维数组
    image = images[i].cpu().squeeze().numpy()

    # 显示图片
    plt.imshow(image, cmap="gray")

    # 获取真实标签和预测标签
    true_label = labels[i].item()
    pred_label = predictions[i].item()

    # 判断预测是否正确
    if true_label == pred_label:
        color = "green"
    else:
        color = "red"

    # 设置标题
    plt.title(
        f"True: {true_label}, Pred: {pred_label}",
        color=color
    )

    # 隐藏坐标轴
    plt.axis("off")

plt.tight_layout()
plt.savefig("prediction_results.png")
plt.show()