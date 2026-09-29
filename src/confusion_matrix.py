import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import DigitCNN
from data_loader import val_loader
import numpy as np

# 1. 选择运行设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. 加载模型
model = DigitCNN().to(device)
model.load_state_dict(
    torch.load(
        "best_model_dropout.pth",
        map_location=device,
        weights_only=True
    )
)
model.eval()

# 3. 保存真实标签和预测结果
all_labels = []
all_predictions = []

# 4. 关闭梯度计算
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.cpu().numpy())

# 5. 计算混淆矩阵
cm = confusion_matrix(
    all_labels,
    all_predictions
)

# 6. 绘制混淆矩阵
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10))
)

display.plot(cmap="Blues", values_format="d")
plt.title("MNIST Confusion Matrix")
plt.show()

# 每一行的样本总数
row_sums = cm.sum(axis=1)
# 对角线：预测正确的数量
correct = np.diag(cm)
# 每个数字的识别率
class_accuracy = correct / row_sums
print("\n各数字识别率：")

for digit, accuracy in enumerate(class_accuracy):
    print(f"数字 {digit}: {accuracy:.2%}")