import torch
import pandas as pd
import numpy as np
from pathlib import Path
from model import DigitCNN
# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)
# 2. 获取项目根目录
project_root = Path(__file__).resolve().parent.parent

# 3. 读取官方测试集
test_path = project_root / "data" / "test.csv"
test_data = pd.read_csv(test_path)
print("测试集形状:", test_data.shape)

# 4. 数据预处理
# 将像素值从 0~255 归一化到 0~1
X_test = test_data.values.astype(np.float32) / 255.0

# 将每张图片转换为 (1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)

# 转换为 PyTorch Tensor
X_test = torch.tensor(X_test, dtype=torch.float32)

# 5. 创建模型
model = DigitCNN().to(device)

# 加载训练好的最佳模型
model_path = project_root / "best_model_dropout.pth"
model.load_state_dict(
    torch.load(model_path, map_location=device)
)

# 切换到评估模式
model.eval()

# 6. 对测试集进行预测
predictions = []
with torch.no_grad():
    # 分批预测，避免一次性加载过多数据
    batch_size = 256
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size].to(device)
        outputs = model(batch)
        # 取每张图片得分最高的类别
        predicted = torch.argmax(outputs, dim=1)
        predictions.extend(predicted.cpu().numpy())

# 7. 生成CSV文件
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(predictions) + 1),
    "Label": predictions
})

# 8. 保存文件到 data 目录
output_path = project_root / "data" / "sample_submission.csv"
submission.to_csv(output_path, index=False)
print("预测完成！")
print("结果文件形状:", submission.shape)
print(submission.head())