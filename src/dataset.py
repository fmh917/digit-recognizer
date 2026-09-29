from torch.utils.data import Dataset
#定义数据集类
class DigitDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    # 返回数据集大小
    def __len__(self):
        return len(self.images)

    # 根据索引返回图片和标签
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label