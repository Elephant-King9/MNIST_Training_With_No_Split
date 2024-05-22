import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, root_path, transform=None):
        # 记录成员变量
        self.root_path = root_path
        self.transform = transform

        # 将图片与标签列为list
        self.imgs_path = []
        self.labels = []

        # 获取root_path下的所有图片
        for label_path in os.listdir(root_path):
            img_path = os.path.join(root_path, label_path)
            if os.path.isdir(img_path):
                for img_name in os.listdir(img_path):
                    pre_img_path = os.path.join(img_path, img_name)
                    self.imgs_path.append(pre_img_path)
                    self.labels.append(label_path)
        # 获取随机的random参数
        random.seed(random.seed)
        # 创建完整的数据集，内容为将图片和labels一一对应并且列为list
        data = list(zip(self.imgs_path, self.labels))
        # print(data)
        # 将数据集打乱
        random.shuffle(data)

        # 设置训练集数据集划分比例
        spilt_size = 0.8
        split = int(len(data) * spilt_size)
        # 根据比例划分训练集与数据集
        self.train_data = data[:split]
        self.test_data = data[split:]

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = self.labels[index]
        # 根据图片地址获取图片信息，并且转化为灰度图像
        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform()
        return img, label

    def spilt_dataset(self):
        train_data = Subset(self, self.train_data, transform=self.transform)
        test_data = Subset(self, self.test_data, transform=self.transform)
        return train_data, test_data


class Subset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img_path = self.indices[index][0]
        label = self.indices[index][1]
        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)
        label = torch.tensor(label)
        return img, label


if __name__ == '__main__':
    root_dir = '../datasets/mnist_png_with_no_spilt'
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Grayscale()
    ])
    dataset = MyDataset(root_dir, transform=transform)
    train_dataset, test_dataset = dataset.spilt_dataset()
    print(f'dataset size = {len(dataset)}, train size = {len(train_dataset)}, test size = {len(test_dataset)}')
    print(f'train_dataset[0]:\n{train_dataset[0]}')
