import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

from MNISTmodel import *
from dataset import *

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),

])
root_dir = '../datasets/mnist_png_with_no_spilt'

my_dataset = MyDataset(root_dir, transform=transform)
train_dataset, test_dataset = my_dataset.spilt_dataset()


# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

writer = SummaryWriter('../logs')

model = MNISTModel().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
learn_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

epoch = 10
total_train_step = 0
start_time = time.time()
for i in range(epoch):
    pre_train_step = 0
    pre_train_loss = 0
    model.train()
    for data in train_loader:
        # print(data)
        images, labels = data
        # print({type(labels)})
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        pre_train_step += 1
        total_train_step += 1
        pre_train_loss += loss.item()
        if pre_train_step % 100 == 0:
            end_time = time.time()
            print(f'Epoch:{i + 1},pre_train_loss:{pre_train_loss / pre_train_step},time = {end_time - start_time}')
            writer.add_scalar('train_loss', pre_train_loss / pre_train_step, total_train_step)

    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_accuracy += outputs.argmax(1).eq(labels).sum().item()

    print(f'Epoch:{i + 1},Test Accuracy: {total_accuracy / len(test_dataset)}')
    writer.add_scalar('test_accuracy', total_accuracy / len(test_dataset), i)

    torch.save(model, f'../models/MNISTModel_{i}.pth')

writer.close()
