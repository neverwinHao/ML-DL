#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# 指定当前的GPU设备
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


# 定义LeNet5模型
class LeNet5(nn.Module):
    def __init__(self):
        # 搭建神经网络
        super(LeNet5, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.Sigmoid(self.s2(x))
        x = self.c3(x)
        x = self.Sigmoid(self.s4(x))
        x = self.c5(x)
        x = self.flatten(x)
        x = self.Sigmoid(self.f6(x))
        x = self.out(x)
        return (x)


# 加载已经训练好的模型
model = LeNet5()
model.load_state_dict(torch.load("models/Mymodel.pth"))
model.to(device)
model.eval()

# 加载数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])

test_Data = datasets.MNIST(
    root='./dataset',
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_Data, shuffle=False, batch_size=64)


# 定义测试函数
def test(model, test_num, test_loader):
    count = 0  # 记录已处理的样本数量

    for i, (X, y) in enumerate(test_loader):
        if count >= test_num:
            break

        batch_size = X.size(0)  # 当前批次中的样本数量
        for j in range(batch_size):
            img_true, label = X[j][0].numpy(), y[j].item()
            X_batch = Variable(X.to(device))

            with torch.no_grad():
                pred = model(X_batch)
                y_pred = torch.argmax(pred[j]).item()
            print("-----------验证模型性能开始------------")
            print("预测结果：", y_pred)
            print("真实标签：", label)
            plt.imshow(img_true, cmap='gray')
            plt.show()

            count += 1
            if count >= test_num:
                break




# 测试自己手写的数字：

from PIL import Image, ImageOps
import os

path = './MyNumber'

# 定义图像处理和归一化的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1037,), (0.3081,))
])

if __name__ == '__main__':
    # 调用测试函数
    test(model, 5, test_loader)

    # 对自己的手写数字处理部分
    output_folder = 'MyNumber_pred'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(path):
        # 构建文件的完整路径
        file_path = os.path.join(path, filename)

        # 打开图像并进行处理
        image = Image.open(file_path)
        image = image.resize((28, 28))
        image = image.convert('L')
        image = ImageOps.invert(image)  # 黑白反转

        processed_image = transform(image)
        input_tensor = torch.unsqueeze(processed_image, dim=0)

        # 利用模型进行预测
        with torch.no_grad():
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs.data, 1)
            prediction = predicted.item()

        # 展示图像和预测结果
        plt.imshow(image, cmap='gray')
        plt.title(f"Prediction: {prediction}")
        plt.savefig(os.path.join(output_folder, f"{filename[0]}预测结果.png"))
        plt.close()


