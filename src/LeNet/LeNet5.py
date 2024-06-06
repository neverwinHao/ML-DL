#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import time

# 模型优化--动态调整学习率
from torch.optim import lr_scheduler

# In[2]:


# 指定当前的GPU设备
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
# 展示高清图
from matplotlib_inline import backend_inline

backend_inline.set_matplotlib_formats('svg')


# In[3]:


# 制作数据集
def data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081)
    ])

    # 下载训练集与测试集
    train_Data = datasets.MNIST(
        root='./dataset',
        train=True,
        download=True,
        transform=transform
    )
    test_Data = datasets.MNIST(
        root='./dataset',
        train=False,
        download=True,
        transform=transform
    )
    return train_Data, test_Data


# In[4]:


train_Data, test_Data = data()
# 批次加载器
train_loader = DataLoader(train_Data, shuffle=True, batch_size=64)
test_loader = DataLoader(test_Data, shuffle=False, batch_size=64)


# >参考网址：https://blog.csdn.net/qq_40714949/article/details/109863595?ops_request_misc=&request_id=&biz_id=102&utm_term=Lenet5%E5%8E%9F%E7%90%86&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-109863595.142^v94^chatsearchT3_1&spm=1018.2226.3001.4187

# In[7]:


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


class LeNet5_ReLU(nn.Module):
    def __init__(self):
        # 搭建神经网络
        super(LeNet5_ReLU, self).__init__()
        self.c1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.RelU = nn.ReLU()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        x = self.c1(x)
        x = self.RelU(self.s2(x))
        x = self.c3(x)
        x = self.RelU(self.s4(x))
        x = self.c5(x)
        x = self.flatten(x)
        x = self.RelU(self.f6(x))
        x = self.out(x)
        return (x)


# #### 为什么c1层的padding=2？
# 查阅资料我发现实际上Minist的输入是28X28的，要是输出想要得到28X28
# 
# $28-5+2*padding+1 = 28$
# 所以得到padding = 2

# In[8]:


model = LeNet5().to(device)
# model = LeNet5_ReLU().to(device)
# 损失函数
lossfunc = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 模型优化 -- 动态调整学习率
# 每隔20步学习率衰减为原来的0.5
# lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# In[9]:


def train(model, lossfunc, optimizer, train_dataloader, test_dataloader):
    train_loss = 0.0
    train_acc = 0.0
    count = 0
    start = time.time()

    for _, (X, y) in enumerate(train_dataloader):
        # 每次提取到 X，y 都先转到 GPU
        (X, y) = X.to(device), y.to(device)

        # 计算当前批次的输出
        y_hat = model(X)

        # 计算当前批次的损失值
        loss_batch = lossfunc(y_hat, y)

        # 计算预测值
        _, y_pred = torch.max(y_hat, axis=1)

        # 计算当前批次的准确率
        acc_batch = torch.sum(y_pred == y) / y_hat.shape[0]

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        train_loss += loss_batch.item()
        train_acc += acc_batch.item()
        count += 1

    end = time.time()
    train_loss = train_loss / count
    train_acc = train_acc * 100 / count
    print("训练误差：", train_loss)
    print("训练精度：", train_acc)
    print(f'训练时间：{end - start}秒')

    # 计算在测试集上的精度
    model.eval()
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for _, (X, y) in enumerate(test_dataloader):
            (X, y) = X.to(device), y.to(device)

            y_hat = model(X)
            _, y_pred = torch.max(y_hat, axis=1)

            correct_predictions += torch.sum(y_pred == y)
            total_samples += y.shape[0]

    test_accuracy = correct_predictions.item() / total_samples * 100
    print("测试精度：", test_accuracy)

    return train_loss, test_accuracy


# In[10]:


# 开始训练
epoch = 120
min_acc = 0
loss_list = []
for i in range(epoch):
    print(f'第{i + 1}轮训练\n----------------------------')
    train_loss, test_accuracy = train(model, lossfunc, optimizer, train_loader, test_loader)
    loss_list.append(train_loss)

    # 保存最优模型
    if test_accuracy > min_acc:
        min_acc = test_accuracy
        torch.save(model.state_dict(), "models/Mymodel.pth")
        print('最佳模型保存成功！')

plt.plot(loss_list)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print("最优模型准确率：", min_acc)
print("训练完成!")
